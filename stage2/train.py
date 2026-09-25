import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .ablations import ablation_names, describe_ablation, make_ablation_config
from .checkpoint import load_stage2_weights, load_training_state, save_checkpoint
from .config import EXPECTED_WINDOWS, SOURCE_CITY_ORDER, Stage2Config
from .data import JointMultiCityDataset
from .distributed import barrier, cleanup, init_distributed, is_distributed, seed_everything, world_size
from .evaluation import evaluate_model, warmup_evaluation
from .exact_accum import exact_nash_accumulated_update
from .metrics import format_metrics, save_metrics
from .model import Stage2UrbanForecaster
from .nash import DistributedNashMTL, EqualWeightSharedBargaining
from .normalization import TorchFlowNormalizer
from .optim import build_cosine_scheduler, build_optimizer
from .prompt import Stage2Collator
from .reporting import architecture_note, save_note
from .runtime import peak_memory_across_ranks, reset_peak_memory, runtime_definition, save_runtime_summary, start_wall_timer, stop_wall_timer

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', default='urbanbind_stage2_v3_2_split_nash_fullgrad')
    ap.add_argument('--ablation', choices=ablation_names(include_full=True), default='full')
    ap.add_argument('--micro-batch-size', type=int, default=None)
    ap.add_argument('--grad-accum', type=int, default=None)
    ap.add_argument('--max-epochs', type=int, default=None)
    ap.add_argument('--stop-after-epoch', type=int, default=None, help='Hard execution budget in completed epochs. Unlike --max-epochs, this does not change the cosine-scheduler horizon.')
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--num-workers', type=int, default=None)
    ap.add_argument('--resume', default=None, help='Stage-2 checkpoint directory to resume')
    ap.add_argument('--max-updates', type=int, default=None, help='For controlled dry runs only')
    ap.add_argument('--dry-valid-steps', type=int, default=8)
    ap.add_argument('--selection-metric', choices=['normalized_l1_macro', 'raw_macro_mae'], default='raw_macro_mae', help='Checkpoint selection only. Final paper metrics are always raw-unit per-city/channel. normalized_l1_macro matches the normalized Stage-2 optimization scale.')
    return ap.parse_args()

def _cfg_from_args(args):
    cfg = make_ablation_config(args.ablation)
    changes = {}
    if args.micro_batch_size is not None:
        changes['micro_batch_size'] = args.micro_batch_size
    if args.grad_accum is not None:
        changes['gradient_accumulation_steps'] = args.grad_accum
    if args.max_epochs is not None:
        changes['max_epochs'] = args.max_epochs
    if args.patience is not None:
        changes['early_stopping_patience'] = args.patience
    if args.num_workers is not None:
        changes['num_workers'] = args.num_workers
    return replace(cfg, **changes) if changes else cfg

def _selection_value(metrics, name):
    if name == 'normalized_l1_macro':
        return float(metrics['normalized_l1_macro'])
    if name == 'raw_macro_mae':
        m = metrics['macro_city']
        return 0.5 * (float(m['inflow_mae']) + float(m['outflow_mae']))
    raise ValueError(name)

def _mean_across_ranks(x):
    y = x.detach().float().clone()
    if is_distributed():
        torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
        y.div_(world_size())
    return y

def _max_across_ranks(value, device):
    x = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if is_distributed():
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.MAX)
    return float(x.cpu())

def _trainable_parameters(model):
    m = model.module if hasattr(model, 'module') else model
    return [p for p in m.parameters() if p.requires_grad]

def main():
    args = _parse_args()
    cfg = _cfg_from_args(args)
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps must be >= 1')
    if cfg.stage1_source not in ('grounded', 'base_pretrained'):
        raise ValueError(cfg.stage1_source)
    if cfg.semantic_mode not in ('dsr_reentry', 'direct_concat'):
        raise ValueError(cfg.semantic_mode)
    if cfg.weighting_mode not in ('nash', 'equal'):
        raise ValueError(cfg.weighting_mode)
    r, w, local_rank = init_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    seed_everything(cfg.seed, rank_offset=r)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    run_out = cfg.run_output_dir(args.run_name)
    run_ckpt = cfg.run_checkpoint_dir(args.run_name)
    run_log = cfg.run_log_dir(args.run_name)
    if r == 0:
        run_out.mkdir(parents=True, exist_ok=True)
        run_ckpt.mkdir(parents=True, exist_ok=True)
        run_log.mkdir(parents=True, exist_ok=True)
    barrier()
    normalizer = TorchFlowNormalizer(cfg.normalization_stats)
    base_model = Stage2UrbanForecaster(cfg, device=device)
    if base_model.bridge.trainable_parameter_count != 0:
        raise RuntimeError('Stage-1 VLM is not fully frozen')
    effective_global_batch = w * cfg.micro_batch_size * cfg.gradient_accumulation_steps
    note_extra = [f'Ablation profile: {cfg.ablation_name}', 'Ablation definition: ' + describe_ablation(cfg.ablation_name), f'Stage-1 source: {cfg.stage1_source}', f'Semantic mode: {cfg.semantic_mode}', f'Shared weighting mode: {cfg.weighting_mode}', f'DDP world size: {w}', f'Micro batch / GPU: {cfg.micro_batch_size}', f'Exact Nash accumulation micro-steps: {cfg.gradient_accumulation_steps}', f'Effective nominal global batch: {effective_global_batch}', f'Nash alpha refresh period: every {cfg.nash_update_every} optimizer updates; cached relative weights are reused in between.', f'Nash mean-one rescaling: {cfg.nash_normalize_mean}; shared alpha floor={cfg.nash_alpha_floor:g}.', 'Gradient routing: Nash weights shared modules ONLY; each city-specific encoder/head uses its own city loss at coefficient 1.', 'Gradient clipping is route-separated: shared once, and each city-private branch independently.', 'Frozen VLM parameters; semantic re-entry is fully differentiable wrt Z (dR/dZ retained); direct Z residual also remains trainable.', f'Manual routed-gradient sync: deterministic NCCL buckets <= {cfg.gradient_sync_bucket_mb:g} MiB (transport only; exact global mean preserved).', 'optimizer.step/scheduler.step occur exactly once per effective-batch update.', f'Schedule cap: {cfg.max_epochs} epochs', f'Early stopping patience: {cfg.early_stopping_patience}', f'Checkpoint selection metric: {args.selection_metric}']
    note = architecture_note(base_model, cfg, note_extra)
    if r == 0:
        print(note, flush=True)
        save_note(note, run_log / 'architecture_note.txt')
        (run_log / 'config.json').write_text(json.dumps(cfg.to_dict(), indent=2), encoding='utf-8')
    model = base_model
    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
    train_ds = JointMultiCityDataset(split='train', normalizer=normalizer, visual_cache_root=str(cfg.visual_cache_root), seed=cfg.seed, training=True)
    train_sampler = DistributedSampler(train_ds, num_replicas=w, rank=r, shuffle=False, drop_last=False) if is_distributed() else None
    collator = Stage2Collator(base_model.bridge.processor, max_seq_length=int(base_model.bridge.stage1_cfg.get('max_seq_length', 8192)))
    loader = DataLoader(train_ds, batch_size=cfg.micro_batch_size, shuffle=False, sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=False, collate_fn=collator, persistent_workers=False)
    updates_per_epoch = math.ceil(len(loader) / cfg.gradient_accumulation_steps)
    total_updates = updates_per_epoch * cfg.max_epochs
    execution_update_cap = int(args.max_updates) if args.max_updates is not None else None
    optimizer = build_optimizer(model, cfg)
    scheduler, warmup_updates = build_cosine_scheduler(optimizer, total_updates, cfg.warmup_ratio)
    if cfg.weighting_mode == 'nash':
        nash = DistributedNashMTL(n_tasks=len(SOURCE_CITY_ORDER), update_weights_every=cfg.nash_update_every, normalize_mean=cfg.nash_normalize_mean, optim_niter=cfg.nash_optim_niter, solver_max_iters=cfg.nash_solver_max_iters, eps=cfg.nash_eps, alpha_floor=cfg.nash_alpha_floor)
    elif cfg.weighting_mode == 'equal':
        nash = EqualWeightSharedBargaining(n_tasks=len(SOURCE_CITY_ORDER))
    else:
        raise ValueError(cfg.weighting_mode)
    start_epoch = 0
    start_batch = 0
    global_update = 0
    best_value = float('inf')
    best_epoch = -1
    bad_epochs = 0
    if args.resume:
        resume_meta = Path(args.resume) / 'metadata.json'
        if resume_meta.exists():
            saved_meta = json.loads(resume_meta.read_text(encoding='utf-8'))
            saved_arch = saved_meta.get('config', {}).get('architecture_version')
            current_arch = cfg.to_dict()['architecture_version']
            if saved_arch != current_arch:
                raise RuntimeError(f'Refusing cross-version Stage-2 resume: saved={saved_arch!r}, current={current_arch!r}. Split-Nash/full-gradient v3.2 must start fresh from frozen Stage-1.')
        load_stage2_weights(args.resume, model)
        progress = load_training_state(args.resume, optimizer, scheduler, nash)
        start_epoch = int(progress.get('epoch', 0))
        start_batch = int(progress.get('next_batch', 0))
        global_update = int(progress.get('global_update', 0))
        best_value = float(progress.get('best_value', float('inf')))
        best_epoch = int(progress.get('best_epoch', -1))
        bad_epochs = int(progress.get('bad_epochs', 0))
        if r == 0:
            print(f'RESUME: {args.resume}; epoch={start_epoch}; next_batch={start_batch}; update={global_update}', flush=True)
    if r == 0:
        print(f'train joint steps={len(train_ds)} | per-rank loader batches={len(loader)} | updates/epoch={updates_per_epoch} | total planned updates={total_updates} | warmup={warmup_updates} | effective_global_batch={effective_global_batch} | execution_update_cap={execution_update_cap}', flush=True)
    reached_update_cap = False
    trainable_params = _trainable_parameters(model)
    runtime_path = run_out / 'runtime_summary.json'
    runtime = {'run_name': args.run_name, 'architecture_version': cfg.to_dict()['architecture_version'], 'world_size': int(w), 'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu', 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda, 'micro_batch_size_per_gpu': int(cfg.micro_batch_size), 'gradient_accumulation_steps': int(cfg.gradient_accumulation_steps), 'nominal_effective_global_batch': int(effective_global_batch), 'completed_epochs': 0, 'timed_optimizer_updates': 0, 'training_wall_seconds': 0.0, 'training_only_seconds': 0.0, 'validation_seconds': 0.0, 'mean_epoch_wall_seconds': None, 'mean_train_seconds_per_update': None, 'epochs': [], 'definitions': runtime_definition()}
    if args.resume and runtime_path.exists():
        try:
            previous_runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
            for key in ('completed_epochs', 'timed_optimizer_updates', 'training_wall_seconds', 'training_only_seconds', 'validation_seconds', 'epochs'):
                if key in previous_runtime:
                    runtime[key] = previous_runtime[key]
        except Exception as exc:
            if r == 0:
                print(f'WARNING: could not restore runtime summary: {exc}', flush=True)
    reset_peak_memory(device)
    try:
        for epoch in range(start_epoch, cfg.max_epochs):
            train_ds.set_epoch(epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pending_batches = []
            pending_indices = []
            epoch_loss = 0.0
            epoch_batches = 0
            epoch_wall_t0 = start_wall_timer(device)
            train_t0 = time.perf_counter()
            epoch_start_update = global_update
            last_consumed_batch = start_batch - 1 if epoch == start_epoch else -1
            for batch_idx, cpu_batch in enumerate(loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue
                pending_batches.append(cpu_batch)
                pending_indices.append(batch_idx)
                is_last_batch = batch_idx == len(loader) - 1
                group_ready = len(pending_batches) >= cfg.gradient_accumulation_steps or is_last_batch
                if not group_ready:
                    continue
                report = exact_nash_accumulated_update(model=model, cpu_batches=pending_batches, device=device, nash=nash, optimizer=optimizer, scheduler=scheduler, trainable_params=trainable_params, max_grad_norm=cfg.max_grad_norm)
                if report.optimizer_steps != 1 or report.scheduler_steps != 1:
                    raise RuntimeError('Exact Nash transaction violated: optimizer/scheduler updated more or less than once.')
                if report.nash_solves not in (0, 1):
                    raise RuntimeError('Periodic Nash transaction may solve alpha at most once')
                if report.used_nash_probe != (report.nash_solves == 1):
                    raise RuntimeError('Nash probe/solve accounting mismatch')
                group_micro_steps = report.micro_steps
                last_consumed_batch = pending_indices[-1]
                epoch_batches += group_micro_steps
                epoch_loss += float(report.weighted_loss_local_mean.cpu()) * group_micro_steps
                global_update += 1
                if global_update % cfg.log_every_updates == 0 or global_update == 1:
                    city_loss_mean = _mean_across_ranks(report.update_city_losses_local_mean)
                    loss_mean = _mean_across_ranks(report.weighted_loss_local_mean)
                    private_loss_mean = _mean_across_ranks(report.private_loss_local_mean)
                    replay_diff_max = _max_across_ranks(report.max_probe_update_city_loss_abs_diff, device)
                    if r == 0:
                        lrs = {g.get('tag', str(i)): g['lr'] for i, g in enumerate(optimizer.param_groups)}
                        info = {'epoch': epoch + 1, 'update': global_update, 'micro_steps_in_update': group_micro_steps, 'effective_global_batch_this_update': w * cfg.micro_batch_size * group_micro_steps, 'weighted_loss': float(loss_mean.cpu()), 'city_losses': {c: float(city_loss_mean[i].cpu()) for i, c in enumerate(SOURCE_CITY_ORDER)}, 'nash_alpha': {c: float(report.alpha[i].detach().float().cpu()) for i, c in enumerate(SOURCE_CITY_ORDER)}, 'grad_norm': float(report.grad_norm.detach().float().cpu()), 'shared_grad_norm': float(report.shared_grad_norm.detach().float().cpu()), 'private_grad_norms': {c: float(report.private_grad_norms[i].detach().float().cpu()) for i, c in enumerate(SOURCE_CITY_ORDER)}, 'shared_weighted_loss': float(loss_mean.cpu()), 'private_unweighted_loss': float(private_loss_mean.cpu()), 'gradient_routing': report.routing_mode, 'probe_update_city_loss_max_abs_diff': replay_diff_max, 'nash_solves': report.nash_solves, 'used_nash_probe': report.used_nash_probe, 'optimizer_steps': report.optimizer_steps, 'scheduler_steps': report.scheduler_steps, 'lrs': lrs}
                        print('TRAIN ' + json.dumps(info), flush=True)
                if global_update % cfg.checkpoint_every_updates == 0:
                    progress = {'epoch': epoch, 'next_batch': last_consumed_batch + 1, 'global_update': global_update, 'best_value': best_value, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'exact_nash_accumulation': True}
                    save_checkpoint(run_ckpt / 'latest', model, optimizer, scheduler, nash, cfg, progress)
                pending_batches = []
                pending_indices = []
                if args.max_updates is not None and global_update >= args.max_updates:
                    reached_update_cap = True
                    break
            if pending_batches:
                raise RuntimeError('Internal error: uncommitted accumulation group at epoch boundary')
            train_seconds = stop_wall_timer(train_t0, device)
            updates_this_epoch = global_update - epoch_start_update
            if reached_update_cap:
                dry_valid_t0 = start_wall_timer(device)
                dry_metrics = evaluate_model(model, cfg, normalizer, 'valid', device, batch_size=cfg.micro_batch_size, max_joint_steps=args.dry_valid_steps)
                dry_valid_seconds = stop_wall_timer(dry_valid_t0, device)
                if r == 0:
                    print('DRY VALIDATION (subset; not paper result)')
                    print(format_metrics(dry_metrics), flush=True)
                progress = {'epoch': epoch, 'next_batch': min(len(loader), last_consumed_batch + 1), 'global_update': global_update, 'best_value': best_value, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'dry_run': True, 'exact_nash_accumulation': True}
                save_checkpoint(run_ckpt / 'dry_latest', model, optimizer, scheduler, nash, cfg, progress, metrics=dry_metrics)
                dry_wall_seconds = stop_wall_timer(epoch_wall_t0, device)
                dry_mem = peak_memory_across_ranks(device)
                if r == 0:
                    save_metrics(dry_metrics, run_out / 'dry_valid_metrics.json', run_out / 'dry_valid_metrics.txt')
                    report_out = {'global_updates': global_update, 'elapsed_seconds': dry_wall_seconds, 'training_only_seconds': train_seconds, 'dry_validation_seconds': dry_valid_seconds, 'seconds_per_update_observed': train_seconds / max(1, updates_this_epoch), 'micro_batch_per_gpu': cfg.micro_batch_size, 'exact_nash_accumulation_steps': cfg.gradient_accumulation_steps, 'world_size': w, 'nominal_effective_global_batch': effective_global_batch, 'max_peak_allocated_gb_across_ranks': dry_mem['peak_allocated_gb'], 'max_peak_reserved_gb_across_ranks': dry_mem['peak_reserved_gb'], 'optimizer_steps_per_update': 1, 'scheduler_steps_per_update': 1, 'nash_refresh_every_updates': cfg.nash_update_every, 'nash_mean_one_rescaling': cfg.nash_normalize_mean, 'nash_shared_alpha_floor': cfg.nash_alpha_floor, 'gradient_routing': 'shared_nash_private_own_city', 'full_gradient_reentry': True, 'formal_total_planned_updates': total_updates, 'execution_update_cap': execution_update_cap, 'note': 'Exact Nash probe/solve occurs only on refresh updates; cached relative weights skip the probe pass between refreshes. Nash coefficients apply only to shared parameters; each city-private encoder/head uses its own loss with coefficient 1. Every update performs exactly one optimizer/scheduler step.'}
                    (run_log / 'dry_run_report.json').write_text(json.dumps(report_out, indent=2), encoding='utf-8')
                    print(f"CONTROLLED SPLIT-NASH DRY RUN COMPLETE: update={global_update}, elapsed={dry_wall_seconds:.1f}s, train={train_seconds:.1f}s, dry_valid={dry_valid_seconds:.1f}s, max_peak={dry_mem['peak_allocated_gb']:.2f} GB", flush=True)
                break
            valid_t0 = start_wall_timer(device)
            valid_metrics = evaluate_model(model, cfg, normalizer, 'valid', device, batch_size=cfg.micro_batch_size, max_joint_steps=None)
            valid_seconds = stop_wall_timer(valid_t0, device)
            value = _selection_value(valid_metrics, args.selection_metric)
            improved = value < best_value - cfg.early_stopping_min_delta
            if improved:
                best_value = value
                best_epoch = epoch + 1
                bad_epochs = 0
            else:
                bad_epochs += 1
            progress_next = {'epoch': epoch + 1, 'next_batch': 0, 'global_update': global_update, 'best_value': best_value, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs, 'selection_metric': args.selection_metric, 'exact_nash_accumulation': True}
            save_checkpoint(run_ckpt / 'latest', model, optimizer, scheduler, nash, cfg, progress_next, metrics=valid_metrics)
            if improved:
                save_checkpoint(run_ckpt / 'best', model, optimizer, scheduler, nash, cfg, progress_next, metrics=valid_metrics)
            if r == 0:
                save_metrics(valid_metrics, run_out / f'valid_epoch_{epoch + 1:03d}.json', run_out / f'valid_epoch_{epoch + 1:03d}.txt')
            epoch_wall_seconds = stop_wall_timer(epoch_wall_t0, device)
            runtime['training_only_seconds'] = float(runtime['training_only_seconds']) + train_seconds
            runtime['validation_seconds'] = float(runtime['validation_seconds']) + valid_seconds
            runtime['training_wall_seconds'] = float(runtime['training_wall_seconds']) + epoch_wall_seconds
            runtime['timed_optimizer_updates'] = int(runtime['timed_optimizer_updates']) + updates_this_epoch
            runtime['epochs'] = [x for x in runtime['epochs'] if int(x.get('epoch', -1)) != epoch + 1]
            runtime['epochs'].append({'epoch': epoch + 1, 'optimizer_updates': updates_this_epoch, 'train_seconds': train_seconds, 'validation_seconds': valid_seconds, 'epoch_wall_seconds': epoch_wall_seconds, 'other_seconds': max(0.0, epoch_wall_seconds - train_seconds - valid_seconds), 'train_seconds_per_update': train_seconds / max(1, updates_this_epoch), 'selection_value': float(value), 'improved': bool(improved)})
            runtime['completed_epochs'] = len(runtime['epochs'])
            runtime['mean_epoch_wall_seconds'] = float(runtime['training_wall_seconds']) / max(1, int(runtime['completed_epochs']))
            runtime['mean_train_seconds_per_update'] = float(runtime['training_only_seconds']) / max(1, int(runtime['timed_optimizer_updates']))
            train_mem_so_far = peak_memory_across_ranks(device)
            runtime['training_peak_allocated_gb_across_ranks'] = train_mem_so_far['peak_allocated_gb']
            runtime['training_peak_reserved_gb_across_ranks'] = train_mem_so_far['peak_reserved_gb']
            save_runtime_summary(runtime, runtime_path)
            if r == 0:
                print(f'VALID epoch={epoch + 1} selection={value:.8f} best={best_value:.8f} best_epoch={best_epoch}')
                print(format_metrics(valid_metrics), flush=True)
                print(f'epoch_seconds={epoch_wall_seconds:.1f}; train_seconds={train_seconds:.1f}; valid_seconds={valid_seconds:.1f}; sec/update={train_seconds / max(1, updates_this_epoch):.3f}; bad_epochs={bad_epochs}/{cfg.early_stopping_patience}', flush=True)
            start_batch = 0
            if args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch:
                if r == 0:
                    print(f'FIXED EPOCH BUDGET REACHED: {epoch + 1}/{args.stop_after_epoch}', flush=True)
                break
            if bad_epochs >= cfg.early_stopping_patience:
                if r == 0:
                    print('EARLY STOPPING', flush=True)
                break
        if not reached_update_cap:
            barrier()
            load_stage2_weights(run_ckpt / 'best', model)
            warmup_steps = 3
            warmup_seen = warmup_evaluation(model, cfg, normalizer, 'test', device, batch_size=cfg.micro_batch_size, warmup_joint_steps=warmup_steps)
            reset_peak_memory(device)
            test_t0 = start_wall_timer(device)
            test_metrics = evaluate_model(model, cfg, normalizer, 'test', device, batch_size=cfg.micro_batch_size, max_joint_steps=None)
            test_seconds = stop_wall_timer(test_t0, device)
            test_mem = peak_memory_across_ranks(device)
            test_city_windows = int(sum(EXPECTED_WINDOWS['test'].values()))
            test_runtime = {'warmup_joint_steps_requested': warmup_steps, 'warmup_examples_processed_local_rank': int(warmup_seen), 'final_test_seconds': test_seconds, 'test_joint_steps': int(test_metrics['joint_steps']), 'test_city_windows': test_city_windows, 'test_joint_steps_per_second': int(test_metrics['joint_steps']) / max(test_seconds, 1e-12), 'test_city_windows_per_second': test_city_windows / max(test_seconds, 1e-12), 'test_peak_allocated_gb_across_ranks': test_mem['peak_allocated_gb'], 'test_peak_reserved_gb_across_ranks': test_mem['peak_reserved_gb']}
            runtime.update(test_runtime)
            runtime['best_epoch'] = int(best_epoch)
            runtime['best_checkpoint'] = str(run_ckpt / 'best')
            runtime['overall_peak_allocated_gb_across_ranks'] = max(float(runtime.get('training_peak_allocated_gb_across_ranks', 0.0)), float(test_mem['peak_allocated_gb']))
            runtime['overall_peak_reserved_gb_across_ranks'] = max(float(runtime.get('training_peak_reserved_gb_across_ranks', 0.0)), float(test_mem['peak_reserved_gb']))
            test_metrics['runtime'] = test_runtime
            save_runtime_summary(runtime, runtime_path)
            if r == 0:
                save_metrics(test_metrics, run_out / 'test_metrics.json', run_out / 'test_metrics.txt')
                print('FINAL FULL RAW TEST WINDOWS')
                print(format_metrics(test_metrics), flush=True)
                print(f"FINAL TEST RUNTIME: {test_seconds:.3f}s | {test_runtime['test_city_windows_per_second']:.3f} city-windows/s | peak_alloc={test_mem['peak_allocated_gb']:.2f} GB | peak_reserved={test_mem['peak_reserved_gb']:.2f} GB", flush=True)
                print(f"TRAINING RUNTIME: wall={runtime['training_wall_seconds']:.1f}s | train_only={runtime['training_only_seconds']:.1f}s | validation={runtime['validation_seconds']:.1f}s | mean_epoch={runtime['mean_epoch_wall_seconds']:.1f}s | mean_train_sec/update={runtime['mean_train_seconds_per_update']:.3f}", flush=True)
                print(f'Runtime summary: {runtime_path}', flush=True)
                print(f"Best epoch: {best_epoch}; checkpoint: {run_ckpt / 'best'}", flush=True)
    finally:
        cleanup()
if __name__ == '__main__':
    main()
