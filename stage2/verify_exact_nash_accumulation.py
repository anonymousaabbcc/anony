import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import SOURCE_CITY_ORDER, Stage2Config
from .data import JointMultiCityDataset
from .distributed import seed_everything
from .exact_accum import exact_nash_accumulated_update
from .model import Stage2UrbanForecaster
from .nash import DistributedNashMTL
from .normalization import TorchFlowNormalizer
from .optim import build_cosine_scheduler, build_optimizer
from .prompt import Stage2Collator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--accum-steps', type=int, default=2)
    args = ap.parse_args()
    if args.accum_steps < 2:
        raise ValueError('This verifier is intended for accumulation >= 2')
    cfg = Stage2Config(micro_batch_size=1, gradient_accumulation_steps=args.accum_steps)
    device = torch.device(args.device)
    seed_everything(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    normalizer = TorchFlowNormalizer(cfg.normalization_stats)
    model = Stage2UrbanForecaster(cfg, device)
    model.train()
    ds = JointMultiCityDataset('train', normalizer, str(cfg.visual_cache_root), cfg.seed, training=True, max_joint_steps=args.accum_steps, debug_sequential=True)
    collator = Stage2Collator(model.bridge.processor, int(model.bridge.stage1_cfg.get('max_seq_length', 8192)))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator)
    cpu_batches = []
    for batch in loader:
        cpu_batches.append(batch)
        if len(cpu_batches) >= args.accum_steps:
            break
    if len(cpu_batches) != args.accum_steps:
        raise RuntimeError('Could not collect enough micro-batches for accumulation verifier')
    optimizer = build_optimizer(model, cfg)
    scheduler, _ = build_cosine_scheduler(optimizer, total_updates=2, warmup_ratio=0.0)
    nash = DistributedNashMTL(n_tasks=len(SOURCE_CITY_ORDER), update_weights_every=1, normalize_mean=cfg.nash_normalize_mean, optim_niter=cfg.nash_optim_niter, solver_max_iters=cfg.nash_solver_max_iters, eps=cfg.nash_eps, alpha_floor=cfg.nash_alpha_floor)
    routing = model.assert_gradient_routing_partition()
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().float().cpu().clone() for p in trainable]
    shared_before = [p.detach().float().cpu().clone() for p in routing['shared']]
    private_before = {city: [p.detach().float().cpu().clone() for p in routing['private_by_city'][city]] for city in SOURCE_CITY_ORDER}
    report = exact_nash_accumulated_update(model=model, cpu_batches=cpu_batches, device=device, nash=nash, optimizer=optimizer, scheduler=scheduler, trainable_params=trainable, max_grad_norm=cfg.max_grad_norm)
    changed_tensors = sum((1 for b, p in zip(before, trainable) if not torch.equal(b, p.detach().float().cpu())))
    if changed_tensors == 0:
        raise RuntimeError('No Stage-2 trainable parameter changed after exact update')
    changed_shared = sum((1 for b, p in zip(shared_before, routing['shared']) if not torch.equal(b, p.detach().float().cpu())))
    changed_private = {city: sum((1 for b, p in zip(private_before[city], routing['private_by_city'][city]) if not torch.equal(b, p.detach().float().cpu()))) for city in SOURCE_CITY_ORDER}
    if changed_shared == 0 or any((v == 0 for v in changed_private.values())):
        raise RuntimeError(f'A split gradient route did not update: shared={changed_shared}, private={changed_private}')
    if float(report.alpha.min().cpu()) + 1e-07 < cfg.nash_alpha_floor:
        raise RuntimeError('Applied Nash alpha floor violated')
    if abs(float(report.alpha.mean().cpu()) - 1.0) > 1e-05:
        raise RuntimeError('Applied Nash alpha is not mean-one')
    if report.optimizer_steps != 1 or report.scheduler_steps != 1 or report.nash_solves != 1:
        raise RuntimeError('Update transaction counts are not exactly 1/1/1')
    if nash.step != 1:
        raise RuntimeError(f'Nash step must be 1 after one effective update, got {nash.step}')
    if model.bridge.trainable_parameter_count != 0:
        raise RuntimeError('Stage-1 is not frozen')
    if any((p.grad is not None for p in model.bridge.stage1.parameters())):
        raise RuntimeError('A frozen Stage-1 parameter received a gradient')
    result = {'status': 'PASS', 'micro_batch_per_gpu': 1, 'accum_steps': args.accum_steps, 'effective_batch_single_gpu': args.accum_steps, 'nash_step_after_transaction': nash.step, 'nash_solves': report.nash_solves, 'optimizer_steps': report.optimizer_steps, 'scheduler_steps': report.scheduler_steps, 'gradient_routing': report.routing_mode, 'full_gradient_reentry': True, 'nash_alpha_floor': cfg.nash_alpha_floor, 'alpha': {city: float(report.alpha[i].cpu()) for i, city in enumerate(SOURCE_CITY_ORDER)}, 'probe_city_losses_local_mean': {city: float(report.probe_city_losses_local_mean[i].cpu()) for i, city in enumerate(SOURCE_CITY_ORDER)}, 'update_city_losses_local_mean': {city: float(report.update_city_losses_local_mean[i].cpu()) for i, city in enumerate(SOURCE_CITY_ORDER)}, 'probe_update_city_loss_max_abs_diff': report.max_probe_update_city_loss_abs_diff, 'grad_norm': float(report.grad_norm.float().cpu()), 'shared_grad_norm': float(report.shared_grad_norm.float().cpu()), 'private_grad_norms': {city: float(report.private_grad_norms[i].cpu()) for i, city in enumerate(SOURCE_CITY_ORDER)}, 'changed_stage2_parameter_tensors': changed_tensors, 'changed_shared_parameter_tensors': changed_shared, 'changed_private_parameter_tensors': changed_private, 'stage1_trainable_params': model.bridge.trainable_parameter_count, 'peak_allocated_gb': torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else 0.0, 'peak_reserved_gb': torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == 'cuda' else 0.0}
    print('STAGE-2 SPLIT-NASH EXACT ACCUMULATION: PASS')
    print(json.dumps(result, indent=2))
    out = Path(cfg.log_root) / 'stage2_split_nash_accumulation_verify.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved:', out)
if __name__ == '__main__':
    main()
