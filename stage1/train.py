import argparse
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from .checkpoint import load_stage1_checkpoint, load_training_state, save_inference_checkpoint, save_training_checkpoint
from .collator import QwenStage1Collator
from .config import Stage1Config
from .dataset import Stage1QADataset
from .model import build_trainable_stage1_model, find_unique_module, trainable_parameter_report
from .normalization import FlowNormalizer

def distributed():
    return dist.is_available() and dist.is_initialized()

def rank():
    return dist.get_rank() if distributed() else 0

def world_size():
    return dist.get_world_size() if distributed() else 1

def is_main():
    return rank() == 0

def barrier():
    if distributed():
        dist.barrier()

def setup_distributed():
    ws = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if ws > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        torch.cuda.set_device(0)
        local_rank = 0
    return local_rank

def cleanup_distributed():
    if distributed():
        dist.barrier()
        dist.destroy_process_group()

def set_seed(seed: int, process_rank: int):
    final_seed = int(seed) + int(process_rank)
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)

def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model

def move_batch_to_device(batch, device):
    for key in ['input_ids', 'attention_mask', 'labels', 'pixel_values', 'image_grid_thw', 'readout_positions', 'value_target', 'value_mask']:
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch

def model_forward(model, batch, return_predictions=False):
    return model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'], readout_positions=batch['readout_positions'], value_target=batch['value_target'], value_mask=batch['value_mask'], tensor_targets=batch['tensor_targets'], return_predictions=return_predictions)

def build_optimizer(model, cfg):
    raw_model = unwrap_model(model)
    _, merger = find_unique_module(raw_model.vlm, 'visual.merger')
    lora_params = [p for name, p in raw_model.vlm.named_parameters() if p.requires_grad and 'lora_' in name]
    merger_params = [p for p in merger.parameters() if p.requires_grad]
    head_params = [p for p in list(raw_model.value_head.parameters()) + list(raw_model.tensor_head.parameters()) if p.requires_grad]
    if not lora_params:
        raise RuntimeError('No LoRA parameters are trainable.')
    if not merger_params:
        raise RuntimeError('Visual merger is not trainable.')
    if not head_params:
        raise RuntimeError('Auxiliary heads are not trainable.')
    known = {id(p) for p in lora_params + merger_params + head_params}
    unexpected = [name for name, p in raw_model.named_parameters() if p.requires_grad and id(p) not in known]
    if unexpected:
        raise RuntimeError('Unexpected trainable parameters. Stage-1 freeze policy is violated:\n' + '\n'.join(unexpected[:50]))
    return AdamW([{'params': lora_params, 'lr': cfg.lora_lr, 'weight_decay': cfg.weight_decay, 'name': 'lora'}, {'params': merger_params, 'lr': cfg.merger_lr, 'weight_decay': cfg.weight_decay, 'name': 'visual_merger'}, {'params': head_params, 'lr': cfg.head_lr, 'weight_decay': cfg.weight_decay, 'name': 'auxiliary_heads'}])

def _new_metric_sums():
    return {'ce_sum': 0.0, 'value_sum': 0.0, 'tensor_sum': 0.0, 'sample_count': 0.0, 'value_active_count': 0.0, 'tensor_active_count': 0.0}

def _accumulate_metrics(sums, out, batch):
    bsz = int(batch['input_ids'].shape[0])
    value_active = int(batch['value_mask'].any(dim=1).sum().item())
    tensor_active = sum((1 for x in batch['tensor_targets'] if bool(x)))
    sums['ce_sum'] += float(out.ce_loss.detach()) * bsz
    if value_active > 0:
        sums['value_sum'] += float(out.value_loss.detach()) * value_active
    if tensor_active > 0:
        sums['tensor_sum'] += float(out.tensor_loss.detach()) * tensor_active
    sums['sample_count'] += bsz
    sums['value_active_count'] += value_active
    sums['tensor_active_count'] += tensor_active

def _reduce_metric_sums(sums, device):
    keys = list(sums.keys())
    values = torch.tensor([sums[k] for k in keys], dtype=torch.float64, device=device)
    if distributed():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}

def _finalize_metrics(sums, lambda_value, lambda_tensor):
    n = max(sums['sample_count'], 1.0)
    ce_mean = sums['ce_sum'] / n
    value_active_mean = sums['value_sum'] / max(sums['value_active_count'], 1.0)
    tensor_active_mean = sums['tensor_sum'] / max(sums['tensor_active_count'], 1.0)
    value_dataset_contribution = sums['value_sum'] / n
    tensor_dataset_contribution = sums['tensor_sum'] / n
    objective = ce_mean + lambda_value * value_dataset_contribution + lambda_tensor * tensor_dataset_contribution
    return {'objective_loss': objective, 'ce_loss': ce_mean, 'value_loss_active': value_active_mean, 'tensor_loss_active': tensor_active_mean, 'value_dataset_contribution': value_dataset_contribution, 'tensor_dataset_contribution': tensor_dataset_contribution, 'samples': int(sums['sample_count']), 'value_active_samples': int(sums['value_active_count']), 'tensor_active_samples': int(sums['tensor_active_count'])}

@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    sums = _new_metric_sums()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            out = model_forward(model, batch)
        _accumulate_metrics(sums, out, batch)
    sums = _reduce_metric_sums(sums, device)
    metrics = _finalize_metrics(sums, lambda_value=cfg.lambda_value, lambda_tensor=cfg.lambda_tensor)
    model.train()
    return metrics

def append_jsonl(path: Path, record: dict):
    if not is_main():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
        f.flush()

def make_dataloaders(cfg, processor, normalizer, max_train_samples=None, max_valid_samples=None):
    train_ds = Stage1QADataset(cfg.train_jsonl(), normalizer, max_samples=max_train_samples)
    valid_ds = Stage1QADataset(cfg.valid_jsonl(), normalizer, max_samples=max_valid_samples)
    collator = QwenStage1Collator(processor, max_seq_length=cfg.max_seq_length)
    train_sampler = None
    valid_sampler = None
    if distributed():
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size(), rank=rank(), shuffle=True, seed=cfg.seed, drop_last=False)
        valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size(), rank=rank(), shuffle=False, seed=cfg.seed, drop_last=False)
    loader_kwargs = {'batch_size': cfg.micro_batch_size, 'collate_fn': collator, 'num_workers': cfg.num_workers, 'pin_memory': True}
    if cfg.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    train_loader = DataLoader(train_ds, shuffle=train_sampler is None, sampler=train_sampler, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, sampler=valid_sampler, **loader_kwargs)
    return (train_ds, valid_ds, train_loader, valid_loader, train_sampler)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qa-root', default='./outputs/qa/source')
    ap.add_argument('--output-dir', default='./outputs/stage1')
    ap.add_argument('--normalization-stats', default='./outputs/stage1_qwen25vl3b_chi_v3_finegrain/normalization_stats.json')
    ap.add_argument('--max-epochs', type=int, default=5)
    ap.add_argument('--micro-batch-size', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--checkpoint-every-updates', type=int, default=250)
    ap.add_argument('--early-stop-patience', type=int, default=2)
    ap.add_argument('--early-stop-min-delta', type=float, default=0.001)
    ap.add_argument('--resume', type=str, default=None, help='Path to a training checkpoint, normally .../stage1_qwen25vl3b_chi_v3_finegrain/latest')
    ap.add_argument('--max-train-samples', type=int, default=None)
    ap.add_argument('--max-valid-samples', type=int, default=None)
    ap.add_argument('--max-updates', type=int, default=None, help='Debug-only hard stop after this many optimizer updates.')
    args = ap.parse_args()
    local_rank = setup_distributed()
    device = torch.device('cuda', local_rank)
    cfg = Stage1Config(qa_root=args.qa_root, output_dir=args.output_dir, normalization_stats=args.normalization_stats, max_epochs=args.max_epochs, micro_batch_size=args.micro_batch_size, gradient_accumulation_steps=args.grad_accum, num_workers=args.num_workers, checkpoint_every_updates=args.checkpoint_every_updates, early_stopping_patience=args.early_stop_patience, early_stopping_min_delta=args.early_stop_min_delta)
    set_seed(cfg.seed, rank())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    stats_path = Path(cfg.normalization_stats)
    if not stats_path.exists():
        raise FileNotFoundError(f'Missing normalization stats: {stats_path}')
    normalizer = FlowNormalizer(cfg.normalization_stats)
    if args.resume is None:
        processor = AutoProcessor.from_pretrained(cfg.model_name, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels, use_fast=cfg.use_fast_image_processor)
        model, merger_name = build_trainable_stage1_model(cfg)
        model.to(device)
    else:
        model, processor, saved_cfg = load_stage1_checkpoint(args.resume, model_name=cfg.model_name, torch_dtype=torch.bfloat16, device=str(device), trainable=True)
        _, _ = find_unique_module(model.vlm, 'visual.merger')
        merger_name = 'restored from checkpoint'
    if is_main():
        report = trainable_parameter_report(model)
        print('World size:', world_size())
        print('Visual merger:', merger_name)
        print('Trainable parameter report:')
        print(json.dumps(report, indent=2))
        if report['other_trainable']['trainable'] != 0:
            raise RuntimeError('Unexpected trainable parameters.')
    train_ds, valid_ds, train_loader, valid_loader, train_sampler = make_dataloaders(cfg, processor, normalizer, max_train_samples=args.max_train_samples, max_valid_samples=args.max_valid_samples)
    optimizer = build_optimizer(model, cfg)
    updates_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_updates = updates_per_epoch * cfg.max_epochs
    warmup_steps = int(total_updates * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates)
    progress = {'epoch': 0, 'next_batch_idx': 0, 'global_update': 0, 'best_val_objective': float('inf'), 'epochs_without_improvement': 0}
    if args.resume is not None:
        progress = load_training_state(args.resume, optimizer, scheduler, device=device)
    if distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True, gradient_as_bucket_view=True)
    if is_main():
        global_batch = world_size() * cfg.micro_batch_size * cfg.gradient_accumulation_steps
        print(f'Train samples: {len(train_ds)}')
        print(f'Valid samples: {len(valid_ds)}')
        print(f'Micro batch / GPU: {cfg.micro_batch_size}')
        print(f'Gradient accumulation: {cfg.gradient_accumulation_steps}')
        print(f'Global effective batch: {global_batch}')
        print(f'Optimizer updates / epoch: {updates_per_epoch}')
        print(f'Max optimizer updates: {total_updates}')
        print(f'Warmup updates: {warmup_steps}')
        print(f'Starting progress: {json.dumps(progress)}')
    output_root = Path(cfg.output_dir)
    log_path = output_root / 'training_log.jsonl'
    start_epoch = int(progress['epoch'])
    global_update = int(progress['global_update'])
    best_val = float(progress['best_val_objective'])
    bad_epochs = int(progress['epochs_without_improvement'])
    optimizer.zero_grad(set_to_none=True)
    stop_training = False
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        start_batch_idx = int(progress['next_batch_idx']) if epoch == start_epoch else 0
        running = _new_metric_sums()
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < start_batch_idx:
                continue
            batch = move_batch_to_device(batch, device)
            do_update = (batch_idx + 1) % cfg.gradient_accumulation_steps == 0 or batch_idx + 1 == len(train_loader)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model_forward(model, batch)
                scaled_loss = out.loss / cfg.gradient_accumulation_steps
            if not torch.isfinite(out.loss):
                raise FloatingPointError(f'Non-finite loss at epoch={epoch}, batch_idx={batch_idx}: {float(out.loss)}')
            scaled_loss.backward()
            _accumulate_metrics(running, out, batch)
            if not do_update:
                continue
            clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            if global_update % cfg.log_every_updates == 0:
                reduced = _reduce_metric_sums(running, device)
                train_metrics = _finalize_metrics(reduced, cfg.lambda_value, cfg.lambda_tensor)
                lr_info = {group.get('name', str(i)): group['lr'] for i, group in enumerate(optimizer.param_groups)}
                record = {'type': 'train', 'epoch': epoch + 1, 'global_update': global_update, **train_metrics, 'lr': lr_info}
                if is_main():
                    print(json.dumps(record))
                append_jsonl(log_path, record)
                running = _new_metric_sums()
            if cfg.checkpoint_every_updates > 0 and global_update % cfg.checkpoint_every_updates == 0:
                progress = {'epoch': epoch, 'next_batch_idx': batch_idx + 1, 'global_update': global_update, 'best_val_objective': best_val, 'epochs_without_improvement': bad_epochs}
                save_training_checkpoint(model=unwrap_model(model), optimizer=optimizer, scheduler=scheduler, processor=processor, output_dir=str(output_root / 'latest'), cfg=cfg, normalization_stats=cfg.normalization_stats, progress=progress, metrics={'kind': 'periodic-latest'})
            if args.max_updates is not None and global_update >= args.max_updates:
                progress = {'epoch': epoch, 'next_batch_idx': batch_idx + 1, 'global_update': global_update, 'best_val_objective': best_val, 'epochs_without_improvement': bad_epochs}
                save_training_checkpoint(model=unwrap_model(model), optimizer=optimizer, scheduler=scheduler, processor=processor, output_dir=str(output_root / 'latest'), cfg=cfg, normalization_stats=cfg.normalization_stats, progress=progress, metrics={'kind': 'debug-max-updates'})
                stop_training = True
                break
        if stop_training:
            break
        val = evaluate(model, valid_loader, device, cfg)
        val_record = {'type': 'validation', 'epoch': epoch + 1, 'global_update': global_update, **val}
        if is_main():
            print(json.dumps(val_record, indent=2))
        append_jsonl(log_path, val_record)
        improved = val['objective_loss'] < best_val - cfg.early_stopping_min_delta
        if improved:
            best_val = val['objective_loss']
            bad_epochs = 0
            save_inference_checkpoint(model=unwrap_model(model), processor=processor, output_dir=str(output_root / 'best'), cfg=cfg, normalization_stats=cfg.normalization_stats, metrics={'epoch': epoch + 1, 'global_update': global_update, 'validation': val, 'best_val_objective': best_val})
        else:
            bad_epochs += 1
        progress = {'epoch': epoch + 1, 'next_batch_idx': 0, 'global_update': global_update, 'best_val_objective': best_val, 'epochs_without_improvement': bad_epochs}
        save_training_checkpoint(model=unwrap_model(model), optimizer=optimizer, scheduler=scheduler, processor=processor, output_dir=str(output_root / 'latest'), cfg=cfg, normalization_stats=cfg.normalization_stats, progress=progress, metrics={'kind': 'epoch-end', 'epoch': epoch + 1, 'validation': val})
        if bad_epochs >= cfg.early_stopping_patience:
            if is_main():
                print(f'Early stopping: {bad_epochs} consecutive epochs without sufficient validation improvement.')
            break
    if is_main():
        print('Training complete.')
        print('Best checkpoint:', output_root / 'best')
        print('Latest resumable checkpoint:', output_root / 'latest')
    cleanup_distributed()
if __name__ == '__main__':
    main()
