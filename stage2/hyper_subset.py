import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from .config import CITY_SHAPES, EXPECTED_WINDOWS, SOURCE_CITY_ORDER
from .data import JointMultiCityDataset
from .distributed import is_distributed, move_batch_to_device, rank, world_size
from .evaluation import ExactStridedDistributedSampler
from .metrics import RawFlowMetrics
from .prompt import Stage2Collator

def _salt(split: str, city: str) -> int:
    return {'train': 1000, 'valid': 2000, 'test': 3000}[split] + 97 * SOURCE_CITY_ORDER.index(city)

def make_fractional_joint_dataset(cfg, normalizer, split: str, fraction: float, subset_seed: int, training: bool):
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f'fraction must be in (0,1], got {fraction}')
    ds = JointMultiCityDataset(split=split, normalizer=normalizer, visual_cache_root=str(cfg.visual_cache_root), seed=cfg.seed, training=training)
    full_counts = {c: len(ds.starts[c]) for c in SOURCE_CITY_ORDER}
    selected_counts = {}
    if fraction < 1.0:
        for city in SOURCE_CITY_ORDER:
            starts = np.asarray(ds.starts[city])
            n = max(1, int(math.ceil(len(starts) * fraction)))
            rng = np.random.default_rng(int(subset_seed) + _salt(split, city))
            idx = rng.choice(len(starts), size=n, replace=False)
            ds.starts[city] = starts[idx]
            selected_counts[city] = int(n)
        ds.full_length = max((len(ds.starts[c]) for c in SOURCE_CITY_ORDER))
        ds.length = ds.full_length
        ds.set_epoch(0)
    else:
        selected_counts = dict(full_counts)
    ds.hyper_full_counts = full_counts
    ds.hyper_selected_counts = selected_counts
    ds.hyper_fraction = fraction
    ds.hyper_subset_seed = int(subset_seed)
    return ds

def _allreduce_scalar_dict(values, device):
    keys = list(values)
    x = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=device)
    if is_distributed():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return {k: float(x[i].cpu()) for i, k in enumerate(keys)}

@torch.no_grad()
def warmup_evaluation_subset(model, cfg, normalizer, split, device, batch_size=1, fraction=0.05, subset_seed=2026, warmup_joint_steps=3):
    ds = make_fractional_joint_dataset(cfg, normalizer, split, fraction, subset_seed, False)
    ds.length = min(len(ds), int(warmup_joint_steps))
    eval_model = model.module if hasattr(model, 'module') else model
    collator = Stage2Collator(eval_model.bridge.processor, max_seq_length=int(eval_model.bridge.stage1_cfg.get('max_seq_length', 8192)))
    sampler = ExactStridedDistributedSampler(ds) if is_distributed() else None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=False, collate_fn=collator, persistent_workers=cfg.num_workers > 0)
    eval_model.eval()
    seen = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            _ = eval_model(batch)
        seen += int(batch['joint_index'].shape[0]) if torch.is_tensor(batch['joint_index']) else batch_size
    return seen

@torch.no_grad()
def evaluate_model_subset(model, cfg, normalizer, split, device, batch_size=1, fraction=0.05, subset_seed=2026):
    ds = make_fractional_joint_dataset(cfg, normalizer, split, fraction, subset_seed, False)
    eval_model = model.module if hasattr(model, 'module') else model
    collator = Stage2Collator(eval_model.bridge.processor, max_seq_length=int(eval_model.bridge.stage1_cfg.get('max_seq_length', 8192)))
    sampler = ExactStridedDistributedSampler(ds) if is_distributed() else None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=False, collate_fn=collator, persistent_workers=cfg.num_workers > 0)
    metrics = RawFlowMetrics(normalizer)
    norm_loss_sum = {c: 0.0 for c in SOURCE_CITY_ORDER}
    norm_loss_n = {c: 0.0 for c in SOURCE_CITY_ORDER}
    eval_model.eval()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out = eval_model(batch)
        for city in SOURCE_CITY_ORDER:
            active = batch['cities'][city]['active_once']
            metrics.update(city, out.predictions[city], batch['cities'][city]['y'], active_mask=active)
            if active.any():
                pred = out.predictions[city][active].float()
                tgt = batch['cities'][city]['y'][active].float()
                n = int(active.sum())
                norm_loss_sum[city] += float((pred - tgt).abs().mean().cpu()) * n
                norm_loss_n[city] += n
    metrics.synchronize(device)
    packed = {}
    for c in SOURCE_CITY_ORDER:
        packed[f'sum::{c}'] = norm_loss_sum[c]
        packed[f'n::{c}'] = norm_loss_n[c]
    packed = _allreduce_scalar_dict(packed, device)
    result = metrics.compute()
    result['normalized_l1_by_city'] = {c: packed[f'sum::{c}'] / max(1.0, packed[f'n::{c}']) for c in SOURCE_CITY_ORDER}
    result['normalized_l1_macro'] = sum(result['normalized_l1_by_city'].values()) / len(SOURCE_CITY_ORDER)
    result.update({'split': f'{split}_random_subset', 'joint_steps': len(ds), 'subset_fraction': float(fraction), 'subset_seed': int(subset_seed), 'subset_counts': dict(ds.hyper_selected_counts), 'full_counts': dict(ds.hyper_full_counts), 'exact_distributed_no_padding': True, 'full_window_count_audit': 'NOT_APPLICABLE_RANDOM_SUBSET'})
    return result
