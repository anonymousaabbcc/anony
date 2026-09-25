import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from .config import CITY_SHAPES, EXPECTED_WINDOWS, SOURCE_CITY_ORDER
from .data import JointMultiCityDataset
from .distributed import is_distributed, move_batch_to_device, rank, world_size
from .metrics import RawFlowMetrics, assert_full_window_counts
from .prompt import Stage2Collator

class ExactStridedDistributedSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset
        self.r = rank()
        self.w = world_size()

    def __iter__(self):
        return iter(range(self.r, len(self.dataset), self.w))

    def __len__(self):
        n = len(self.dataset)
        return max(0, (n - self.r + self.w - 1) // self.w)

def _allreduce_scalar_dict(values, device):
    keys = list(values)
    x = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=device)
    if is_distributed():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return {k: float(x[i].cpu()) for i, k in enumerate(keys)}

@torch.no_grad()
def warmup_evaluation(model, cfg, normalizer, split, device, batch_size=1, warmup_joint_steps=3):
    warmup_joint_steps = int(warmup_joint_steps)
    if warmup_joint_steps <= 0:
        return 0
    dataset = JointMultiCityDataset(split=split, normalizer=normalizer, visual_cache_root=str(cfg.visual_cache_root), seed=cfg.seed, training=False, max_joint_steps=warmup_joint_steps)
    eval_model = model.module if hasattr(model, 'module') else model
    collator = Stage2Collator(eval_model.bridge.processor, max_seq_length=int(eval_model.bridge.stage1_cfg.get('max_seq_length', 8192)))
    sampler = ExactStridedDistributedSampler(dataset) if is_distributed() else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=False, collate_fn=collator, persistent_workers=cfg.num_workers > 0)
    eval_model.eval()
    seen = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            _ = eval_model(batch)
        seen += int(batch['joint_index'].shape[0]) if torch.is_tensor(batch['joint_index']) else batch_size
    return seen

@torch.no_grad()
def evaluate_model(model, cfg, normalizer, split, device, batch_size=1, max_joint_steps=None):
    dataset = JointMultiCityDataset(split=split, normalizer=normalizer, visual_cache_root=str(cfg.visual_cache_root), seed=cfg.seed, training=False, max_joint_steps=max_joint_steps)
    eval_model = model.module if hasattr(model, 'module') else model
    collator = Stage2Collator(eval_model.bridge.processor, max_seq_length=int(eval_model.bridge.stage1_cfg.get('max_seq_length', 8192)))
    sampler = ExactStridedDistributedSampler(dataset) if is_distributed() else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=False, collate_fn=collator, persistent_workers=cfg.num_workers > 0)
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
    result['split'] = split
    result['joint_steps'] = len(dataset)
    if max_joint_steps is None:
        assert_full_window_counts(result, split, EXPECTED_WINDOWS, CITY_SHAPES, cfg.forecast_len)
        result['full_window_count_audit'] = 'PASS'
    result['exact_distributed_no_padding'] = True
    return result
