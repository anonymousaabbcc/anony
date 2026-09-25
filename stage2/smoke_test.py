import argparse
import gc
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import SOURCE_CITY_ORDER, Stage2Config
from .data import JointMultiCityDataset
from .distributed import move_batch_to_device, seed_everything
from .exact_accum import exact_nash_accumulated_update
from .metrics import RawFlowMetrics, format_metrics
from .model import Stage2UrbanForecaster
from .nash import DistributedNashMTL
from .normalization import TorchFlowNormalizer
from .optim import build_cosine_scheduler, build_optimizer
from .prompt import Stage2Collator

def _snapshot(params):
    return [p.detach().float().cpu().clone() for p in params]

def _count_changed(before, params):
    return sum((1 for b, p in zip(before, params) if not torch.equal(b, p.detach().float().cpu())))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--samples', type=int, default=1)
    args = ap.parse_args()
    cfg = Stage2Config(micro_batch_size=args.batch_size)
    device = torch.device(args.device)
    seed_everything(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    normalizer = TorchFlowNormalizer(cfg.normalization_stats)
    model = Stage2UrbanForecaster(cfg, device)
    model.train()
    routing = model.assert_gradient_routing_partition()
    ds = JointMultiCityDataset('train', normalizer, str(cfg.visual_cache_root), cfg.seed, training=True, max_joint_steps=max(args.samples, args.batch_size), debug_sequential=True)
    collator = Stage2Collator(model.bridge.processor, int(model.bridge.stage1_cfg.get('max_seq_length', 8192)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    cpu_batch = next(iter(loader))
    batch = move_batch_to_device(cpu_batch, device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        out = model(batch)
    for city in SOURCE_CITY_ORDER:
        expected = batch['cities'][city]['y'].shape
        got = out.predictions[city].shape
        if got != expected:
            raise RuntimeError(f'{city}: prediction shape {got} != target {expected}')
    if model.bridge.trainable_parameter_count != 0:
        raise RuntimeError('Stage-1 is not frozen')
    if not out.r_hidden.requires_grad:
        raise RuntimeError('Full-gradient re-entry requires R to remain differentiable wrt Z')
    if not out.z.requires_grad:
        raise RuntimeError('Aligned Z must remain differentiable')
    qwen_probe = out.r_hidden.float().square().mean()
    qwen_grad_z = torch.autograd.grad(qwen_probe, out.z, retain_graph=False, create_graph=False, allow_unused=False)[0]
    if not torch.isfinite(qwen_grad_z).all() or float(qwen_grad_z.abs().sum().detach().cpu()) <= 0.0:
        raise RuntimeError('Frozen-Qwen re-entry Jacobian dR/dZ is missing or non-finite')
    qwen_grad_z_l1 = float(qwen_grad_z.abs().mean().detach().cpu())
    metrics = RawFlowMetrics(normalizer)
    for city in SOURCE_CITY_ORDER:
        metrics.update(city, out.predictions[city], batch['cities'][city]['y'])
    m = metrics.compute()
    forward_city_losses = out.city_losses.detach().float().cpu()
    selector_mean = out.projector_weights.detach().float().mean(dim=0).cpu().tolist()
    gate_mean = float(out.gates.detach().float().mean().cpu())
    prediction_shapes = {c: list(out.predictions[c].shape) for c in SOURCE_CITY_ORDER}
    del qwen_grad_z, qwen_probe, out, batch
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    optimizer = build_optimizer(model, cfg)
    scheduler, _ = build_cosine_scheduler(optimizer, total_updates=10, warmup_ratio=0.0)
    nash = DistributedNashMTL(n_tasks=len(SOURCE_CITY_ORDER), update_weights_every=1, normalize_mean=cfg.nash_normalize_mean, optim_niter=cfg.nash_optim_niter, solver_max_iters=cfg.nash_solver_max_iters, eps=cfg.nash_eps, alpha_floor=cfg.nash_alpha_floor)
    shared_before = _snapshot(routing['shared'])
    private_before = {city: _snapshot(routing['private_by_city'][city]) for city in SOURCE_CITY_ORDER}
    trainable = [p for p in model.parameters() if p.requires_grad]
    tx = exact_nash_accumulated_update(model=model, cpu_batches=[cpu_batch], device=device, nash=nash, optimizer=optimizer, scheduler=scheduler, trainable_params=trainable, max_grad_norm=cfg.max_grad_norm)
    changed = {'shared': _count_changed(shared_before, routing['shared']), 'private': {city: _count_changed(private_before[city], routing['private_by_city'][city]) for city in SOURCE_CITY_ORDER}}
    if changed['shared'] == 0 or any((v == 0 for v in changed['private'].values())):
        raise RuntimeError(f'Split routing failed to update a parameter route: {changed}')
    if float(tx.alpha.min().cpu()) + 1e-07 < cfg.nash_alpha_floor:
        raise RuntimeError('Nash floor invariant failed')
    if abs(float(tx.alpha.mean().cpu()) - 1.0) > 1e-05:
        raise RuntimeError('Nash mean-one invariant failed')
    if any((p.grad is not None for p in model.bridge.stage1.parameters())):
        raise RuntimeError('A frozen Stage-1 parameter received a gradient')
    report = {'architecture': 'urbanbind_split_nash_fullgrad_v3_2_bucketed_sync', 'forecast_len': cfg.forecast_len, 'vlm_reentry_calls_per_forward': 1, 'full_gradient_reentry': True, 'r_hidden_requires_grad': True, 'z_requires_grad': True, 'qwen_jacobian_grad_z_l1': qwen_grad_z_l1, 'gradient_routing': tx.routing_mode, 'nash_applies_to': 'shared_only', 'private_city_coefficient': 1.0, 'nash_alpha_floor': cfg.nash_alpha_floor, 'nash_alpha': {c: float(tx.alpha[i].cpu()) for i, c in enumerate(SOURCE_CITY_ORDER)}, 'forward_city_losses': {c: float(forward_city_losses[i]) for i, c in enumerate(SOURCE_CITY_ORDER)}, 'shared_weighted_loss': float(tx.weighted_loss_local_mean.cpu()), 'private_unweighted_loss': float(tx.private_loss_local_mean.cpu()), 'shared_grad_norm_preclip': float(tx.shared_grad_norm.cpu()), 'private_grad_norms_preclip': {c: float(tx.private_grad_norms[i].cpu()) for i, c in enumerate(SOURCE_CITY_ORDER)}, 'changed_parameter_tensors': changed, 'stage1_trainable_params': model.bridge.trainable_parameter_count, 'prediction_shapes': prediction_shapes, 'selector_mean': selector_mean, 'gate_mean': gate_mean, 'peak_allocated_gb': torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else 0.0, 'peak_reserved_gb': torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == 'cuda' else 0.0, 'raw_metrics_debug': m}
    print('URBANBIND SPLIT-NASH FULL-GRAD SINGLE-GPU SMOKE: PASS')
    print(json.dumps(report, indent=2))
    print(format_metrics(m))
    out_path = Path(cfg.log_root) / 'urbanbind_split_nash_fullgrad_smoke_test.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', out_path)
if __name__ == '__main__':
    main()
