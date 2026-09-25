import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import SOURCE_CITY_ORDER, Stage2Config
from .data import JointMultiCityDataset
from .distributed import move_batch_to_device, seed_everything
from .model import Stage2UrbanForecaster
from .normalization import TorchFlowNormalizer
from .prompt import Stage2Collator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    cfg = Stage2Config()
    device = torch.device(args.device)
    seed_everything(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    normalizer = TorchFlowNormalizer(cfg.normalization_stats)
    model = Stage2UrbanForecaster(cfg, device=device)
    model.train()
    ds = JointMultiCityDataset('train', normalizer, str(cfg.visual_cache_root), cfg.seed, training=True, max_joint_steps=1, debug_sequential=True)
    collator = Stage2Collator(model.bridge.processor, int(model.bridge.stage1_cfg.get('max_seq_length', 8192)))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator)
    batch = move_batch_to_device(next(iter(loader)), device)
    context = model.bridge.prepare_context(batch)
    city_inputs = {city: batch['cities'][city]['x'] for city in SOURCE_CITY_ORDER}
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        global_h, _, _ = model.multi_city(city_inputs)
        z, _, _ = model.alignment(global_h, context.s_qa)
        r = model.bridge.reenter(context, z)
        probe = r.float().square().mean()
    if not r.requires_grad:
        raise RuntimeError('R is detached: full-gradient semantic re-entry is not active')
    if not z.requires_grad:
        raise RuntimeError('Z is detached before semantic re-entry')
    alignment_params = [p for p in model.alignment.parameters() if p.requires_grad]
    requested = [z] + alignment_params
    grads = torch.autograd.grad(probe, requested, create_graph=False, retain_graph=False, allow_unused=False)
    grad_z = grads[0]
    dsr_grads = grads[1:]
    if not torch.isfinite(grad_z).all():
        raise RuntimeError('dR/dZ is non-finite')
    grad_z_l1 = float(grad_z.abs().mean().detach().cpu())
    grad_z_max = float(grad_z.abs().max().detach().cpu())
    if grad_z_l1 <= 0.0:
        raise RuntimeError('dR/dZ is exactly zero; DSR cannot receive the Qwen path')
    if len(dsr_grads) != len(alignment_params):
        raise RuntimeError('DSR gradient list length mismatch')
    if any((not torch.isfinite(g).all() for g in dsr_grads)):
        raise RuntimeError('Non-finite DSR gradient through frozen Qwen')
    nonzero_dsr_tensors = sum((float(g.abs().sum().detach().cpu()) > 0.0 for g in dsr_grads))
    dsr_grad_l1_sum = sum((float(g.abs().mean().detach().cpu()) for g in dsr_grads))
    if nonzero_dsr_tensors == 0 or dsr_grad_l1_sum <= 0.0:
        raise RuntimeError('DSR received no gradient through frozen Qwen')
    if model.bridge.trainable_parameter_count != 0:
        raise RuntimeError('Stage-1 parameters must remain frozen')
    if any((p.grad is not None for p in model.bridge.stage1.parameters())):
        raise RuntimeError('Frozen Stage-1 parameter unexpectedly accumulated .grad')
    report = {'status': 'PASS', 'stage1_trainable_params': model.bridge.trainable_parameter_count, 'r_requires_grad': bool(r.requires_grad), 'z_requires_grad': bool(z.requires_grad), 'qwen_jacobian_grad_z_l1': grad_z_l1, 'qwen_jacobian_grad_z_max': grad_z_max, 'dsr_parameter_tensors_tested': len(alignment_params), 'dsr_parameter_tensors_nonzero_through_qwen': nonzero_dsr_tensors, 'dsr_qwen_path_grad_l1_sum': dsr_grad_l1_sum, 'semantic_reentry': 'frozen_parameters_but_full_dR_dZ', 'proof_path': 'probe(R) -> FrozenQwen -> Z -> DSR (direct-Z decoder residual excluded)', 'peak_allocated_gb': torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else 0.0, 'peak_reserved_gb': torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == 'cuda' else 0.0}
    print('URBANBIND FULL-GRAD FROZEN-QWEN RE-ENTRY: PASS')
    print(json.dumps(report, indent=2))
    out = Path(cfg.log_root) / 'stage2_fullgrad_semantic_reentry_verify.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', out)
if __name__ == '__main__':
    main()
