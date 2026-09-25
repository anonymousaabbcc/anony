import argparse
import gc
import json
import time
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
from .optim import build_optimizer
from .prompt import Stage2Collator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--candidates', default='1,2,4')
    args = ap.parse_args()
    candidates = [int(x) for x in args.candidates.split(',') if x.strip()]
    if not candidates:
        raise ValueError('No candidates')
    cfg = Stage2Config()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_everything(cfg.seed)
    normalizer = TorchFlowNormalizer(cfg.normalization_stats)
    model = Stage2UrbanForecaster(cfg, device)
    model.train()
    collator = Stage2Collator(model.bridge.processor, int(model.bridge.stage1_cfg.get('max_seq_length', 8192)))
    report = []
    for bs in candidates:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.zero_grad(set_to_none=True)
        ds = JointMultiCityDataset('train', normalizer, str(cfg.visual_cache_root), cfg.seed, training=True, max_joint_steps=bs, debug_sequential=True)
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collator)
        try:
            cpu_batch = next(iter(loader))
            nash = DistributedNashMTL(n_tasks=len(SOURCE_CITY_ORDER), update_weights_every=1, normalize_mean=cfg.nash_normalize_mean, optim_niter=cfg.nash_optim_niter, solver_max_iters=cfg.nash_solver_max_iters, eps=cfg.nash_eps, alpha_floor=cfg.nash_alpha_floor)
            optimizer = build_optimizer(model, cfg)
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            tx = exact_nash_accumulated_update(model=model, cpu_batches=[cpu_batch], device=device, nash=nash, optimizer=optimizer, scheduler=None, trainable_params=trainable, max_grad_norm=cfg.max_grad_norm)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            item = {'micro_batch': bs, 'status': 'PASS', 'step_seconds': elapsed, 'peak_allocated_gb': torch.cuda.max_memory_allocated(device) / 1024 ** 3, 'peak_reserved_gb': torch.cuda.max_memory_reserved(device) / 1024 ** 3, 'nash_alpha': tx.alpha.detach().float().cpu().tolist(), 'gradient_routing': tx.routing_mode, 'full_gradient_reentry': True}
            report.append(item)
            print(json.dumps(item, indent=2))
            del cpu_batch, tx, optimizer
        except torch.cuda.OutOfMemoryError as e:
            item = {'micro_batch': bs, 'status': 'OOM', 'error': str(e)}
            report.append(item)
            print(json.dumps(item, indent=2))
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
    passing = [x for x in report if x['status'] == 'PASS']
    recommendation = max((x['micro_batch'] for x in passing), default=None)
    payload = {'device': str(device), 'candidates': candidates, 'results': report, 'largest_passing_micro_batch': recommendation, 'note': 'Single-GPU formal-path split-Nash + full-gradient frozen-VLM re-entry update test. Confirm the chosen microbatch again with a distributed dry run.'}
    path = Path(cfg.log_root) / 'urbanbind_split_nash_fullgrad_batch_autotune.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('Saved:', path)
if __name__ == '__main__':
    main()
