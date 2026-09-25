import argparse
import json
from dataclasses import replace
from pathlib import Path
import torch
from .checkpoint import load_stage2_weights
from .config import Stage2Config
from .hyper_subset import evaluate_model_subset, warmup_evaluation_subset
from .metrics import format_metrics, save_metrics
from .model import Stage2UrbanForecaster
from .normalization import TorchFlowNormalizer

def apply_shift(model, shift):
    for enc in model.multi_city.city_encoders.values():
        for block in enc.spatial:
            block.shift_size = int(shift)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--test-fraction', type=float, default=0.05)
    ap.add_argument('--subset-seed', type=int, default=2026)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--num-workers', type=int, default=2)
    args = ap.parse_args()
    base = Stage2Config()
    out = Path(base.output_root) / args.run_name
    settings_path = out / 'hyper_settings.json'
    if not settings_path.exists():
        raise FileNotFoundError(settings_path)
    hs = json.loads(settings_path.read_text())
    shift = int(hs['shift_size'])
    nproj = int(hs['num_projectors'])
    cfg = replace(base, num_residual_projectors=nproj, num_workers=args.num_workers)
    ckpt = Path(cfg.checkpoint_root) / args.run_name / 'best'
    if not (ckpt / 'stage2_model.pt').exists():
        raise FileNotFoundError(f'No best checkpoint: {ckpt}')
    device = torch.device(args.device)
    model = Stage2UrbanForecaster(cfg, device=device)
    apply_shift(model, shift)
    load_stage2_weights(ckpt, model)
    model.eval()
    norm = TorchFlowNormalizer(cfg.normalization_stats)
    warmup_evaluation_subset(model, cfg, norm, 'test', device, batch_size=1, fraction=args.test_fraction, subset_seed=args.subset_seed, warmup_joint_steps=3)
    metrics = evaluate_model_subset(model, cfg, norm, 'test', device, batch_size=1, fraction=args.test_fraction, subset_seed=args.subset_seed)
    metrics['hyper_repair_eval_only'] = True
    save_metrics(metrics, out / 'test_metrics.json', out / 'test_metrics.txt')
    print(format_metrics(metrics))
    print('SAVED', out / 'test_metrics.json')
if __name__ == '__main__':
    main()
