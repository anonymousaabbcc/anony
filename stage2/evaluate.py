import argparse
import json
from dataclasses import replace
from pathlib import Path
import torch
from .checkpoint import load_stage2_weights
from .config import EXPECTED_WINDOWS, Stage2Config
from .distributed import cleanup, init_distributed, rank, seed_everything, world_size
from .evaluation import evaluate_model, warmup_evaluation
from .metrics import format_metrics, save_metrics
from .model import Stage2UrbanForecaster
from .normalization import TorchFlowNormalizer
from .runtime import peak_memory_across_ranks, reset_peak_memory, runtime_definition, save_runtime_summary, start_wall_timer, stop_wall_timer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', choices=['valid', 'test'], default='test')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--num-workers', type=int, default=None)
    ap.add_argument('--output-prefix', default=None)
    ap.add_argument('--runtime-warmup-steps', type=int, default=3, help='Untimed joint batches before formal runtime measurement.')
    args = ap.parse_args()
    meta_path = Path(args.checkpoint) / 'metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'Missing checkpoint metadata: {meta_path}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    cfg = Stage2Config.from_dict(meta.get('config', {}))
    if args.num_workers is not None:
        cfg = replace(cfg, num_workers=args.num_workers)
    r, _, local_rank = init_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    seed_everything(cfg.seed, rank_offset=r)
    try:
        model = Stage2UrbanForecaster(cfg, device=device)
        load_stage2_weights(args.checkpoint, model)
        normalizer = TorchFlowNormalizer(cfg.normalization_stats)
        warmup_seen = warmup_evaluation(model, cfg, normalizer, args.split, device, batch_size=args.batch_size, warmup_joint_steps=args.runtime_warmup_steps)
        reset_peak_memory(device)
        t0 = start_wall_timer(device)
        result = evaluate_model(model, cfg, normalizer, args.split, device, batch_size=args.batch_size, max_joint_steps=None)
        eval_seconds = stop_wall_timer(t0, device)
        mem = peak_memory_across_ranks(device)
        city_windows = int(sum(EXPECTED_WINDOWS[args.split].values()))
        joint_steps = int(result['joint_steps'])
        runtime = {'phase': f'standalone_{args.split}_evaluation', 'checkpoint': str(args.checkpoint), 'world_size': int(world_size()), 'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu', 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda, 'batch_size_per_rank': int(args.batch_size), 'warmup_joint_steps_requested': int(args.runtime_warmup_steps), 'warmup_examples_processed_local_rank': int(warmup_seen), 'evaluation_seconds': float(eval_seconds), 'joint_steps': joint_steps, 'city_windows': city_windows, 'joint_steps_per_second': joint_steps / max(eval_seconds, 1e-12), 'city_windows_per_second': city_windows / max(eval_seconds, 1e-12), **mem, 'definitions': runtime_definition()}
        result['runtime'] = runtime
        if r == 0:
            print(format_metrics(result))
            print(f"RUNTIME {args.split}: {eval_seconds:.3f}s | {runtime['city_windows_per_second']:.3f} city-windows/s | peak_alloc={mem['peak_allocated_gb']:.2f} GB | peak_reserved={mem['peak_reserved_gb']:.2f} GB", flush=True)
            prefix = Path(args.output_prefix) if args.output_prefix else Path(cfg.output_root) / 'evaluation' / f'stage2_{args.split}'
            save_metrics(result, str(prefix) + '.json', str(prefix) + '.txt')
            save_runtime_summary(runtime, str(prefix) + '_runtime.json')
            print('Saved:', str(prefix) + '.json')
            print('Saved:', str(prefix) + '.txt')
            print('Saved:', str(prefix) + '_runtime.json')
    finally:
        cleanup()
if __name__ == '__main__':
    main()
