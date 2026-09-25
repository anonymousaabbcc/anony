import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from .config import CITY_SHAPES, EXPECTED_WINDOWS, SOURCE_CITY_ORDER, Stage2Config
from .data import JointMultiCityDataset
from .distributed import cleanup, init_distributed, is_distributed, move_batch_to_device, rank, seed_everything
from .evaluation import ExactStridedDistributedSampler
from .metrics import RawFlowMetrics, assert_full_window_counts, format_metrics, save_metrics
from .normalization import TorchFlowNormalizer
from .prompt import Stage2Collator
from .vlm_bridge import FrozenStage1Bridge

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['valid', 'test'], default='test')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--output-prefix', default=None)
    args = ap.parse_args()
    cfg = Stage2Config(num_workers=args.num_workers)
    r, _, local_rank = init_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    seed_everything(cfg.seed, rank_offset=r)
    try:
        normalizer = TorchFlowNormalizer(cfg.normalization_stats)
        bridge = FrozenStage1Bridge(cfg, device=device)
        bridge.eval()
        ds = JointMultiCityDataset(args.split, normalizer, str(cfg.visual_cache_root), cfg.seed, training=False)
        sampler = ExactStridedDistributedSampler(ds) if is_distributed() else None
        collator = Stage2Collator(bridge.processor, max_seq_length=int(bridge.stage1_cfg.get('max_seq_length', 8192)))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=False, collate_fn=collator, persistent_workers=cfg.num_workers > 0)
        metrics = RawFlowMetrics(normalizer)
        head = bridge.stage1.tensor_head
        head_param = next(head.parameters())
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            context = bridge.prepare_context(batch)
            s_qa_head = context.s_qa.to(device=head_param.device, dtype=head_param.dtype)
            for city in SOURCE_CITY_ORDER:
                pred = head.forward_city(s_qa_head, city).float()
                metrics.update(city, pred, batch['cities'][city]['y'], active_mask=batch['cities'][city]['active_once'])
        metrics.synchronize(device)
        result = metrics.compute()
        assert_full_window_counts(result, args.split, EXPECTED_WINDOWS, CITY_SHAPES, cfg.forecast_len)
        result['full_window_count_audit'] = 'PASS'
        result['split'] = args.split
        result['baseline'] = 'Frozen Stage-1 P8 TensorHead'
        result['protocol'] = 'same complete raw 8->4 windows and same multi-city prompt/context as Stage-2'
        if r == 0:
            print('FROZEN STAGE-1 BASELINE -- FULL RAW WINDOWS')
            print(format_metrics(result))
            prefix = Path(args.output_prefix) if args.output_prefix else Path(cfg.output_root) / 'baseline' / f'stage1_{args.split}'
            save_metrics(result, str(prefix) + '.json', str(prefix) + '.txt')
            print('Saved:', str(prefix) + '.json')
            print('Saved:', str(prefix) + '.txt')
    finally:
        cleanup()
if __name__ == '__main__':
    main()
