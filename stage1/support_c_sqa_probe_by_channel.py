import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .checkpoint import load_stage1_checkpoint
from .collator import QwenStage1Collator
from .config import CITY_ORDER
from .model import find_language_final_norm
from .normalization import FlowNormalizer
from .support_common import RecordDataset, load_jsonl_records, move_multimodal_batch, p8_future_mean_target, regression_metrics, ridge_fit_dual, ridge_predict, write_json_and_text
CHANNEL_NAMES = ('inflow', 'outflow')
HORIZONS = 4
CHANNELS = 2
DIMS_PER_CITY = HORIZONS * CHANNELS

class ReadoutCapture:

    def __init__(self, vlm):
        _, norm = find_language_final_norm(vlm)
        self.hidden = None
        self.handle = norm.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.hidden = output

    def pop(self, positions):
        if self.hidden is None:
            raise RuntimeError('final language hidden state was not captured')
        h = self.hidden
        idx = torch.arange(h.shape[0], device=h.device)
        out = h[idx, positions].detach().float().cpu().numpy()
        self.hidden = None
        return out

    def close(self):
        self.handle.remove()

def processor_from_checkpoint(checkpoint: Path, model_name: str, cfg: dict):
    pdir = checkpoint / 'processor'
    if pdir.exists():
        return AutoProcessor.from_pretrained(pdir, use_fast=False)
    return AutoProcessor.from_pretrained(model_name, min_pixels=int(cfg['min_pixels']), max_pixels=int(cfg['max_pixels']), use_fast=False)

@torch.no_grad()
def extract_p8(vlm, processor, records, normalizer, cfg, device):
    ds = RecordDataset(records, normalizer)
    collator = QwenStage1Collator(processor, max_seq_length=int(cfg['max_seq_length']))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)
    cap = ReadoutCapture(vlm)
    vlm.eval()
    X, Y, sample_ids = ([], [], [])
    for i, batch in enumerate(loader, start=1):
        item = ds[i - 1]
        target = p8_future_mean_target(item)
        if target.shape != (len(CITY_ORDER) * DIMS_PER_CITY,):
            raise ValueError(f'Expected P8 target dim {len(CITY_ORDER) * DIMS_PER_CITY}, got {target.shape}')
        batch = move_multimodal_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = vlm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=None, pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'], use_cache=False, return_dict=True)
        x = cap.pop(batch['readout_positions'])[0]
        X.append(x)
        Y.append(target)
        sample_ids.append(records[i - 1]['sample_id'])
        if i % 50 == 0:
            print(f'processed {i}/{len(loader)}')
    cap.close()
    return (np.stack(X), np.stack(Y), sample_ids)

def city_slice(city_idx: int):
    start = city_idx * DIMS_PER_CITY
    return slice(start, start + DIMS_PER_CITY)

def channel_indices(channel_idx: int):
    idx = []
    for city_idx in range(len(CITY_ORDER)):
        base = city_idx * DIMS_PER_CITY
        idx.extend((base + h * CHANNELS + channel_idx for h in range(HORIZONS)))
    return np.asarray(idx, dtype=np.int64)

def city_channel_indices(city_idx: int, channel_idx: int):
    base = city_idx * DIMS_PER_CITY
    return np.asarray([base + h * CHANNELS + channel_idx for h in range(HORIZONS)], dtype=np.int64)

def evaluate_probe(name, splits, alpha_grid):
    Xtr, Ytr = splits['train']
    Xva, Yva = splits['valid']
    Xte, Yte = splits['test']
    candidates = []
    for alpha in alpha_grid:
        m = ridge_fit_dual(Xtr, Ytr, alpha)
        pred = ridge_predict(m, Xva)
        met = regression_metrics(pred, Yva)
        candidates.append({'alpha': float(alpha), **met})
    best = min(candidates, key=lambda x: x['rmse'])
    model = ridge_fit_dual(Xtr, Ytr, best['alpha'])
    pred = ridge_predict(model, Xte)
    overall = regression_metrics(pred, Yte)
    by_channel = {}
    for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
        idx = channel_indices(channel_idx)
        by_channel[channel_name] = regression_metrics(pred[:, idx], Yte[:, idx])
    per_city = {}
    per_city_channel = {}
    for city_idx, city in enumerate(CITY_ORDER):
        sl = city_slice(city_idx)
        per_city[city] = regression_metrics(pred[:, sl], Yte[:, sl])
        per_city_channel[city] = {}
        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            idx = city_channel_indices(city_idx, channel_idx)
            per_city_channel[city][channel_name] = regression_metrics(pred[:, idx], Yte[:, idx])
    mean_pred = np.broadcast_to(Ytr.mean(axis=0, keepdims=True), Yte.shape)
    mean_baseline = regression_metrics(mean_pred, Yte)
    mean_baseline_by_channel = {}
    for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
        idx = channel_indices(channel_idx)
        mean_baseline_by_channel[channel_name] = regression_metrics(mean_pred[:, idx], Yte[:, idx])
    return {'name': name, 'selected_alpha': best['alpha'], 'validation_grid': candidates, 'test': overall, 'test_by_channel': by_channel, 'test_by_city': per_city, 'test_by_city_channel': per_city_channel, 'mean_target_baseline_test': mean_baseline, 'mean_target_baseline_test_by_channel': mean_baseline_by_channel}

def verify_against_existing_formal(output_dir: Path, base_probe: dict, stage1_probe: dict, atol: float=1e-06):
    path = output_dir / 'C_sqa_linear_probe.json'
    if not path.exists():
        return {'checked': False, 'reason': 'formal C JSON not found'}
    old = json.loads(path.read_text(encoding='utf-8'))
    diffs = {}
    for key, cur in [('base_probe', base_probe), ('stage1_probe', stage1_probe)]:
        prev = old[key]
        alpha_diff = abs(float(prev['selected_alpha']) - float(cur['selected_alpha']))
        d = {'selected_alpha_abs_diff': alpha_diff}
        if alpha_diff > atol:
            raise RuntimeError(f'Channel-wise C selected alpha differs from formal C for {key}: {alpha_diff}')
        for metric in ['mae', 'rmse', 'r2']:
            a = prev['test'][metric]
            b = cur['test'][metric]
            if a is None or b is None:
                metric_diff = 0.0 if a is None and b is None else float('inf')
            else:
                metric_diff = abs(float(a) - float(b))
            d[f'{metric}_abs_diff'] = metric_diff
            if metric_diff > atol:
                raise RuntimeError(f'Channel-wise C does not reproduce formal C for {key}/{metric}: diff={metric_diff}')
        diffs[key] = d
    return {'checked': True, 'tolerance': atol, 'diffs': diffs}

def comparison_metrics(base_probe, stage1_probe):
    out = {}

    def comp(b, s):
        return {'mae_reduction_pct': 100.0 * (b['mae'] - s['mae']) / max(b['mae'], 1e-12), 'rmse_reduction_pct': 100.0 * (b['rmse'] - s['rmse']) / max(b['rmse'], 1e-12), 'r2_gain': None if b['r2'] is None or s['r2'] is None else s['r2'] - b['r2']}
    out['overall'] = comp(base_probe['test'], stage1_probe['test'])
    out['by_channel'] = {ch: comp(base_probe['test_by_channel'][ch], stage1_probe['test_by_channel'][ch]) for ch in CHANNEL_NAMES}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./outputs/stage1/best')
    ap.add_argument('--qa-root', default='./outputs/qa/source')
    ap.add_argument('--output-dir', default='./outputs/stage1/support_evidence')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-train', type=int, default=None, help='Debug only; omit for all 500 P8 train records.')
    ap.add_argument('--max-valid', type=int, default=None, help='Debug only; omit for all 50 P8 valid records.')
    ap.add_argument('--max-test', type=int, default=None, help='Debug only; omit for all 100 P8 test records.')
    args = ap.parse_args()
    checkpoint = Path(args.checkpoint)
    cfg = json.loads((checkpoint / 'stage1_config.json').read_text(encoding='utf-8'))
    model_name = cfg.get('model_name', 'Qwen/Qwen2.5-VL-3B-Instruct')
    device = torch.device(args.device)
    normalizer = FlowNormalizer(str(checkpoint / 'normalization_stats.json'))
    processor = processor_from_checkpoint(checkpoint, model_name, cfg)
    limits = {'train': args.max_train, 'valid': args.max_valid, 'test': args.max_test}
    formal_expected = {'train': 500, 'valid': 50, 'test': 100}
    records = {}
    for split in ['train', 'valid', 'test']:
        rs = load_jsonl_records(str(Path(args.qa_root) / split / f'QA_{split}.jsonl'), template_ids=['P8'])
        if limits[split] is not None:
            rs = rs[:limits[split]]
        records[split] = rs
        print(f'P8 {split}: {len(rs)} records')
    if all((v is None for v in limits.values())):
        actual = {k: len(v) for k, v in records.items()}
        if actual != formal_expected:
            raise RuntimeError(f'Formal channel-wise C expects {formal_expected}, got {actual}')
    print('=== C-channel 1: original base VLM S_QA ===')
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    base.config.use_cache = False
    base_splits = {}
    base_ids = {}
    for split in ['train', 'valid', 'test']:
        X, Y, ids = extract_p8(base, processor, records[split], normalizer, cfg, device)
        base_splits[split] = (X, Y)
        base_ids[split] = ids
    del base
    gc.collect()
    torch.cuda.empty_cache()
    print('=== C-channel 2: Stage-1 adapted VLM S_QA ===')
    stage1, _, _ = load_stage1_checkpoint(args.checkpoint, device=str(device))
    stage1.eval()
    stage1_splits = {}
    stage1_ids = {}
    for split in ['train', 'valid', 'test']:
        X, Y, ids = extract_p8(stage1.vlm, processor, records[split], normalizer, cfg, device)
        stage1_splits[split] = (X, Y)
        stage1_ids[split] = ids
    for split in ['train', 'valid', 'test']:
        if base_ids[split] != stage1_ids[split]:
            raise RuntimeError(f'Sample-order mismatch in {split}')
        if not np.allclose(base_splits[split][1], stage1_splits[split][1], atol=0, rtol=0):
            raise RuntimeError(f'Target mismatch in {split}')
    alpha_grid = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    base_probe = evaluate_probe('base_vlm', base_splits, alpha_grid)
    stage1_probe = evaluate_probe('stage1_vlm', stage1_splits, alpha_grid)
    comparison = comparison_metrics(base_probe, stage1_probe)
    is_formal_run = all((v is None for v in limits.values()))
    if is_formal_run:
        consistency = verify_against_existing_formal(Path(args.output_dir), base_probe, stage1_probe)
    else:
        consistency = {'checked': False, 'reason': 'debug subset: formal-C consistency is only valid on the full 500/50/100 P8 split'}
    if consistency['checked']:
        print('Formal-C consistency check: PASS')
    else:
        print(f"Formal-C consistency check: SKIPPED ({consistency['reason']})")
    payload = {'analysis': 'C_linear_probe_on_P8_S_QA_by_channel', 'target': '24-D city-normalized future mean flow = 3 cities x 4 horizons x 2 channels; channel-wise metrics slice the same fitted 24-D ridge probe', 'channel_definition': {'0': 'inflow', '1': 'outflow'}, 'protocol': 'freeze VLM; extract pre-answer S_QA; fit one identical 24-D ridge probe on P8 train; select one alpha by overall P8-valid RMSE; evaluate once on P8 test; then decompose the same predictions into inflow/outflow.', 'record_counts': {k: len(v) for k, v in records.items()}, 'formal_C_consistency_check': consistency, 'base_probe': base_probe, 'stage1_probe': stage1_probe, 'comparison': comparison}
    lines = ['C. P8 S_QA linear probe -- inflow/outflow breakdown', 'The channel metrics are computed from the SAME 24-D ridge probe used by the formal C experiment.', 'No channel-specific alpha tuning or refitting is performed.', '', f"Base selected alpha   : {base_probe['selected_alpha']:.6g}", f"Stage1 selected alpha : {stage1_probe['selected_alpha']:.6g}", f"Overall Base MAE/RMSE/R2   : {base_probe['test']['mae']:.6f}/{base_probe['test']['rmse']:.6f}/{base_probe['test']['r2']:.6f}", f"Overall Stage1 MAE/RMSE/R2 : {stage1_probe['test']['mae']:.6f}/{stage1_probe['test']['rmse']:.6f}/{stage1_probe['test']['r2']:.6f}", '', f"{'Representation':<16} {'Channel':<9} {'MAE':>11} {'RMSE':>11} {'R2':>11}"]
    for label, probe in [('Base VLM', base_probe), ('Stage-1 VLM', stage1_probe)]:
        for ch in CHANNEL_NAMES:
            r = probe['test_by_channel'][ch]
            lines.append(f"{label:<16} {ch:<9} {r['mae']:>11.6f} {r['rmse']:>11.6f} {r['r2']:>11.6f}")
    lines.extend(['', 'Stage-1 improvement vs Base by channel:'])
    for ch in CHANNEL_NAMES:
        r = comparison['by_channel'][ch]
        lines.append(f"  {ch:<8} MAE reduction={r['mae_reduction_pct']:.2f}% | RMSE reduction={r['rmse_reduction_pct']:.2f}% | R2 gain={r['r2_gain']:.6f}")
    lines.extend(['', 'Per-city / per-channel normalized test metrics:'])
    for city in CITY_ORDER:
        lines.append(f'[{city}]')
        for ch in CHANNEL_NAMES:
            br = base_probe['test_by_city_channel'][city][ch]
            sr = stage1_probe['test_by_city_channel'][city][ch]
            lines.append(f"  {ch:<8} Base={br['mae']:.6f}/{br['rmse']:.6f}/{br['r2']:.6f} | Stage1={sr['mae']:.6f}/{sr['rmse']:.6f}/{sr['r2']:.6f}")
    write_json_and_text(Path(args.output_dir), 'C_sqa_linear_probe_by_channel', payload, '\n'.join(lines))
if __name__ == '__main__':
    main()
