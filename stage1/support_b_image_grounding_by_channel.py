import argparse
import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from .checkpoint import load_stage1_checkpoint
from .collator import QwenStage1Collator
from .config import CITY_ORDER
from .normalization import FlowNormalizer
from .support_b_image_grounding import build_temporal_shuffle_images
from .support_common import ErrorAccumulator, RecordDataset, load_jsonl_records, move_multimodal_batch, write_json_and_text
CHANNEL_NAMES = ('inflow', 'outflow')

def make_blank_image(records, output_dir: Path) -> str:
    first = None
    for rec in records:
        if rec.get('image_paths'):
            first = rec['image_paths'][0]
            break
    if first is None:
        raise ValueError('No image path found')
    with Image.open(first) as im:
        size = im.size
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'blank_no_information.png'
    Image.new('RGB', size, (127, 127, 127)).save(path)
    return str(path.resolve())

def perturb_p8(records, condition: str, blank_path: str, temporal_shuffle_paths=None):
    records = [copy.deepcopy(r) for r in records]
    n = len(records)
    if n < 2:
        raise ValueError('Need at least 2 P8 records for perturbation')
    if condition == 'correct':
        return records
    if condition == 'shuffled_time_same_city':
        if temporal_shuffle_paths is None:
            raise ValueError('temporal_shuffle_paths is required')
        for r in records:
            r['image_paths'] = list(temporal_shuffle_paths[r['sample_id']])
        return records
    if condition == 'permuted_city_images':
        for r in records:
            paths = list(r['image_paths'])
            if len(paths) != len(CITY_ORDER):
                raise ValueError(f"{r['sample_id']}: expected {len(CITY_ORDER)} images, got {len(paths)}")
            r['image_paths'] = paths[1:] + paths[:1]
        return records
    if condition == 'blank_no_information':
        for r in records:
            r['image_paths'] = [blank_path for _ in r['image_paths']]
        return records
    raise ValueError(condition)

@torch.no_grad()
def evaluate_condition(model, processor, normalizer, records, cfg, device):
    ds = RecordDataset(records, normalizer)
    collator = QwenStage1Collator(processor, max_seq_length=int(cfg['max_seq_length']))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)
    pooled = ErrorAccumulator()
    pooled_by_channel = {ch: ErrorAccumulator() for ch in CHANNEL_NAMES}
    by_city = defaultdict(ErrorAccumulator)
    by_city_channel = {city: {ch: ErrorAccumulator() for ch in CHANNEL_NAMES} for city in CITY_ORDER}
    norm_sse = 0.0
    norm_n = 0
    norm_channel_sse = {ch: 0.0 for ch in CHANNEL_NAMES}
    norm_channel_n = {ch: 0 for ch in CHANNEL_NAMES}
    samples = 0
    model.eval()
    for i, batch in enumerate(loader, start=1):
        batch = move_multimodal_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'], readout_positions=batch['readout_positions'], value_target=batch['value_target'], value_mask=batch['value_mask'], tensor_targets=batch['tensor_targets'], return_predictions=True)
        samples += 1
        pred_dict = out.tensor_predictions[0]
        target_dict = batch['tensor_targets'][0]
        for city in CITY_ORDER:
            pred_norm = pred_dict[city].float().cpu().numpy()
            target_norm = target_dict[city].float().cpu().numpy()
            if pred_norm.shape != target_norm.shape:
                raise ValueError(f'{city}: normalized prediction/target shape mismatch: {pred_norm.shape} vs {target_norm.shape}')
            if pred_norm.ndim != 4 or pred_norm.shape[1] != 2:
                raise ValueError(f'{city}: expected normalized tensor [4,2,H,W], got {pred_norm.shape}')
            diff_norm = pred_norm.astype(np.float64) - target_norm.astype(np.float64)
            norm_sse += float(np.square(diff_norm).sum())
            norm_n += int(diff_norm.size)
            for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
                d = diff_norm[:, channel_idx]
                norm_channel_sse[channel_name] += float(np.square(d).sum())
                norm_channel_n[channel_name] += int(d.size)
            pred_raw = normalizer.inverse_tensor(city, pred_norm)
            target_raw = normalizer.inverse_tensor(city, target_norm)
            pooled.update(pred_raw, target_raw)
            by_city[city].update(pred_raw, target_raw)
            for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
                pred_ch = pred_raw[:, channel_idx]
                target_ch = target_raw[:, channel_idx]
                pooled_by_channel[channel_name].update(pred_ch, target_ch)
                by_city_channel[city][channel_name].update(pred_ch, target_ch)
        if i % 25 == 0:
            print(f'processed {i}/{len(loader)}')
    city_results = {city: by_city[city].result() for city in CITY_ORDER}
    channel_results = {ch: pooled_by_channel[ch].result() for ch in CHANNEL_NAMES}
    city_channel_results = {city: {ch: by_city_channel[city][ch].result() for ch in CHANNEL_NAMES} for city in CITY_ORDER}
    macro = {'mae': float(np.mean([city_results[c]['mae'] for c in CITY_ORDER])), 'rmse': float(np.mean([city_results[c]['rmse'] for c in CITY_ORDER]))}
    macro_by_channel = {}
    for ch in CHANNEL_NAMES:
        macro_by_channel[ch] = {'mae': float(np.mean([city_channel_results[c][ch]['mae'] for c in CITY_ORDER])), 'rmse': float(np.mean([city_channel_results[c][ch]['rmse'] for c in CITY_ORDER]))}
    normalized_tensor_mse = norm_sse / max(norm_n, 1)
    normalized_tensor_mse_by_channel = {ch: norm_channel_sse[ch] / max(norm_channel_n[ch], 1) for ch in CHANNEL_NAMES}
    return {'pooled': pooled.result(), 'pooled_by_channel': channel_results, 'by_city': city_results, 'by_city_channel': city_channel_results, 'macro_city': macro, 'macro_city_by_channel': macro_by_channel, 'normalized_tensor_mse': normalized_tensor_mse, 'normalized_tensor_mse_by_channel': normalized_tensor_mse_by_channel, 'samples': samples}

def verify_against_existing_formal(output_dir: Path, results: dict, atol: float=1e-06):
    path = output_dir / 'B_image_grounding.json'
    if not path.exists():
        return {'checked': False, 'reason': 'formal B JSON not found'}
    import json
    old = json.loads(path.read_text(encoding='utf-8'))
    diffs = {}
    for cond, cur in results.items():
        if cond not in old.get('conditions', {}):
            raise RuntimeError(f'Existing formal B JSON is missing condition {cond}')
        old_macro = old['conditions'][cond]['macro_city']
        cur_macro = cur['macro_city']
        mae_diff = abs(float(old_macro['mae']) - float(cur_macro['mae']))
        rmse_diff = abs(float(old_macro['rmse']) - float(cur_macro['rmse']))
        diffs[cond] = {'mae_abs_diff': mae_diff, 'rmse_abs_diff': rmse_diff}
        if mae_diff > atol or rmse_diff > atol:
            raise RuntimeError(f'Channel-wise B does not reproduce formal B for {cond}: MAE diff={mae_diff}, RMSE diff={rmse_diff}')
    return {'checked': True, 'tolerance': atol, 'diffs': diffs}

def add_degradation(results, conditions):
    correct = results['correct']
    for cond in conditions[1:]:
        cur = results[cond]
        cur['macro_city']['mae_increase_vs_correct_pct'] = 100.0 * (cur['macro_city']['mae'] - correct['macro_city']['mae']) / max(correct['macro_city']['mae'], 1e-12)
        cur['macro_city']['rmse_increase_vs_correct_pct'] = 100.0 * (cur['macro_city']['rmse'] - correct['macro_city']['rmse']) / max(correct['macro_city']['rmse'], 1e-12)
        for ch in CHANNEL_NAMES:
            base_ch = correct['macro_city_by_channel'][ch]
            cur_ch = cur['macro_city_by_channel'][ch]
            cur_ch['mae_increase_vs_correct_pct'] = 100.0 * (cur_ch['mae'] - base_ch['mae']) / max(base_ch['mae'], 1e-12)
            cur_ch['rmse_increase_vs_correct_pct'] = 100.0 * (cur_ch['rmse'] - base_ch['rmse']) / max(base_ch['rmse'], 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./outputs/stage1/best')
    ap.add_argument('--test-jsonl', default='./outputs/qa/source/test/QA_test.jsonl')
    ap.add_argument('--output-dir', default='./outputs/stage1/support_evidence')
    ap.add_argument('--max-samples', type=int, default=None, help='Debug only; omit for all 100 P8 test records.')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--temporal-shuffle-seed', type=int, default=20260903)
    args = ap.parse_args()
    records = load_jsonl_records(args.test_jsonl, template_ids=['P8'])
    if args.max_samples is not None:
        records = records[:args.max_samples]
    print(f'P8 test records: {len(records)}')
    if args.max_samples is None and len(records) != 100:
        raise RuntimeError(f'Formal channel-wise B expects exactly 100 P8 test records, got {len(records)}')
    device = torch.device(args.device)
    model, processor, cfg = load_stage1_checkpoint(args.checkpoint, device=str(device))
    normalizer = FlowNormalizer(str(Path(args.checkpoint) / 'normalization_stats.json'))
    blank_path = make_blank_image(records, Path(args.output_dir) / 'controls')
    temporal_shuffle_paths = build_temporal_shuffle_images(records, Path(args.output_dir) / 'controls' / 'temporal_shuffle', split='test', seed=args.temporal_shuffle_seed)
    conditions = ['correct', 'shuffled_time_same_city', 'permuted_city_images', 'blank_no_information']
    results = {}
    for cond in conditions:
        print(f'=== B-channel: {cond} ===')
        perturbed = perturb_p8(records, cond, blank_path, temporal_shuffle_paths=temporal_shuffle_paths)
        results[cond] = evaluate_condition(model, processor, normalizer, perturbed, cfg, device)
    add_degradation(results, conditions)
    if args.max_samples is None:
        consistency = verify_against_existing_formal(Path(args.output_dir), results)
    else:
        consistency = {'checked': False, 'reason': 'debug subset: formal-B consistency is only valid on the full 100-record P8 test set'}
    if consistency['checked']:
        print('Formal-B consistency check: PASS')
    else:
        print(f"Formal-B consistency check: SKIPPED ({consistency['reason']})")
    payload = {'analysis': 'B_image_grounding_P8_by_channel', 'channel_definition': {'0': 'inflow', '1': 'outflow'}, 'protocol': 'Exact frozen support-B P8 visual perturbation protocol; temporal shuffle keeps the same eight frames and changes only their order; raw MAE/RMSE are additionally decomposed by inflow/outflow. Normalized Tensor MSE is recomputed from prediction-target tensors overall and by channel.', 'record_count': len(records), 'formal_B_consistency_check': consistency, 'conditions': results}
    lines = ['B. P8 visual grounding -- inflow/outflow breakdown', 'Same frozen Stage-1 checkpoint and same P8 perturbation protocol as the formal B experiment.', 'Raw metrics are inverse-normalized. Channel 0=inflow; channel 1=outflow.', '', f"{'Condition':<26} {'In MAE':>10} {'In RMSE':>10} {'In nMSE':>10} {'Out MAE':>10} {'Out RMSE':>10} {'Out nMSE':>10}"]
    for cond in conditions:
        r = results[cond]
        ri = r['macro_city_by_channel']['inflow']
        ro = r['macro_city_by_channel']['outflow']
        lines.append(f"{cond:<26} {ri['mae']:>10.6f} {ri['rmse']:>10.6f} {r['normalized_tensor_mse_by_channel']['inflow']:>10.6f} {ro['mae']:>10.6f} {ro['rmse']:>10.6f} {r['normalized_tensor_mse_by_channel']['outflow']:>10.6f}")
    lines.extend(['', 'MAE degradation vs correct images:'])
    for cond in conditions[1:]:
        ri = results[cond]['macro_city_by_channel']['inflow']
        ro = results[cond]['macro_city_by_channel']['outflow']
        lines.append(f"  {cond:<24} inflow={ri['mae_increase_vs_correct_pct']:.2f}%  outflow={ro['mae_increase_vs_correct_pct']:.2f}%")
    lines.extend(['', 'Per-city / per-channel raw test metrics:'])
    for cond in conditions:
        lines.append(f'[{cond}]')
        for city in CITY_ORDER:
            ri = results[cond]['by_city_channel'][city]['inflow']
            ro = results[cond]['by_city_channel'][city]['outflow']
            lines.append(f"  {city:<10} inflow MAE/RMSE={ri['mae']:.6f}/{ri['rmse']:.6f} n={ri['n']} | outflow MAE/RMSE={ro['mae']:.6f}/{ro['rmse']:.6f} n={ro['n']}")
    write_json_and_text(Path(args.output_dir), 'B_image_grounding_by_channel', payload, '\n'.join(lines))
if __name__ == '__main__':
    main()
