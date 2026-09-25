import argparse
import copy
import hashlib
import sys
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
from .support_common import ErrorAccumulator, RecordDataset, load_jsonl_records, move_multimodal_batch, write_json_and_text
QA_DIR = Path(__file__).resolve().parents[1] / 'QA_design'
if str(QA_DIR) not in sys.path:
    sys.path.insert(0, str(QA_DIR))
from split_loader import load_all_split, slice_window
from renderer import load_train_scales, render_city

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

def _temporal_derangement(sample_id: str, city: str, seed: int) -> np.ndarray:
    key = f'{seed}|{sample_id}|{city}'.encode('utf-8')
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
    rng = np.random.default_rng(local_seed)
    base = np.arange(8, dtype=np.int64)
    for _ in range(1000):
        perm = rng.permutation(base)
        if np.all(perm != base):
            return perm
    raise RuntimeError(f'Could not generate temporal derangement for {sample_id}/{city}')

def build_temporal_shuffle_images(records, output_dir: Path, split: str='test', seed: int=20260903):
    city_data = load_all_split(split)
    scales = load_train_scales()
    output_dir.mkdir(parents=True, exist_ok=True)
    shuffled_paths = {}
    manifest = {}
    for rec in records:
        sample_id = rec['sample_id']
        shuffled_paths[sample_id] = []
        manifest[sample_id] = {}
        for city, original_path in zip(rec['cities'], rec['image_paths']):
            start = int(rec['window_start_index'][city])
            window = slice_window(city_data[city], start)
            if int(window['X_hist'].shape[0]) != 8:
                raise ValueError(f'{sample_id}/{city}: expected 8 historical frames')
            perm = _temporal_derangement(sample_id, city, seed)
            shuffled_window = dict(window)
            shuffled_window['X_hist'] = np.asarray(window['X_hist'])[perm].copy()
            shuffled_window['time_hist'] = np.asarray(window['time_hist'])[perm].copy()
            out_path = (output_dir / Path(original_path).name).resolve()
            render_city(rec, city, shuffled_window, out_path, scales)
            shuffled_paths[sample_id].append(str(out_path))
            manifest[sample_id][city] = {'display_position_to_original_t': (perm + 1).tolist()}
    import json
    (output_dir / 'temporal_shuffle_manifest.json').write_text(json.dumps({'seed': seed, 'samples': manifest}, indent=2), encoding='utf-8')
    return shuffled_paths

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
                raise ValueError(f"{r['sample_id']}: expected {len(CITY_ORDER)} images")
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
    by_city = defaultdict(ErrorAccumulator)
    tensor_mse_sum = 0.0
    samples = 0
    model.eval()
    for i, batch in enumerate(loader, start=1):
        batch = move_multimodal_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'], readout_positions=batch['readout_positions'], value_target=batch['value_target'], value_mask=batch['value_mask'], tensor_targets=batch['tensor_targets'], return_predictions=True)
        tensor_mse_sum += float(out.tensor_loss.item())
        samples += 1
        pred_dict = out.tensor_predictions[0]
        target_dict = batch['tensor_targets'][0]
        for city in CITY_ORDER:
            pred_norm = pred_dict[city].float().cpu().numpy()
            target_norm = target_dict[city].float().cpu().numpy()
            pred_raw = normalizer.inverse_tensor(city, pred_norm)
            target_raw = normalizer.inverse_tensor(city, target_norm)
            pooled.update(pred_raw, target_raw)
            by_city[city].update(pred_raw, target_raw)
        if i % 25 == 0:
            print(f'processed {i}/{len(loader)}')
    city_results = {city: by_city[city].result() for city in CITY_ORDER}
    macro = {'mae': float(np.mean([city_results[c]['mae'] for c in CITY_ORDER])), 'rmse': float(np.mean([city_results[c]['rmse'] for c in CITY_ORDER]))}
    return {'pooled': pooled.result(), 'by_city': city_results, 'macro_city': macro, 'normalized_tensor_mse': tensor_mse_sum / max(samples, 1), 'samples': samples}

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
    device = torch.device(args.device)
    model, processor, cfg = load_stage1_checkpoint(args.checkpoint, device=str(device))
    normalizer = FlowNormalizer(str(Path(args.checkpoint) / 'normalization_stats.json'))
    blank_path = make_blank_image(records, Path(args.output_dir) / 'controls')
    temporal_shuffle_paths = build_temporal_shuffle_images(records, Path(args.output_dir) / 'controls' / 'temporal_shuffle', split='test', seed=args.temporal_shuffle_seed)
    conditions = ['correct', 'shuffled_time_same_city', 'permuted_city_images', 'blank_no_information']
    results = {}
    for cond in conditions:
        print(f'=== B: {cond} ===')
        perturbed = perturb_p8(records, cond, blank_path, temporal_shuffle_paths=temporal_shuffle_paths)
        results[cond] = evaluate_condition(model, processor, normalizer, perturbed, cfg, device)
    correct = results['correct']['macro_city']
    for cond in conditions[1:]:
        r = results[cond]['macro_city']
        r['mae_increase_vs_correct_pct'] = 100.0 * (r['mae'] - correct['mae']) / max(correct['mae'], 1e-12)
        r['rmse_increase_vs_correct_pct'] = 100.0 * (r['rmse'] - correct['rmse']) / max(correct['rmse'], 1e-12)
    payload = {'analysis': 'B_image_grounding_P8', 'interpretation': 'Performance should degrade when the same P8 historical frames are shown in a shuffled temporal order, city-mismatched, or replaced by no-information blank images.', 'conditions': results}
    lines = ['B. Image-grounding perturbation on P8 multi-city forecasting', 'All conditions keep the same prompt/targets. Only visual evidence changes.', 'shuffled_time_same_city keeps the same eight frames per city and changes only their display order.', 'blank_no_information preserves the multimodal token structure while removing image content.', '', f"{'Condition':<26} {'Macro MAE':>11} {'Macro RMSE':>12} {'Norm Tensor MSE':>16} {'MAE delta%':>11}"]
    for cond in conditions:
        r = results[cond]
        delta = r['macro_city'].get('mae_increase_vs_correct_pct', 0.0)
        lines.append(f"{cond:<26} {r['macro_city']['mae']:>11.6f} {r['macro_city']['rmse']:>12.6f} {r['normalized_tensor_mse']:>16.6f} {delta:>11.2f}")
    lines.append('')
    for cond in conditions:
        lines.append(f'[{cond}]')
        for city in CITY_ORDER:
            r = results[cond]['by_city'][city]
            lines.append(f"  {city:<10} MAE={r['mae']:.6f} RMSE={r['rmse']:.6f} n={r['n']}")
    write_json_and_text(Path(args.output_dir), 'B_image_grounding', payload, '\n'.join(lines))
if __name__ == '__main__':
    main()
