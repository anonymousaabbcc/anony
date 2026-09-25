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
    X, Y, cities_meta = ([], [], [])
    for i, batch in enumerate(loader, start=1):
        item = ds[i - 1]
        target = p8_future_mean_target(item)
        batch = move_multimodal_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = vlm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=None, pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'], use_cache=False, return_dict=True)
        x = cap.pop(batch['readout_positions'])[0]
        X.append(x)
        Y.append(target)
        cities_meta.append(records[i - 1]['sample_id'])
        if i % 50 == 0:
            print(f'processed {i}/{len(loader)}')
    cap.close()
    return (np.stack(X), np.stack(Y), cities_meta)

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
    per_city = {}
    for ci, city in enumerate(CITY_ORDER):
        sl = slice(ci * 8, (ci + 1) * 8)
        per_city[city] = regression_metrics(pred[:, sl], Yte[:, sl])
    mean_pred = np.broadcast_to(Ytr.mean(axis=0, keepdims=True), Yte.shape)
    mean_baseline = regression_metrics(mean_pred, Yte)
    return {'name': name, 'selected_alpha': best['alpha'], 'validation_grid': candidates, 'test': overall, 'test_by_city': per_city, 'mean_target_baseline_test': mean_baseline}

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
    records = {}
    for split in ['train', 'valid', 'test']:
        rs = load_jsonl_records(str(Path(args.qa_root) / split / f'QA_{split}.jsonl'), template_ids=['P8'])
        if limits[split] is not None:
            rs = rs[:limits[split]]
        records[split] = rs
        print(f'P8 {split}: {len(rs)} records')
    print('=== C1: original base VLM S_QA ===')
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    base.config.use_cache = False
    base_splits = {}
    for split in ['train', 'valid', 'test']:
        X, Y, _ = extract_p8(base, processor, records[split], normalizer, cfg, device)
        base_splits[split] = (X, Y)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    print('=== C2: Stage-1 adapted VLM S_QA ===')
    stage1, _, _ = load_stage1_checkpoint(args.checkpoint, device=str(device))
    stage1.eval()
    stage1_splits = {}
    for split in ['train', 'valid', 'test']:
        X, Y, _ = extract_p8(stage1.vlm, processor, records[split], normalizer, cfg, device)
        stage1_splits[split] = (X, Y)
    for split in ['train', 'valid', 'test']:
        if not np.allclose(base_splits[split][1], stage1_splits[split][1], atol=0, rtol=0):
            raise RuntimeError(f'Target mismatch in {split}')
    alpha_grid = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    base_probe = evaluate_probe('base_vlm', base_splits, alpha_grid)
    stage1_probe = evaluate_probe('stage1_vlm', stage1_splits, alpha_grid)
    b = base_probe['test']
    s = stage1_probe['test']
    comparison = {'rmse_reduction_pct': 100.0 * (b['rmse'] - s['rmse']) / max(b['rmse'], 1e-12), 'mae_reduction_pct': 100.0 * (b['mae'] - s['mae']) / max(b['mae'], 1e-12), 'r2_gain': None if b['r2'] is None or s['r2'] is None else s['r2'] - b['r2']}
    payload = {'analysis': 'C_linear_probe_on_P8_S_QA', 'target': '24-D city-normalized future mean flow = 3 cities x 4 horizons x 2 channels, spatially averaged from P8 future tensors', 'protocol': 'freeze VLM; extract pre-answer S_QA; fit identical ridge probe on P8 train; select alpha on P8 valid; report once on P8 test', 'record_counts': {k: len(v) for k, v in records.items()}, 'base_probe': base_probe, 'stage1_probe': stage1_probe, 'comparison': comparison}
    lines = ['C. Linear probe on multi-city P8 S_QA', 'Target: 24-D normalized future mean flow (3 cities x 4 horizons x 2 channels).', 'Same frozen train/valid/test P8 records and same ridge protocol for Base and Stage-1.', 'Lower MAE/RMSE and higher R2 indicate more forecast-relevant information in S_QA.', '', f"{'Representation':<16} {'alpha':>10} {'Test MAE':>11} {'Test RMSE':>11} {'Test R2':>11}", f"{'Base VLM':<16} {base_probe['selected_alpha']:>10.4g} {b['mae']:>11.6f} {b['rmse']:>11.6f} {b['r2']:>11.6f}", f"{'Stage-1 VLM':<16} {stage1_probe['selected_alpha']:>10.4g} {s['mae']:>11.6f} {s['rmse']:>11.6f} {s['r2']:>11.6f}", '', f"MAE reduction vs Base: {comparison['mae_reduction_pct']:.2f}%", f"RMSE reduction vs Base: {comparison['rmse_reduction_pct']:.2f}%", f"R2 gain vs Base: {comparison['r2_gain']:.6f}", '', 'Per-city normalized test metrics:']
    for city in CITY_ORDER:
        br = base_probe['test_by_city'][city]
        sr = stage1_probe['test_by_city'][city]
        lines.append(f"  {city:<10} Base MAE/RMSE={br['mae']:.6f}/{br['rmse']:.6f} | Stage1={sr['mae']:.6f}/{sr['rmse']:.6f}")
    write_json_and_text(Path(args.output_dir), 'C_sqa_linear_probe', payload, '\n'.join(lines))
if __name__ == '__main__':
    main()
