import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .checkpoint import load_stage1_checkpoint
from .config import CITY_ORDER
from .collator import QwenStage1Collator
from .dataset import Stage1QADataset
from .normalization import FlowNormalizer, decode_prediction_value_pair

class MetricAccumulator:

    def __init__(self):
        self.sae = 0.0
        self.sse = 0.0
        self.n = 0

    def update(self, pred, target):
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if p.shape != y.shape:
            raise ValueError(f'Shape mismatch: {p.shape} vs {y.shape}')
        e = p - y
        self.sae += float(np.abs(e).sum())
        self.sse += float(np.square(e).sum())
        self.n += int(e.size)

    def result(self):
        if self.n == 0:
            return {'mae': None, 'rmse': None, 'n': 0}
        return {'mae': self.sae / self.n, 'rmse': math.sqrt(self.sse / self.n), 'n': self.n}

def move_batch(batch, device):
    for key in ['input_ids', 'attention_mask', 'labels', 'pixel_values', 'image_grid_thw', 'readout_positions', 'value_target', 'value_mask']:
        batch[key] = batch[key].to(device)
    return batch

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./outputs/stage1/best')
    ap.add_argument('--test-jsonl', default='./outputs/qa/source/test/QA_test.jsonl')
    ap.add_argument('--max-samples', type=int, default=None)
    args = ap.parse_args()
    device = torch.device('cuda')
    model, processor, cfg_dict = load_stage1_checkpoint(args.checkpoint, device=str(device))
    normalizer = FlowNormalizer(str(Path(args.checkpoint) / 'normalization_stats.json'))
    ds = Stage1QADataset(args.test_jsonl, normalizer, template_ids=['P3', 'P4', 'P7', 'P8'], max_samples=args.max_samples)
    collator = QwenStage1Collator(processor, max_seq_length=int(cfg_dict['max_seq_length']))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)
    metrics = defaultdict(MetricAccumulator)
    city_metrics = defaultdict(MetricAccumulator)
    for batch in loader:
        rec = batch['records'][0]
        tid = rec['template_id']
        batch = move_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'], readout_positions=batch['readout_positions'], value_target=batch['value_target'], value_mask=batch['value_mask'], tensor_targets=batch['tensor_targets'], return_predictions=True)
        if tid in {'P3', 'P7'}:
            pred_norm = out.value_predictions[0].float().cpu().numpy()
            pairs = decode_prediction_value_pair(rec, pred_norm, normalizer)
            for city, pair in pairs.items():
                metrics[tid].update(pair['pred'], pair['target'])
                city_metrics[tid, city].update(pair['pred'], pair['target'])
        elif tid in {'P4', 'P8'}:
            pred_dict = out.tensor_predictions[0]
            target_dict = batch['tensor_targets'][0]
            for city, pred_norm in pred_dict.items():
                target_norm = target_dict[city]
                pred_raw = normalizer.inverse_tensor(city, pred_norm.float().cpu().numpy())
                target_raw = normalizer.inverse_tensor(city, target_norm.float().cpu().numpy())
                metrics[tid].update(pred_raw, target_raw)
                city_metrics[tid, city].update(pred_raw, target_raw)
    result = {'by_template': {tid: metrics[tid].result() for tid in ['P3', 'P4', 'P7', 'P8']}, 'by_template_city': {f'{tid}/{city}': city_metrics[tid, city].result() for tid in ['P3', 'P4', 'P7', 'P8'] for city in CITY_ORDER if city_metrics[tid, city].n > 0}}
    macro = {}
    for tid in ['P3', 'P4', 'P7', 'P8']:
        rows = [city_metrics[tid, city].result() for city in CITY_ORDER if city_metrics[tid, city].n > 0]
        if rows:
            macro[tid] = {'mae': float(np.mean([x['mae'] for x in rows])), 'rmse': float(np.mean([x['rmse'] for x in rows]))}
    result['macro_city'] = macro
    print(json.dumps(result, indent=2))
    output = Path(args.checkpoint) / 'prediction_baseline_metrics.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved:', output)
if __name__ == '__main__':
    main()
