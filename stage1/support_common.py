import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import CITY_ORDER
from .normalization import encode_value_target, load_normalized_tensor_targets

class RecordDataset(Dataset):

    def __init__(self, records: Sequence[dict], normalizer):
        self.records = list(records)
        self.normalizer = normalizer
        if not self.records:
            raise ValueError('RecordDataset received no records')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        value_target, value_mask = encode_value_target(record, self.normalizer)
        tensor_targets = load_normalized_tensor_targets(record, self.normalizer)
        return {'record': record, 'value_target': value_target, 'value_mask': value_mask, 'tensor_targets': tensor_targets}

def load_jsonl_records(jsonl_path: str, template_ids: Optional[Sequence[str]]=None, max_per_template: Optional[int]=None) -> List[dict]:
    allowed = set(template_ids) if template_ids else None
    counts = defaultdict(int)
    out = []
    with Path(jsonl_path).open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            tid = rec['template_id']
            if allowed is not None and tid not in allowed:
                continue
            bad = [c for c in rec.get('cities', []) if c not in CITY_ORDER]
            if bad:
                raise ValueError(f"{rec.get('sample_id')}: non-source city in support analysis: {bad}")
            if max_per_template is not None and counts[tid] >= max_per_template:
                continue
            out.append(rec)
            counts[tid] += 1
    if not out:
        raise ValueError(f'No records loaded from {jsonl_path}')
    return out

def move_multimodal_batch(batch: dict, device: torch.device) -> dict:
    for key in ['input_ids', 'attention_mask', 'labels', 'pixel_values', 'image_grid_thw', 'readout_positions', 'value_target', 'value_mask']:
        batch[key] = batch[key].to(device)
    return batch

def causal_supervised_token_count(labels: torch.Tensor) -> int:
    if labels.shape[1] <= 1:
        return 0
    return int((labels[:, 1:] != -100).sum().item())

class ErrorAccumulator:

    def __init__(self):
        self.sae = 0.0
        self.sse = 0.0
        self.n = 0

    def update(self, pred, target):
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if p.shape != y.shape:
            raise ValueError(f'shape mismatch: {p.shape} vs {y.shape}')
        e = p - y
        self.sae += float(np.abs(e).sum())
        self.sse += float(np.square(e).sum())
        self.n += int(e.size)

    def result(self):
        if self.n == 0:
            return {'mae': None, 'rmse': None, 'n': 0}
        return {'mae': self.sae / self.n, 'rmse': math.sqrt(self.sse / self.n), 'n': self.n}

def write_json_and_text(output_dir: Path, stem: str, payload: dict, text: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f'{stem}.json'
    txt_path = output_dir / f'{stem}.txt'
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    txt_path.write_text(text.rstrip() + '\n', encoding='utf-8')
    print(text.rstrip())
    print(f'\nSaved JSON: {json_path}')
    print(f'Saved TXT : {txt_path}')

def p8_future_mean_target(item: dict) -> np.ndarray:
    target_dict = item['tensor_targets']
    chunks = []
    for city in CITY_ORDER:
        t = target_dict[city]
        v = t.float().mean(dim=(-1, -2)).reshape(-1).cpu().numpy()
        chunks.append(v.astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)

def ridge_fit_dual(X: np.ndarray, Y: np.ndarray, alpha: float):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True)
    x_std[x_std < 1e-06] = 1.0
    y_mean = Y.mean(axis=0, keepdims=True)
    Xs = (X - x_mean) / x_std
    Yc = Y - y_mean
    gram = Xs @ Xs.T
    A = gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64)
    dual = np.linalg.solve(A, Yc)
    W = Xs.T @ dual
    return {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'W': W}

def ridge_predict(model: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - model['x_mean']) / model['x_std']
    return Xs @ model['W'] + model['y_mean']

def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    e = p - y
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(np.square(e))))
    denom = float(np.sum(np.square(y - y.mean(axis=0, keepdims=True))))
    r2 = None if denom <= 0 else float(1.0 - np.sum(np.square(e)) / denom)
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'n': int(y.size)}
