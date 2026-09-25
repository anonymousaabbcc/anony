from __future__ import annotations
import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from .checkpoint import load_stage2_weights
from .config import CHANNELS, FORECAST_HORIZONS, SOURCE_CITY_ORDER, Stage2Config
from .model import Stage2UrbanForecaster, _spatial_gradient_l1
from .normalization import TorchFlowNormalizer
from .qa_compat import import_qa_design
TARGET_CITY = 'BIKEDC'
TARGET_SHAPE = (16, 8)
TARGET_TRAIN_WINDOWS = 1525
TARGET_VALID_WINDOWS = 205
TARGET_TEST_WINDOWS = 445
SUPPORT_SEED = 2026
SOURCE_SLOT = 'NYC-BIKE'
P8_STAGE2_QUESTION = "Using each city's 8-hour history, forecast the complete fine-grained inflow and outflow maps for horizons [1, 2, 3, 4]."

def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _fmt_time(x) -> str:
    return str(x).replace('T', ' ')

class TargetSupportNormalizer:

    def __init__(self, stats: dict):
        self.stats = stats

    def mean_std(self, city: str, channel: str):
        if city != TARGET_CITY:
            raise KeyError(city)
        item = self.stats['cities'][TARGET_CITY][channel]
        return (float(item['mean']), float(item['std']))

    def normalize(self, city: str, x: torch.Tensor) -> torch.Tensor:
        if city != TARGET_CITY:
            raise KeyError(city)
        if x.ndim not in (4, 5):
            raise ValueError(f'{city}: expected [...,2,H,W], got {tuple(x.shape)}')
        axis = 1 if x.ndim == 4 else 2
        y = x.float().clone()
        for ci, ch in enumerate(CHANNELS):
            mean, std = self.mean_std(city, ch)
            if x.ndim == 4:
                y[:, ci] = (y[:, ci] - mean) / std
            else:
                y[:, :, ci] = (y[:, :, ci] - mean) / std
        return y

    def inverse(self, city: str, x: torch.Tensor) -> torch.Tensor:
        if city != TARGET_CITY:
            raise KeyError(city)
        axis = 1 if x.ndim == 4 else 2
        if x.shape[axis] != 2:
            raise ValueError(tuple(x.shape))
        y = x.float().clone()
        for ci, ch in enumerate(CHANNELS):
            mean, std = self.mean_std(city, ch)
            if x.ndim == 4:
                y[:, ci] = y[:, ci] * std + mean
            else:
                y[:, :, ci] = y[:, :, ci] * std + mean
        return y

def _load_target_split(api, split: str):
    cities = api['load_all_split'](split, city_order=[TARGET_CITY])
    if TARGET_CITY not in cities:
        raise RuntimeError(f'{TARGET_CITY} missing from QA_design load_all_split({split!r}); available={list(cities.keys())}')
    starts = np.asarray(api['valid_window_starts'](cities[TARGET_CITY]), dtype=np.int64)
    expected = {'train': TARGET_TRAIN_WINDOWS, 'valid': TARGET_VALID_WINDOWS, 'test': TARGET_TEST_WINDOWS}[split]
    if len(starts) != expected:
        raise RuntimeError(f'{TARGET_CITY}/{split}: found {len(starts)} windows, expected {expected}')
    return (cities[TARGET_CITY], starts)

def nested_support_starts(api, k: int, seed: int=SUPPORT_SEED):
    city_data, starts = _load_target_split(api, 'train')
    if not 1 <= k <= len(starts):
        raise ValueError(f'K={k}, train windows={len(starts)}')
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(starts))
    chosen = starts[order[:int(k)]]
    return (city_data, starts, chosen, order[:int(k)])

def fit_support_normalizer(api, k: int, seed: int=SUPPORT_SEED) -> dict:
    city_data, _, chosen, support_indices = nested_support_starts(api, k, seed)
    vals = {ch: [] for ch in CHANNELS}
    for start in chosen.tolist():
        w = api['slice_window'](city_data, int(start))
        x = np.asarray(w['X_hist'], dtype=np.float64)
        y = np.asarray(w['Y_future'], dtype=np.float64)
        for ci, ch in enumerate(CHANNELS):
            vals[ch].append(x[:, ci].reshape(-1))
            vals[ch].append(y[:, ci].reshape(-1))
    stats = {'cities': {TARGET_CITY: {}}, 'support_k': int(k), 'support_seed': int(seed)}
    for ch in CHANNELS:
        z = np.concatenate(vals[ch])
        mean = float(z.mean())
        std = float(z.std())
        if not np.isfinite(std) or std < 1e-06:
            std = 1.0
        stats['cities'][TARGET_CITY][ch] = {'mean': mean, 'std': std, 'count': int(z.size)}
    stats['support_window_indices'] = [int(x) for x in support_indices.tolist()]
    stats['support_start_indices'] = [int(x) for x in chosen.tolist()]
    return stats

def _source_reference_scales(api):
    errors = []
    try:
        return (api['load_train_scales'](city_order=[TARGET_CITY], reference_city_order=[SOURCE_SLOT]), 'reference_city_order=NYC-BIKE')
    except Exception as e:
        errors.append(repr(e))
    try:
        scales = api['load_train_scales'](city_order=[SOURCE_SLOT], reference_city_order=None)
        scales = copy.deepcopy(scales)
        if isinstance(scales, dict) and SOURCE_SLOT in scales:
            scales[TARGET_CITY] = copy.deepcopy(scales[SOURCE_SLOT])
            return (scales, 'aliased NYC-BIKE source visual scale')
    except Exception as e:
        errors.append(repr(e))
    raise RuntimeError('Could not construct source-reference visual scale for BIKEDC. ' + ' | '.join(errors))

def prepare_target_visual_cache(*, cfg: Stage2Config, api, cache_root: Path, max_k: int=200, support_seed: int=SUPPORT_SEED, overwrite: bool=False):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    scales, scale_policy = _source_reference_scales(api)
    record = {'block': 'Prediction', 'region_level': 'finegrain'}
    train_data, _, support_starts, _ = nested_support_starts(api, max_k, support_seed)
    split_jobs = {'train': (train_data, support_starts)}
    for split in ('valid', 'test'):
        d, starts = _load_target_split(api, split)
        split_jobs[split] = (d, starts)
    summary = {'target_city': TARGET_CITY, 'max_support_k': int(max_k), 'support_seed': int(support_seed), 'visual_scale_policy': scale_policy, 'splits': {}}
    for split, (city_data, starts) in split_jobs.items():
        out_dir = cache_root / split / TARGET_CITY
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered = 0
        skipped = 0
        for j, start in enumerate(starts.tolist()):
            out = out_dir / f'{int(start):07d}.png'
            if out.exists() and (not overwrite):
                skipped += 1
                continue
            w = api['slice_window'](city_data, int(start))
            api['render_city'](record, TARGET_CITY, w, out, scales)
            rendered += 1
            if (j + 1) % 100 == 0:
                print(f'CACHE {split}/{TARGET_CITY}: {j + 1}/{len(starts)} rendered={rendered} skipped={skipped}', flush=True)
        missing = [int(s) for s in starts.tolist() if not (out_dir / f'{int(s):07d}.png').exists()]
        if missing:
            raise RuntimeError(f'cache incomplete {split}: first missing={missing[:10]}')
        summary['splits'][split] = {'windows': int(len(starts)), 'rendered': int(rendered), 'skipped': int(skipped), 'directory': str(out_dir)}
    manifest = cache_root / 'manifest.json'
    manifest.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('BIKEDC VISUAL CACHE READY:', manifest, flush=True)
    return summary

class BIKEDCFewShotDataset(Dataset):

    def __init__(self, *, split: str, api, source_normalizer: TorchFlowNormalizer, target_normalizer: TargetSupportNormalizer, source_visual_cache_root: Path, target_visual_cache_root: Path, support_k: int, support_seed: int):
        self.split = split
        self.api = api
        self.slice_window = api['slice_window']
        self.source_normalizer = source_normalizer
        self.target_normalizer = target_normalizer
        self.source_visual_cache_root = Path(source_visual_cache_root)
        self.target_visual_cache_root = Path(target_visual_cache_root)
        source_train = api['load_all_split']('train')
        self.source_data = {}
        self.source_starts = {}
        for city in ('NYCTAXI', 'BIKECHI'):
            if city not in source_train:
                raise RuntimeError(f'{city} missing from source train split')
            self.source_data[city] = source_train[city]
            self.source_starts[city] = np.asarray(api['valid_window_starts'](source_train[city]), dtype=np.int64)
        if split == 'train':
            target_data, _, starts, _ = nested_support_starts(api, support_k, support_seed)
            self.target_data = target_data
            self.target_starts = np.asarray(starts, dtype=np.int64)
        else:
            self.target_data, self.target_starts = _load_target_split(api, split)

    def __len__(self):
        return len(self.target_starts)

    def _source_start(self, city: str, i: int) -> int:
        starts = self.source_starts[city]
        if city == 'NYCTAXI':
            j = (37 * int(i) + 11) % len(starts)
        else:
            j = (53 * int(i) + 17) % len(starts)
        return int(starts[j])

    def __getitem__(self, i: int):
        i = int(i)
        tstart = int(self.target_starts[i])
        tw = self.slice_window(self.target_data, tstart)
        tx = torch.from_numpy(np.asarray(tw['X_hist'], dtype=np.float32))
        ty = torch.from_numpy(np.asarray(tw['Y_future'], dtype=np.float32))
        tx = self.target_normalizer.normalize(TARGET_CITY, tx)
        ty = self.target_normalizer.normalize(TARGET_CITY, ty)
        timg = self.target_visual_cache_root / self.split / TARGET_CITY / f'{tstart:07d}.png'
        if not timg.exists():
            raise FileNotFoundError(timg)
        cities = {}
        for city in ('NYCTAXI', 'BIKECHI'):
            start = self._source_start(city, i)
            w = self.slice_window(self.source_data[city], start)
            x = torch.from_numpy(np.asarray(w['X_hist'], dtype=np.float32))
            x = self.source_normalizer.normalize(city, x)
            img = self.source_visual_cache_root / 'train' / city / f'{start:07d}.png'
            if not img.exists():
                raise FileNotFoundError(f'Missing source visual cache {img}. The source full model should have built train visuals already.')
            cities[city] = {'x': x, 'image_path': str(img), 'time_hist': np.asarray(w['time_hist']).astype('datetime64[h]').astype(str).tolist(), 'time_future': np.asarray(w['time_future']).astype('datetime64[h]').astype(str).tolist(), 'start_index': int(start)}
        cities[TARGET_CITY] = {'x': tx, 'y': ty, 'image_path': str(timg), 'time_hist': np.asarray(tw['time_hist']).astype('datetime64[h]').astype(str).tolist(), 'time_future': np.asarray(tw['time_future']).astype('datetime64[h]').astype(str).tolist(), 'start_index': int(tstart)}
        return {'target_index': i, 'cities': cities}

def build_heldout_messages(sample: dict, system_prompt: str):
    order = ('NYCTAXI', 'BIKECHI', TARGET_CITY)
    chunks = []
    for city in order:
        x = sample['cities'][city]
        chunks.append(f"{city}: history {_fmt_time(x['time_hist'][0])} to {_fmt_time(x['time_hist'][-1])}; forecast {_fmt_time(x['time_future'][0])} to {_fmt_time(x['time_future'][-1])}")
    system = system_prompt + ' Current held-out adaptation windows contain two source-city histories and one BIKEDC target history, all at one-hour intervals; absolute dates do not need to match. ' + ' | '.join(chunks) + '.'
    user = []
    for city in order:
        user.append({'type': 'text', 'text': f'{city} historical urban flow maps:'})
        user.append({'type': 'image', 'image': Path(sample['cities'][city]['image_path']).resolve().as_uri()})
    user.append({'type': 'text', 'text': P8_STAGE2_QUESTION})
    return [{'role': 'system', 'content': [{'type': 'text', 'text': system}]}, {'role': 'user', 'content': user}]

class BIKEDCCollator:

    def __init__(self, processor, system_prompt: str, max_seq_length: int=8192):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.system_prompt = system_prompt
        self.max_seq_length = int(max_seq_length)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _load_images(paths: Sequence[str]):
        out = []
        for p in paths:
            with Image.open(p) as im:
                out.append(im.convert('RGB').copy())
        return out

    def _encode(self, sample):
        messages = build_heldout_messages(sample, self.system_prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        order = ('NYCTAXI', 'BIKECHI', TARGET_CITY)
        images = self._load_images([sample['cities'][c]['image_path'] for c in order])
        enc = self.processor(text=[text], images=images, padding=False, return_tensors='pt')
        ids = enc['input_ids'][0]
        if ids.numel() > self.max_seq_length:
            raise RuntimeError(f'held-out prompt length={ids.numel()} > max_seq_length={self.max_seq_length}')
        return {'input_ids': ids, 'attention_mask': enc['attention_mask'][0], 'pixel_values': enc['pixel_values'], 'image_grid_thw': enc['image_grid_thw']}

    def __call__(self, samples: List[dict]):
        encoded = [self._encode(s) for s in samples]
        b = len(samples)
        max_len = max((x['input_ids'].numel() for x in encoded))
        input_ids = torch.full((b, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        prompt_lengths = torch.zeros((b,), dtype=torch.long)
        for i, x in enumerate(encoded):
            n = x['input_ids'].numel()
            input_ids[i, :n] = x['input_ids']
            attention_mask[i, :n] = x['attention_mask']
            prompt_lengths[i] = n
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'prompt_lengths': prompt_lengths, 'pixel_values': torch.cat([x['pixel_values'] for x in encoded], dim=0), 'image_grid_thw': torch.cat([x['image_grid_thw'] for x in encoded], dim=0), 'cities': {}, 'target_index': torch.tensor([s['target_index'] for s in samples], dtype=torch.long)}
        for city in ('NYCTAXI', 'BIKECHI', TARGET_CITY):
            item = {'x': torch.stack([s['cities'][city]['x'] for s in samples], dim=0)}
            if city == TARGET_CITY:
                item['y'] = torch.stack([s['cities'][city]['y'] for s in samples], dim=0)
            batch['cities'][city] = item
        return batch

def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj

class BIKEDCAdapter(nn.Module):

    def __init__(self, source: Stage2UrbanForecaster):
        super().__init__()
        self.source = source
        for p in self.source.parameters():
            p.requires_grad_(False)
        self.source.eval()
        key = SOURCE_SLOT.replace('-', '_')
        self.target_encoder = copy.deepcopy(self.source.multi_city.city_encoders[key])
        self.target_head = copy.deepcopy(self.source.regression.heads[key])
        self.target_encoder.city = TARGET_CITY
        self.target_head.city = TARGET_CITY
        self.target_encoder.train()
        self.target_head.train()
        for p in self.target_encoder.parameters():
            p.requires_grad_(True)
        for p in self.target_head.parameters():
            p.requires_grad_(True)

    def train(self, mode: bool=True):
        self.source.eval()
        self.target_encoder.train(mode)
        self.target_head.train(mode)
        return self

    def trainable_parameters(self):
        return [p for p in list(self.target_encoder.parameters()) + list(self.target_head.parameters()) if p.requires_grad]

    def forward(self, batch):
        src = self.source
        context = src.bridge.prepare_context(batch)
        with torch.no_grad():
            tok_taxi, _ = src.multi_city.city_encoders['NYCTAXI'](batch['cities']['NYCTAXI']['x'])
            tok_chi, _ = src.multi_city.city_encoders['BIKECHI'](batch['cities']['BIKECHI']['x'])
        tx = batch['cities'][TARGET_CITY]['x']
        tok_target, map_target = self.target_encoder(tx)
        city_tokens = torch.stack([tok_taxi.detach(), tok_chi.detach(), tok_target], dim=1)
        for block in src.multi_city.city_blocks:
            city_tokens = block(city_tokens)
        global_h = city_tokens.reshape(city_tokens.shape[0], -1)
        z, _, _ = src.alignment(global_h, context.s_qa)
        if src.cfg.semantic_mode == 'dsr_reentry':
            r = src.bridge.reenter(context, z)
        else:
            r = z
        semantic = src.regression.semantic_context(r, z)
        pred = self.target_head(dense_history=map_target, city_context=city_tokens[:, 2], semantic_context=semantic, x_hist=tx)
        return pred

def target_loss(cfg, pred: torch.Tensor, target: torch.Tensor):
    l1 = F.l1_loss(pred.float(), target.float(), reduction='mean')
    mse = F.mse_loss(pred.float(), target.float(), reduction='mean')
    spatial = _spatial_gradient_l1(pred, target)
    total = cfg.loss_l1_weight * l1 + cfg.loss_mse_weight * mse + cfg.loss_spatial_grad_weight * spatial
    return (total, {'l1': float(l1.detach()), 'mse': float(mse.detach()), 'spatial': float(spatial.detach())})

class SingleCityMetrics:

    def __init__(self, normalizer: TargetSupportNormalizer):
        self.normalizer = normalizer
        self.stats = {ch: {'abs': 0.0, 'sq': 0.0, 'den': 0.0, 'n': 0} for ch in CHANNELS}
        self.by_horizon = {int(h): {ch: {'abs': 0.0, 'sq': 0.0, 'den': 0.0, 'n': 0} for ch in CHANNELS} for h in FORECAST_HORIZONS}

    @staticmethod
    def _add(stat, pred, target):
        err = pred - target
        stat['abs'] += float(err.abs().sum().cpu())
        stat['sq'] += float((err * err).sum().cpu())
        stat['den'] += float(target.abs().sum().cpu())
        stat['n'] += int(err.numel())

    @torch.no_grad()
    def update(self, pred_norm, target_norm):
        for ci, ch in enumerate(CHANNELS):
            mean, std = self.normalizer.mean_std(TARGET_CITY, ch)
            pred = pred_norm[:, :, ci].float() * std + mean
            target = target_norm[:, :, ci].float() * std + mean
            self._add(self.stats[ch], pred, target)
            for hi, h in enumerate(FORECAST_HORIZONS):
                self._add(self.by_horizon[int(h)][ch], pred[:, hi], target[:, hi])

    @staticmethod
    def _finish(x):
        if x['n'] <= 0:
            raise RuntimeError('empty metric')
        return {'mae': x['abs'] / x['n'], 'rmse': math.sqrt(x['sq'] / x['n']), 'wmape': x['abs'] / max(x['den'], 1e-12), 'n': int(x['n'])}

    def compute(self):
        out = {'city': TARGET_CITY, 'inflow': self._finish(self.stats['inflow']), 'outflow': self._finish(self.stats['outflow']), 'by_horizon': {}}
        for h in FORECAST_HORIZONS:
            out['by_horizon'][f'h{h}'] = {ch: self._finish(self.by_horizon[int(h)][ch]) for ch in CHANNELS}
        out['selection_mae'] = 0.5 * (out['inflow']['mae'] + out['outflow']['mae'])
        out['avg_rmse'] = 0.5 * (out['inflow']['rmse'] + out['outflow']['rmse'])
        return out

def format_target_metrics(m: dict) -> str:
    lines = ['BIKEDC AGGREGATE OVER HORIZONS 1-4', '             Inflow MAE   Inflow RMSE   Outflow MAE  Outflow RMSE   Avg MAE', '-------------------------------------------------------------------------------', f"BIKEDC       {m['inflow']['mae']:11.6f} {m['inflow']['rmse']:13.6f} {m['outflow']['mae']:12.6f} {m['outflow']['rmse']:13.6f} {m['selection_mae']:9.6f}", '', 'BY HORIZON', 'Horizon       Inflow MAE   Inflow RMSE   Outflow MAE  Outflow RMSE', '-------------------------------------------------------------------']
    for h in FORECAST_HORIZONS:
        x = m['by_horizon'][f'h{h}']
        lines.append(f"h{h:<11d} {x['inflow']['mae']:11.6f} {x['inflow']['rmse']:13.6f} {x['outflow']['mae']:12.6f} {x['outflow']['rmse']:13.6f}")
    return '\n'.join(lines)

def save_target_metrics(m: dict, prefix: Path):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
    prefix.with_suffix('.txt').write_text(format_target_metrics(m) + '\n', encoding='utf-8')

def make_loader(*, split: str, api, cfg, processor, system_prompt, source_normalizer, target_normalizer, target_visual_cache_root, k, support_seed, batch_size, num_workers, shuffle):
    ds = BIKEDCFewShotDataset(split=split, api=api, source_normalizer=source_normalizer, target_normalizer=target_normalizer, source_visual_cache_root=cfg.visual_cache_root, target_visual_cache_root=target_visual_cache_root, support_k=k, support_seed=support_seed)
    collator = BIKEDCCollator(processor, system_prompt=system_prompt, max_seq_length=int(getattr(cfg, 'max_seq_length', 8192)))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=int(num_workers), pin_memory=True, collate_fn=collator, drop_last=False)

@torch.no_grad()
def evaluate_target(adapter, loader, normalizer, device):
    adapter.eval()
    metrics = SingleCityMetrics(normalizer)
    t0 = time.time()
    for batch in loader:
        batch = _to_device(batch, device)
        pred = adapter(batch)
        metrics.update(pred, batch['cities'][TARGET_CITY]['y'])
    out = metrics.compute()
    out['windows'] = int(len(loader.dataset))
    out['seconds'] = float(time.time() - t0)
    out['windows_per_second'] = out['windows'] / max(out['seconds'], 1e-12)
    return out

def save_adapter(path: Path, adapter: BIKEDCAdapter, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'target_encoder': {k: v.detach().cpu() for k, v in adapter.target_encoder.state_dict().items()}, 'target_head': {k: v.detach().cpu() for k, v in adapter.target_head.state_dict().items()}, 'meta': meta}, path)

def load_adapter(path: Path, adapter: BIKEDCAdapter):
    p = torch.load(path, map_location='cpu')
    adapter.target_encoder.load_state_dict(p['target_encoder'], strict=True)
    adapter.target_head.load_state_dict(p['target_head'], strict=True)
    return p.get('meta', {})

def train_one(args):
    source_ckpt = Path(args.source_checkpoint)
    meta_path = source_ckpt / 'metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    source_meta = json.loads(meta_path.read_text(encoding='utf-8'))
    cfg = Stage2Config.from_dict(source_meta.get('config', {}))
    device = torch.device(args.device if args.device else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    _seed_all(args.train_seed + int(args.k))
    api = import_qa_design()
    target_cache = Path(args.target_visual_cache_root)
    if args.prepare_cache or args.prepare_only:
        prepare_target_visual_cache(cfg=cfg, api=api, cache_root=target_cache, max_k=args.max_k, support_seed=args.support_seed, overwrite=args.overwrite_cache)
        if args.prepare_only:
            return
    out_dir = Path(args.output_root) / f'K{int(args.k)}'
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = fit_support_normalizer(api, args.k, args.support_seed)
    (out_dir / 'target_normalization.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    (out_dir / 'support_indices.json').write_text(json.dumps({'K': int(args.k), 'seed': int(args.support_seed), 'window_indices': stats['support_window_indices'], 'start_indices': stats['support_start_indices']}, indent=2), encoding='utf-8')
    target_norm = TargetSupportNormalizer(stats)
    source_norm = TorchFlowNormalizer(cfg.normalization_stats)
    print('Loading source UrbanBind checkpoint:', source_ckpt, flush=True)
    source = Stage2UrbanForecaster(cfg, device=device)
    load_stage2_weights(source_ckpt, source)
    source.eval()
    adapter = BIKEDCAdapter(source).to(device)
    qa = import_qa_design()
    system_prompt = qa['system_prompt']
    train_loader = make_loader(split='train', api=api, cfg=cfg, processor=source.bridge.processor, system_prompt=system_prompt, source_normalizer=source_norm, target_normalizer=target_norm, target_visual_cache_root=target_cache, k=args.k, support_seed=args.support_seed, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_loader = make_loader(split='valid', api=api, cfg=cfg, processor=source.bridge.processor, system_prompt=system_prompt, source_normalizer=source_norm, target_normalizer=target_norm, target_visual_cache_root=target_cache, k=args.k, support_seed=args.support_seed, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = make_loader(split='test', api=api, cfg=cfg, processor=source.bridge.processor, system_prompt=system_prompt, source_normalizer=source_norm, target_normalizer=target_norm, target_visual_cache_root=target_cache, k=args.k, support_seed=args.support_seed, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    enc_params = [p for p in adapter.target_encoder.parameters() if p.requires_grad]
    head_params = [p for p in adapter.target_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': enc_params, 'lr': args.encoder_lr}, {'params': head_params, 'lr': args.head_lr}], weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum) * args.max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    protocol = {'source_checkpoint': str(source_ckpt), 'target_city': TARGET_CITY, 'K': int(args.k), 'support_seed': int(args.support_seed), 'support_nested': True, 'initialization': 'clone NYC-BIKE private encoder/head (same 16x8 grid)', 'trainable': ['BIKEDC target-private encoder', 'BIKEDC target-private dense head'], 'frozen': ['Stage-1 grounded VLM', 'NYCTAXI/BIKECHI source-private encoders', 'cross-city blocks', 'alignment/DSR', 'shared semantic interface'], 'target_normalization': 'fit only on K BIKEDC support windows', 'source_context': 'deterministic source TRAIN histories only; no source labels', 'selection': 'full BIKEDC validation windows', 'final_test': 'full BIKEDC test windows once after best validation checkpoint', 'max_epochs': int(args.max_epochs), 'eval_every': int(args.eval_every), 'patience': int(args.patience), 'batch_size': int(args.batch_size), 'grad_accum': int(args.grad_accum), 'encoder_lr': float(args.encoder_lr), 'head_lr': float(args.head_lr)}
    (out_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    best = float('inf')
    best_epoch = 0
    bad = 0
    best_path = out_dir / 'best_target.pt'
    history = []
    global_step = 0
    print(json.dumps(protocol, indent=2), flush=True)
    for epoch in range(1, args.max_epochs + 1):
        adapter.train(True)
        optimizer.zero_grad(set_to_none=True)
        train_sum = 0.0
        seen = 0
        t0 = time.time()
        for bi, batch in enumerate(train_loader, start=1):
            batch = _to_device(batch, device)
            pred = adapter(batch)
            loss, parts = target_loss(cfg, pred, batch['cities'][TARGET_CITY]['y'])
            (loss / args.grad_accum).backward()
            train_sum += float(loss.detach())
            seen += 1
            if bi % args.grad_accum == 0 or bi == len(train_loader):
                torch.nn.utils.clip_grad_norm_(adapter.trainable_parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
        train_sec = time.time() - t0
        train_mean = train_sum / max(1, seen)
        row = {'epoch': epoch, 'train_loss': train_mean, 'train_seconds': train_sec, 'global_step': global_step, 'lr_encoder': optimizer.param_groups[0]['lr'], 'lr_head': optimizer.param_groups[1]['lr']}
        do_valid = epoch % args.eval_every == 0 or epoch == args.max_epochs or epoch == 1
        if do_valid:
            valid = evaluate_target(adapter, valid_loader, target_norm, device)
            value = float(valid['selection_mae'])
            row['valid_selection_mae'] = value
            save_target_metrics(valid, out_dir / f'valid_epoch_{epoch:03d}')
            improved = value < best - args.min_delta
            if improved:
                best = value
                best_epoch = epoch
                bad = 0
                save_adapter(best_path, adapter, {'epoch': epoch, 'valid': valid})
                save_target_metrics(valid, out_dir / 'valid_best')
            else:
                bad += 1
            print(f'K={args.k} epoch={epoch} train={train_mean:.6f} valid_mae={value:.6f} best={best:.6f}@{best_epoch} bad={bad}/{args.patience} train_sec={train_sec:.1f}', flush=True)
            print(format_target_metrics(valid), flush=True)
            if bad >= args.patience:
                print(f'EARLY STOP K={args.k}', flush=True)
                history.append(row)
                break
        else:
            print(f'K={args.k} epoch={epoch} train={train_mean:.6f} train_sec={train_sec:.1f}', flush=True)
        history.append(row)
        (out_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    if not best_path.exists():
        raise RuntimeError('No best target checkpoint was saved')
    best_meta = load_adapter(best_path, adapter)
    print(f'Loaded best target adapter K={args.k}: {best_meta}', flush=True)
    test = evaluate_target(adapter, test_loader, target_norm, device)
    test['K'] = int(args.k)
    test['support_seed'] = int(args.support_seed)
    test['best_epoch'] = int(best_epoch)
    test['source_checkpoint'] = str(source_ckpt)
    save_target_metrics(test, out_dir / 'test_best')
    print('FINAL HELD-OUT BIKEDC TEST', flush=True)
    print(format_target_metrics(test), flush=True)
    print('Saved:', out_dir / 'test_best.json', flush=True)

def summarize(root: Path):
    root = Path(root)
    print(f"{'K':>5} {'In MAE':>10} {'In RMSE':>10} {'Out MAE':>10} {'Out RMSE':>10} {'Avg MAE':>10} {'Avg RMSE':>10} {'BestEp':>7}")
    print('-' * 86)
    for k in (50, 100, 200):
        f = root / f'K{k}' / 'test_best.json'
        if not f.exists():
            print(f"{k:>5} {'MISSING':>10}")
            continue
        m = json.loads(f.read_text(encoding='utf-8'))
        print(f"{k:5d} {m['inflow']['mae']:10.4f} {m['inflow']['rmse']:10.4f} {m['outflow']['mae']:10.4f} {m['outflow']['rmse']:10.4f} {m['selection_mae']:10.4f} {m['avg_rmse']:10.4f} {m.get('best_epoch', -1):7d}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-checkpoint', default='./checkpoints/stage2/urbanbind_stage2_v3_2_2_fullgrad_4gpu_acc3/best')
    ap.add_argument('--k', type=int, choices=[50, 100, 200], default=100)
    ap.add_argument('--support-seed', type=int, default=SUPPORT_SEED)
    ap.add_argument('--train-seed', type=int, default=42)
    ap.add_argument('--max-k', type=int, default=200)
    ap.add_argument('--target-visual-cache-root', default='./cache/stage2/heldout_bikedc_visuals')
    ap.add_argument('--output-root', default='./outputs/stage2/heldout_bikedc_oldfull')
    ap.add_argument('--prepare-cache', action='store_true')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--overwrite-cache', action='store_true')
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--max-epochs', type=int, default=10)
    ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--min-delta', type=float, default=0.0001)
    ap.add_argument('--encoder-lr', type=float, default=0.0001)
    ap.add_argument('--head-lr', type=float, default=0.0005)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--max-grad-norm', type=float, default=1.0)
    return ap.parse_args()

def main():
    args = parse_args()
    if args.summarize:
        summarize(Path(args.output_root))
        return
    train_one(args)
if __name__ == '__main__':
    main()
