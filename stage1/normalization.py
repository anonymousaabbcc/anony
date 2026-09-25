import json
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from .config import CITY_ORDER, CHANNELS, MAX_VALUE_DIM
EXPECTED_TRAIN_TIMESTAMPS = {'NYCTAXI': 1512, 'BIKECHI': 1536, 'NYC-BIKE': 3072}

def _npz_city_key(city: str) -> str:
    return city.replace('-', '_')

class RunningMoments:

    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def update(self, values: np.ndarray):
        x = np.asarray(values, dtype=np.float64)
        self.count += int(x.size)
        self.sum += float(x.sum())
        self.sumsq += float(np.square(x).sum())

    def finalize(self) -> Tuple[float, float, int]:
        if self.count <= 0:
            raise ValueError('No values were accumulated.')
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 1e-12)
        std = float(np.sqrt(var))
        return (float(mean), std, self.count)

def _load_train_split_api():
    qa_design_dir = Path(__file__).resolve().parents[1] / 'QA_design'
    if not qa_design_dir.exists():
        raise FileNotFoundError(f'Cannot find QA_design directory at {qa_design_dir}')
    qa_design_str = str(qa_design_dir)
    if qa_design_str not in sys.path:
        sys.path.insert(0, qa_design_str)
    from split_loader import load_all_split, slice_window, valid_window_starts
    return (load_all_split, slice_window, valid_window_starts)

def build_stats_from_full_train_split(output_path: str) -> dict:
    load_all_split, slice_window, valid_window_starts = _load_train_split_api()
    train_cities = load_all_split('train')
    moments = {city: {channel: RunningMoments() for channel in CHANNELS} for city in CITY_ORDER}
    timestamp_counts = {}
    for city in CITY_ORDER:
        if city not in train_cities:
            raise KeyError(f"{city} is missing from load_all_split('train'). Available cities: {list(train_cities.keys())}")
        city_data = train_cities[city]
        starts = valid_window_starts(city_data)
        if len(starts) == 0:
            raise ValueError(f'{city}: no valid train windows.')
        seen_timestamps = set()
        for start in starts:
            window = slice_window(city_data, int(start))
            values = np.concatenate([np.asarray(window['X_hist']), np.asarray(window['Y_future'])], axis=0)
            times = np.concatenate([np.asarray(window['time_hist']), np.asarray(window['time_future'])], axis=0)
            if values.ndim != 4 or values.shape[0] != 12 or values.shape[1] != 2:
                raise ValueError(f'{city}: expected reconstructed window [12,2,H,W], got {values.shape}')
            if len(times) != 12:
                raise ValueError(f'{city}: expected 12 timestamps per window, got {len(times)}')
            for timestamp, frame in zip(times, values):
                timestamp_key = str(np.datetime_as_string(np.datetime64(timestamp), unit='h'))
                if timestamp_key in seen_timestamps:
                    continue
                seen_timestamps.add(timestamp_key)
                moments[city]['inflow'].update(frame[0])
                moments[city]['outflow'].update(frame[1])
        timestamp_counts[city] = len(seen_timestamps)
        expected = EXPECTED_TRAIN_TIMESTAMPS[city]
        if timestamp_counts[city] != expected:
            raise RuntimeError(f'{city}: recovered {timestamp_counts[city]} unique train timestamps, expected {expected}. Do not use these normalization statistics until this is resolved.')
    stats = {'source': 'complete chronological train split', 'method': "load_all_split('train') + overlapping 12-hour windows + timestamp deduplication", 'timestamp_counts': timestamp_counts, 'cities': {}}
    for city in CITY_ORDER:
        stats['cities'][city] = {}
        for channel in CHANNELS:
            mean, std, count = moments[city][channel].finalize()
            stats['cities'][city][channel] = {'mean': mean, 'std': std, 'count': count}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    return stats

def load_raw_tensor_targets(record: dict) -> Dict[str, np.ndarray]:
    ref = record.get('tensor_target_ref')
    if not ref:
        return {}
    with np.load(ref) as z:
        if record['template_id'] == 'P4':
            city = record['cities'][0]
            return {city: np.asarray(z['target'], dtype=np.float32)}
        if record['template_id'] == 'P8':
            out = {}
            for city in CITY_ORDER:
                out[city] = np.asarray(z[_npz_city_key(city)], dtype=np.float32)
            return out
    raise ValueError(f"Unsupported tensor template: {record['template_id']}")

class FlowNormalizer:

    def __init__(self, stats_path: str):
        self.stats_path = str(stats_path)
        self.stats = json.loads(Path(stats_path).read_text(encoding='utf-8'))

    def mean_std(self, city: str, channel: str) -> Tuple[float, float]:
        item = self.stats['cities'][city][channel]
        return (float(item['mean']), float(item['std']))

    def normalize_direct(self, city: str, channel: str, values):
        mean, std = self.mean_std(city, channel)
        return (np.asarray(values, dtype=np.float32) - mean) / std

    def normalize_difference(self, city: str, channel: str, values):
        _, std = self.mean_std(city, channel)
        return np.asarray(values, dtype=np.float32) / std

    def inverse_direct(self, city: str, channel: str, values):
        mean, std = self.mean_std(city, channel)
        return np.asarray(values, dtype=np.float32) * std + mean

    def normalize_tensor(self, city: str, tensor: np.ndarray) -> np.ndarray:
        x = np.asarray(tensor, dtype=np.float32).copy()
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f'{city}: expected [4,2,H,W], got {x.shape}')
        x[:, 0] = self.normalize_direct(city, 'inflow', x[:, 0])
        x[:, 1] = self.normalize_direct(city, 'outflow', x[:, 1])
        return x

    def inverse_tensor(self, city: str, tensor) -> np.ndarray:
        x = np.asarray(tensor, dtype=np.float32).copy()
        x[:, 0] = self.inverse_direct(city, 'inflow', x[:, 0])
        x[:, 1] = self.inverse_direct(city, 'outflow', x[:, 1])
        return x

def _flatten(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)

def encode_value_target(record: dict, normalizer: FlowNormalizer):
    target = np.zeros(MAX_VALUE_DIM, dtype=np.float32)
    mask = np.zeros(MAX_VALUE_DIM, dtype=np.bool_)
    if not int(record.get('loss_mask', {}).get('value_mse', 0)):
        return (torch.from_numpy(target), torch.from_numpy(mask))
    tid = record['template_id']
    numeric = record['numeric_target']
    values = []
    direct_single = {'U1', 'U7', 'U13'}
    diff_single = {'U2', 'U8', 'U14'}
    direct_city = {'U4', 'U10', 'U16'}
    diff_city = {'U5', 'U17'}
    if tid in direct_single:
        city = record['cities'][0]
        modality = record['modality']
        values.extend(_flatten(normalizer.normalize_direct(city, modality, numeric)))
    elif tid in diff_single:
        city = record['cities'][0]
        modality = record['modality']
        values.extend(_flatten(normalizer.normalize_difference(city, modality, numeric)))
    elif tid in direct_city:
        modality = record['modality']
        for city in CITY_ORDER:
            values.extend(_flatten(normalizer.normalize_direct(city, modality, numeric[city])))
    elif tid in diff_city:
        modality = record['modality']
        for city in CITY_ORDER:
            values.extend(_flatten(normalizer.normalize_difference(city, modality, numeric[city])))
    elif tid == 'U11':
        for city in CITY_ORDER:
            values.extend(_flatten(numeric[city]))
    elif tid == 'P3':
        city = record['cities'][0]
        for channel in CHANNELS:
            values.extend(_flatten(normalizer.normalize_direct(city, channel, numeric[channel])))
    elif tid == 'P7':
        for city in CITY_ORDER:
            for channel in CHANNELS:
                values.extend(_flatten(normalizer.normalize_direct(city, channel, numeric[city][channel])))
    else:
        raise ValueError(f"{record['sample_id']}: value_mse=1 but no value encoder for {tid}")
    values = np.asarray(values, dtype=np.float32)
    if values.size > MAX_VALUE_DIM:
        raise ValueError(f"{record['sample_id']}: numerical target has {values.size} values, exceeding MAX_VALUE_DIM={MAX_VALUE_DIM}")
    target[:values.size] = values
    mask[:values.size] = True
    return (torch.from_numpy(target), torch.from_numpy(mask))

def load_normalized_tensor_targets(record: dict, normalizer: FlowNormalizer):
    if not int(record.get('loss_mask', {}).get('tensor_mse', 0)):
        return {}
    raw = load_raw_tensor_targets(record)
    return {city: torch.from_numpy(normalizer.normalize_tensor(city, arr)) for city, arr in raw.items()}

def decode_prediction_value_pair(record: dict, pred_norm, normalizer: FlowNormalizer):
    tid = record['template_id']
    pred = np.asarray(pred_norm, dtype=np.float32).reshape(-1)
    out = {}
    pos = 0
    if tid == 'P3':
        city = record['cities'][0]
        pred_parts = []
        target_parts = []
        for channel in CHANNELS:
            n = len(record['numeric_target'][channel])
            p = normalizer.inverse_direct(city, channel, pred[pos:pos + n])
            y = _flatten(record['numeric_target'][channel])
            pred_parts.append(_flatten(p))
            target_parts.append(y)
            pos += n
        out[city] = {'pred': np.concatenate(pred_parts), 'target': np.concatenate(target_parts)}
        return out
    if tid == 'P7':
        for city in CITY_ORDER:
            pred_parts = []
            target_parts = []
            for channel in CHANNELS:
                n = len(record['numeric_target'][city][channel])
                p = normalizer.inverse_direct(city, channel, pred[pos:pos + n])
                y = _flatten(record['numeric_target'][city][channel])
                pred_parts.append(_flatten(p))
                target_parts.append(y)
                pos += n
            out[city] = {'pred': np.concatenate(pred_parts), 'target': np.concatenate(target_parts)}
        return out
    raise ValueError(f'decode_prediction_value_pair only supports P3/P7, got {tid}')
