import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import ALL_CITY_ORDER, CITY_SLUG, RAW_CITY_CONFIG, SPLIT_ROOT, TRAIN_RATIO, VALID_RATIO
from raw_loader import load_all_cities

def sha256(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def validate_full_series(city, cd):
    cfg = RAW_CITY_CONFIG[city]
    errors = []
    if tuple(cd.data.shape) != tuple(cfg['expected_shape']):
        errors.append(f"shape {cd.data.shape} != {cfg['expected_shape']}")
    if len(cd.time) != cd.T:
        errors.append('time length != T')
    if not np.isfinite(cd.data).all():
        errors.append('NaN/Inf found')
    if np.any(cd.data < 0):
        errors.append('negative flow values found')
    if len(np.unique(cd.time)) != len(cd.time):
        errors.append('duplicate timestamps found')
    mins = cd.time.astype('datetime64[m]').astype(np.int64)
    if len(mins) > 1 and (not np.all(np.diff(mins) == cfg['interval_minutes'])):
        errors.append('timestamps are not strictly hourly')
    dates = pd.DatetimeIndex(cd.time).normalize()
    counts = pd.Series(1, index=dates).groupby(level=0).sum()
    if not (counts == 24).all():
        errors.append(f'incomplete days: {counts[counts != 24].to_dict()}')
    if errors:
        raise ValueError(f'{city} validation failed: ' + ' | '.join(errors))
    return pd.DatetimeIndex(sorted(counts.index.unique()))

def split_days(unique_days):
    n_days = len(unique_days)
    n_train = n_days * 70 // 100
    n_valid = n_days * 10 // 100
    n_test = n_days - n_train - n_valid
    assert n_train > 0
    assert n_valid > 0
    assert n_test > 0
    return {'train': unique_days[:n_train], 'valid': unique_days[n_train:n_train + n_valid], 'test': unique_days[n_train + n_valid:]}

def select_days(cd, days):
    normalized = pd.DatetimeIndex(cd.time).normalize()
    idx = np.flatnonzero(normalized.isin(days))
    if len(idx) == 0:
        raise ValueError('empty split')
    if not np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)):
        raise ValueError('split is not contiguous')
    return (cd.data[idx].astype(np.float32, copy=False), cd.time[idx].astype('datetime64[ns]', copy=False))

def compute_train_stats(data):
    stats = {}
    for ch, name in enumerate(['inflow', 'outflow']):
        x = data[:, ch].astype(np.float64)
        stats[name] = {'mean': float(x.mean()), 'std': float(x.std()), 'min': float(x.min()), 'max': float(x.max()), 'p01': float(np.quantile(x, 0.01)), 'p99': float(np.quantile(x, 0.99))}
    return stats

def main():
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    cities = load_all_cities()
    global_manifest = {'policy': {'type': 'chronological_full_day', 'ratios': {'train': 0.7, 'valid': 0.1, 'test': 0.2}, 'window_rule': 'Construct 8-history + 4-future windows only after splitting; windows may not cross split boundaries.', 'stats_rule': 'Raw values remain unchanged. Normalization/visual scaling statistics are estimated from train only.'}, 'cities': {}}
    for city in ALL_CITY_ORDER:
        cd = cities[city]
        days = validate_full_series(city, cd)
        parts = split_days(days)
        city_dir = SPLIT_ROOT / CITY_SLUG[city]
        city_dir.mkdir(parents=True, exist_ok=True)
        manifest = {'city': city, 'raw_source': str(cd.source_path), 'raw_shape': list(cd.data.shape), 'total_days': len(days), 'splits': {}}
        saved_times = {}
        train_data = None
        for split in ['train', 'valid', 'test']:
            data, time = select_days(cd, parts[split])
            slug = CITY_SLUG[city]
            data_path = city_dir / f'{slug}_{split}.npy'
            time_path = city_dir / f'{slug}_{split}_time.npy'
            np.save(data_path, data)
            np.save(time_path, time)
            saved_times[split] = set(time.astype('datetime64[ns]').astype(np.int64).tolist())
            if split == 'train':
                train_data = data
            manifest['splits'][split] = {'days': len(parts[split]), 'timestamps': len(time), 'shape': list(data.shape), 'start': str(np.datetime_as_string(time[0], unit='h')).replace('T', ' '), 'end': str(np.datetime_as_string(time[-1], unit='h')).replace('T', ' '), 'data_file': str(data_path.resolve()), 'time_file': str(time_path.resolve()), 'data_sha256': sha256(data_path), 'time_sha256': sha256(time_path)}
        assert not saved_times['train'] & saved_times['valid']
        assert not saved_times['train'] & saved_times['test']
        assert not saved_times['valid'] & saved_times['test']
        stats = compute_train_stats(train_data)
        stats_path = city_dir / f'{CITY_SLUG[city]}_train_stats.json'
        stats_path.write_text(json.dumps(stats, indent=2), encoding='utf-8')
        manifest['train_stats'] = stats
        manifest['train_stats_file'] = str(stats_path.resolve())
        mp = city_dir / f'{CITY_SLUG[city]}_split_manifest.json'
        mp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        global_manifest['cities'][city] = manifest
    gp = SPLIT_ROOT / 'split_manifest_all.json'
    gp.write_text(json.dumps(global_manifest, indent=2), encoding='utf-8')
    print('=== CHRONOLOGICAL FULL-DAY SPLIT ===')
    for city in ALL_CITY_ORDER:
        print(f'\\n{city}')
        for split in ['train', 'valid', 'test']:
            s = global_manifest['cities'][city]['splits'][split]
            print(f"  {split:5s}: days={s['days']:3d}, T={s['timestamps']:4d}, shape={tuple(s['shape'])}, {s['start']} -> {s['end']}")
    print('\\nManifest:', gp)
if __name__ == '__main__':
    main()
