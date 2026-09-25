from dataclasses import dataclass
import numpy as np
from config import SOURCE_CITY_ORDER, TARGET_CITY_ORDER, CITY_SLUG, RAW_CITY_CONFIG, SPLIT_ROOT, WINDOW_LEN

@dataclass
class SplitCityData:
    name: str
    split: str
    data: np.ndarray
    time: np.ndarray

    @property
    def T(self):
        return int(self.data.shape[0])

    @property
    def H(self):
        return int(self.data.shape[2])

    @property
    def W(self):
        return int(self.data.shape[3])

def load_split(city, split):
    slug = CITY_SLUG[city]
    root = SPLIT_ROOT / slug
    dp = root / f'{slug}_{split}.npy'
    tp = root / f'{slug}_{split}_time.npy'
    if not dp.exists() or not tp.exists():
        raise FileNotFoundError(f'Missing {city}/{split}. Run split_data.py first.')
    data = np.load(dp)
    time = np.load(tp, allow_pickle=False)
    hw = RAW_CITY_CONFIG[city]['grid_shape']
    if data.ndim != 4 or data.shape[1] != 2 or tuple(data.shape[2:]) != tuple(hw):
        raise ValueError(f'{city}/{split}: bad shape {data.shape}')
    if len(time) != len(data):
        raise ValueError(f'{city}/{split}: time length mismatch')
    return SplitCityData(city, split, data, time)

def load_all_split(split, city_order=None):
    order = SOURCE_CITY_ORDER if city_order is None else list(city_order)
    return {c: load_split(c, split) for c in order}

def load_target_split(split):
    return load_all_split(split, city_order=TARGET_CITY_ORDER)

def valid_window_starts(cd, interval_minutes=60):
    if cd.T < WINDOW_LEN:
        return np.empty((0,), dtype=np.int64)
    mins = cd.time.astype('datetime64[m]').astype(np.int64)
    bad = (np.diff(mins) != interval_minutes).astype(np.int64)
    prefix = np.concatenate([[0], np.cumsum(bad)])
    starts = []
    for s in range(cd.T - WINDOW_LEN + 1):
        if prefix[s + WINDOW_LEN - 1] - prefix[s] == 0:
            starts.append(s)
    return np.asarray(starts, dtype=np.int64)

def slice_window(cd, start):
    return {'start_index': int(start), 'X_hist': cd.data[start:start + 8], 'Y_future': cd.data[start + 8:start + 12], 'time_hist': cd.time[start:start + 8], 'time_future': cd.time[start + 8:start + 12]}
