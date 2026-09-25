from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Iterable
import h5py
import numpy as np
import pandas as pd
from config import RAW_CITY_CONFIG as CITY_CONFIG, ALL_CITY_ORDER, WINDOW_LEN

@dataclass
class CityData:
    name: str
    data: np.ndarray
    time: np.ndarray
    source_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self):
        return int(self.data.shape[0])

    @property
    def H(self):
        return int(self.data.shape[2])

    @property
    def W(self):
        return int(self.data.shape[3])

def _decode_strings(values: Iterable) -> list[str]:
    out = []
    for x in values:
        if isinstance(x, (bytes, bytearray, np.bytes_)):
            out.append(x.decode('utf-8'))
        else:
            out.append(str(x))
    return out

def _parse_grid_time(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, utc=True, errors='raise')
    ts = ts.dt.tz_convert(None)
    return ts.to_numpy(dtype='datetime64[ns]')

def _parse_bikenyc_date(values) -> np.ndarray:
    strings = _decode_strings(values)
    stripped = [s.strip() for s in strings]
    if stripped and all((s.isdigit() and len(s) == 10 for s in stripped)):
        day_part = [s[:8] for s in stripped]
        slot_part = np.asarray([int(s[8:]) for s in stripped], dtype=np.int64)
        base = pd.to_datetime(day_part, format='%Y%m%d', errors='raise')
        if slot_part.min() >= 1 and slot_part.max() <= 24:
            hours = slot_part - 1
            ts = base + pd.to_timedelta(hours, unit='h')
            return ts.to_numpy(dtype='datetime64[ns]')
        if slot_part.min() >= 0 and slot_part.max() <= 23:
            ts = base + pd.to_timedelta(slot_part, unit='h')
            return ts.to_numpy(dtype='datetime64[ns]')
    ts = pd.to_datetime(stripped, utc=True, errors='raise')
    if isinstance(ts, pd.DatetimeIndex) and ts.tz is not None:
        ts = ts.tz_convert(None)
    return np.asarray(ts, dtype='datetime64[ns]')

def load_grid_city(city: str, cfg: dict) -> CityData:
    path = Path(cfg['path'])
    H, W = cfg['grid_shape']
    df = pd.read_csv(path)
    required = {'time', 'row_id', 'column_id', 'inflow', 'outflow'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{city}: missing required columns: {sorted(missing)}')
    parsed_time = _parse_grid_time(df['time'])
    df = df.copy()
    df['_parsed_time'] = parsed_time
    duplicates = int(df.duplicated(subset=['_parsed_time', 'row_id', 'column_id']).sum())
    if duplicates:
        raise ValueError(f'{city}: found {duplicates} duplicate grid cells.')
    observed_rows = sorted(df['row_id'].unique().tolist())
    observed_cols = sorted(df['column_id'].unique().tolist())
    expected_rows = list(range(H))
    expected_cols = list(range(W))
    if observed_rows != expected_rows:
        raise ValueError(f'{city}: row_id mismatch. observed={observed_rows}, expected={expected_rows}')
    if observed_cols != expected_cols:
        raise ValueError(f'{city}: column_id mismatch. observed={observed_cols}, expected={expected_cols}')
    counts = df.groupby('_parsed_time').size()
    expected_cells = H * W
    if not (counts == expected_cells).all():
        bad = counts[counts != expected_cells]
        raise ValueError(f'{city}: incomplete timestamps found. Expected {expected_cells} cells/timestamp; examples={bad.head().to_dict()}')
    df = df.sort_values(['_parsed_time', 'row_id', 'column_id']).reset_index(drop=True)
    unique_times = np.sort(df['_parsed_time'].unique()).astype('datetime64[ns]')
    T = len(unique_times)
    inflow = df['inflow'].to_numpy(dtype=np.float32).reshape(T, H, W)
    outflow = df['outflow'].to_numpy(dtype=np.float32).reshape(T, H, W)
    data = np.stack([inflow, outflow], axis=1).astype(np.float32, copy=False)
    metadata = {'raw_rows': int(len(df)), 'expected_cells_per_timestamp': expected_cells, 'min_cells_per_timestamp': int(counts.min()), 'max_cells_per_timestamp': int(counts.max()), 'duplicate_cells': duplicates}
    return CityData(city, data, unique_times, path, metadata)

def load_h5_city(city: str, cfg: dict) -> CityData:
    path = Path(cfg['path'])
    with h5py.File(path, 'r') as f:
        if 'data' not in f or 'date' not in f:
            raise ValueError(f"{city}: expected H5 datasets 'data' and 'date', found {list(f.keys())}")
        data = np.asarray(f['data'][:], dtype=np.float32)
        raw_date = f['date'][:]
    time = _parse_bikenyc_date(raw_date)
    if len(time) != len(data):
        raise ValueError(f'{city}: len(date)={len(time)} does not match data T={len(data)}')
    metadata = {'h5_keys': ['data', 'date'], 'date_entries': int(len(time))}
    return CityData(city, data, time, path, metadata)

def load_city(city: str) -> CityData:
    cfg = CITY_CONFIG[city]
    path = Path(cfg['path'])
    if not path.exists():
        raise FileNotFoundError(f'{city}: file does not exist: {path}')
    if cfg['format'] == 'grid':
        return load_grid_city(city, cfg)
    if cfg['format'] == 'h5':
        return load_h5_city(city, cfg)
    raise ValueError(f"{city}: unsupported format {cfg['format']}")

def load_all_cities(city_order=None) -> dict[str, CityData]:
    order = ALL_CITY_ORDER if city_order is None else list(city_order)
    return {city: load_city(city) for city in order}

def valid_window_starts(city_data: CityData, interval_minutes: int=60) -> np.ndarray:
    n = city_data.T
    if n < WINDOW_LEN:
        return np.empty((0,), dtype=np.int64)
    mins = city_data.time.astype('datetime64[m]').astype(np.int64)
    diffs = np.diff(mins)
    bad = (diffs != interval_minutes).astype(np.int64)
    prefix = np.concatenate([[0], np.cumsum(bad)])
    starts = []
    gap_count = WINDOW_LEN - 1
    for s in range(n - WINDOW_LEN + 1):
        if prefix[s + gap_count] - prefix[s] == 0:
            starts.append(s)
    return np.asarray(starts, dtype=np.int64)

def slice_window(city_data: CityData, start: int) -> dict:
    end_hist = start + 8
    end_future = end_hist + 4
    return {'start_index': int(start), 'X_hist': city_data.data[start:end_hist], 'Y_future': city_data.data[end_hist:end_future], 'time_hist': city_data.time[start:end_hist], 'time_future': city_data.time[end_hist:end_future]}
