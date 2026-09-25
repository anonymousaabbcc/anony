from typing import Dict, Tuple
import re
import numpy as np
from config import CITY_CONFIG
_FINE_RE = re.compile('^F\\((\\d+),(\\d+)\\)$')

def split_ranges(n: int, k: int) -> list[tuple[int, int]]:
    q, r = divmod(n, k)
    out = []
    start = 0
    for i in range(k):
        size = q + (1 if i < r else 0)
        out.append((start, start + size))
        start += size
    return out

def finegrain_id(row0: int, col0: int) -> str:
    return f'F({int(row0) + 1},{int(col0) + 1})'

def parse_finegrain_id(region_id: str) -> tuple[int, int]:
    m = _FINE_RE.match(str(region_id))
    if not m:
        raise ValueError(f'Invalid finegrain region ID: {region_id}')
    return (int(m.group(1)) - 1, int(m.group(2)) - 1)

def region_ids(level: str, city: str | None=None) -> list[str]:
    if level == 'whole':
        return ['whole_city']
    if level == 'coarse':
        return [f'coarse_{i}' for i in range(1, 5)]
    if level == 'local':
        return [f'local_{i}' for i in range(1, 17)]
    if level == 'finegrain':
        if city is None:
            raise ValueError("region_ids('finegrain') requires city because native grids differ")
        H, W = CITY_CONFIG[city]['grid_shape']
        return [finegrain_id(r, c) for r in range(H) for c in range(W)]
    raise ValueError(f'Unknown region level: {level}')

def get_region_slices(city: str, level: str) -> Dict[str, Tuple[slice, slice]]:
    H, W = CITY_CONFIG[city]['grid_shape']
    if level == 'whole':
        return {'whole_city': (slice(0, H), slice(0, W))}
    if level == 'finegrain':
        return {finegrain_id(r, c): (slice(r, r + 1), slice(c, c + 1)) for r in range(H) for c in range(W)}
    if level == 'coarse':
        k, prefix = (2, 'coarse')
    elif level == 'local':
        k, prefix = (4, 'local')
    else:
        raise ValueError(f'Unknown region level: {level}')
    row_ranges = split_ranges(H, k)
    col_ranges = split_ranges(W, k)
    out = {}
    idx = 1
    for rs, re in row_ranges:
        for cs, ce in col_ranges:
            out[f'{prefix}_{idx}'] = (slice(rs, re), slice(cs, ce))
            idx += 1
    return out

def region_value(map_2d: np.ndarray, city: str, level: str, region_id: str) -> float:
    if level == 'finegrain':
        r, c = parse_finegrain_id(region_id)
        H, W = CITY_CONFIG[city]['grid_shape']
        if not (0 <= r < H and 0 <= c < W):
            raise ValueError(f'{city}: {region_id} outside native grid {H}x{W}')
        return float(map_2d[r, c])
    rs, cs = get_region_slices(city, level)[region_id]
    return float(np.mean(map_2d[rs, cs], dtype=np.float64))

def region_vector(map_2d: np.ndarray, city: str, level: str) -> np.ndarray:
    if level == 'finegrain':
        return np.asarray(map_2d, dtype=np.float64).reshape(-1)
    slices = get_region_slices(city, level)
    return np.asarray([region_value(map_2d, city, level, rid) for rid in slices.keys()], dtype=np.float64)

def dominant_region(map_2d: np.ndarray, city: str, level: str) -> str:
    values = region_vector(map_2d, city, level)
    ids = region_ids(level, city=city if level == 'finegrain' else None)
    return ids[int(np.argmax(values))]

def spatial_unit_count(city: str, level: str) -> int:
    if level == 'whole':
        return 1
    if level == 'coarse':
        return 4
    if level == 'local':
        return 16
    if level == 'finegrain':
        H, W = CITY_CONFIG[city]['grid_shape']
        return int(H * W)
    raise ValueError(level)

def region_center(city: str, level: str, region_id: str) -> tuple[float, float]:
    H, W = CITY_CONFIG[city]['grid_shape']
    if level == 'finegrain':
        r, c = parse_finegrain_id(region_id)
        return (float((r + 0.5) / H), float((c + 0.5) / W))
    rs, cs = get_region_slices(city, level)[region_id]
    r = ((rs.start + rs.stop - 1) / 2.0 + 0.5) / H
    c = ((cs.start + cs.stop - 1) / 2.0 + 0.5) / W
    return (float(r), float(c))

def validate_region_partitions(city: str, level: str) -> dict:
    H, W = CITY_CONFIG[city]['grid_shape']
    slices = get_region_slices(city, level)
    mask = np.zeros((H, W), dtype=np.int32)
    for _, (rs, cs) in slices.items():
        mask[rs, cs] += 1
    return {'city': city, 'level': level, 'regions': len(slices), 'all_cells_covered': bool(np.all(mask >= 1)), 'no_overlap': bool(np.all(mask <= 1)), 'covered_cells': int(np.sum(mask == 1)), 'total_cells': int(H * W), 'pass': bool(np.all(mask == 1))}
