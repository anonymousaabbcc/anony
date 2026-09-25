import numpy as np
from config import EPS, TREND_SLOPE_THRESHOLD, TREND_RANGE_THRESHOLD, TREND_SIGN_CHANGES, CITY_ORDER
from regions import region_center, region_ids, region_vector

def _sign_changes(diffs: np.ndarray) -> int:
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))

def historical_trend(values) -> str:
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or len(v) < 2:
        raise ValueError('historical_trend expects a 1D sequence with >=2 values')
    x = np.arange(1, len(v) + 1, dtype=np.float64)
    mean_abs = float(np.mean(np.abs(v))) + EPS
    slope = float(np.polyfit(x, v, 1)[0])
    slope_norm = slope / mean_abs
    relative_range = float((v.max() - v.min()) / mean_abs)
    sign_changes = _sign_changes(np.diff(v))
    if abs(slope_norm) <= TREND_SLOPE_THRESHOLD and relative_range <= TREND_RANGE_THRESHOLD:
        return 'stable'
    max_sign_changes = max(len(v) - 2, 0)
    required_sign_changes = min(TREND_SIGN_CHANGES, max_sign_changes)
    if max_sign_changes > 0 and sign_changes >= required_sign_changes and (relative_range > TREND_RANGE_THRESHOLD):
        return 'fluctuating'
    return 'increasing' if slope_norm > 0 else 'decreasing'

def normalized_temporal_change(v1: float, v2: float) -> float:
    a, b = (float(v1), float(v2))
    return float(2.0 * abs(b - a) / (abs(a) + abs(b) + EPS))

def normalized_peak_share(values) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1:
        v = v.reshape(-1)
    if len(v) < 2:
        raise ValueError('normalized_peak_share requires at least two spatial units')
    if np.any(v < -EPS):
        raise ValueError('Urban flow concentration expects nonnegative values')
    v = np.maximum(v, 0.0)
    total = float(v.sum())
    if total <= EPS:
        return 0.0
    n = float(len(v))
    pmax = float(v.max() / total)
    c = (n * pmax - 1.0) / (n - 1.0)
    return float(np.clip(c, 0.0, 1.0))

def tied_hotspot_centroid(map_2d: np.ndarray, city: str, level: str) -> tuple[float, float]:
    values = region_vector(map_2d, city, level)
    vmax = float(np.max(values))
    ids = region_ids(level, city=city if level == 'finegrain' else None)
    max_idx = np.flatnonzero(values == vmax)
    centers = np.asarray([region_center(city, level, ids[int(i)]) for i in max_idx], dtype=np.float64)
    return (float(centers[:, 0].mean()), float(centers[:, 1].mean()))

def hotspot_path(history_maps, city: str, level: str) -> np.ndarray:
    maps = np.asarray(history_maps, dtype=np.float64)
    if maps.ndim != 3:
        raise ValueError('hotspot_path expects [T,H,W]')
    totals = maps.reshape(maps.shape[0], -1).sum(axis=1)
    if np.any(totals <= EPS):
        raise ValueError('Hotspot movement requires positive whole-city activity at every historical step')
    return np.asarray([tied_hotspot_centroid(m, city, level) for m in maps], dtype=np.float64)

def hotspot_movement(history_maps, city: str, level: str) -> tuple[str, float, list[list[float]]]:
    path = hotspot_path(history_maps, city, level)
    steps = np.diff(path, axis=0)
    dist = np.sqrt(np.sum(steps ** 2, axis=1))
    score = float(np.mean(dist)) if len(dist) else 0.0
    if score <= EPS:
        label = 'stationary'
    else:
        dr, dc = path[-1] - path[0]
        if abs(dr) <= EPS and abs(dc) <= EPS:
            label = 'mixed'
        elif abs(dr) > abs(dc):
            label = 'down' if dr > 0 else 'up'
        elif abs(dc) > abs(dr):
            label = 'right' if dc > 0 else 'left'
        else:
            label = 'mixed'
    return (label, score, [[float(r), float(c)] for r, c in path])

def prediction_trend(future_values) -> str:
    return historical_trend(future_values)

def city_argmax(score_by_city: dict[str, float], city_order=None) -> str:
    if city_order is None:
        city_order = [c for c in CITY_ORDER if c in score_by_city]
    else:
        city_order = [c for c in city_order if c in score_by_city]
    if not city_order:
        raise ValueError('city_argmax received no valid cities')
    best_city = city_order[0]
    best_score = float(score_by_city[best_city])
    for city in city_order[1:]:
        score = float(score_by_city[city])
        if score > best_score:
            best_city, best_score = (city, score)
    return best_city
