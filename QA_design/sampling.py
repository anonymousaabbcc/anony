from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable
import numpy as np
from config import CHANNEL_TO_INDEX, FINEGRAIN_ZERO_CAP, FINEGRAIN_Q_LOW, FINEGRAIN_Q_HIGH, FINEGRAIN_ACTIVITY_WEIGHT, FINEGRAIN_DYNAMICS_WEIGHT, FINEGRAIN_SAMPLING_POLICY
from regions import finegrain_id
from split_loader import SplitCityData, valid_window_starts

@dataclass(frozen=True)
class CellCandidate:
    start_index: int
    row: int
    col: int
    activity: float
    dynamics: float
    score: float | None
    nonzero_fraction: float
    stratum: str

    @property
    def region_id(self) -> str:
        return finegrain_id(self.row, self.col)

@dataclass
class ChannelReferenceStats:
    city: str
    channel: str
    p0: float
    q0: float
    activity_q33: float
    activity_q67: float
    score_q33: float
    score_q67: float
    sorted_positive_activity: np.ndarray
    sorted_positive_dynamics: np.ndarray
    positive_count: int
    zero_count: int
    total_count: int

    def rank_activity(self, x: float) -> float:
        a = self.sorted_positive_activity
        return float(np.searchsorted(a, float(x), side='right') / max(len(a), 1))

    def rank_dynamics(self, x: float) -> float:
        d = self.sorted_positive_dynamics
        return float(np.searchsorted(d, float(x), side='right') / max(len(d), 1))

    def score(self, activity: float, dynamics: float) -> float:
        return float(FINEGRAIN_ACTIVITY_WEIGHT * self.rank_activity(activity) + FINEGRAIN_DYNAMICS_WEIGHT * self.rank_dynamics(dynamics))

    def classify(self, is_zero: bool, activity: float, dynamics: float, mode: str) -> tuple[str, float | None]:
        if is_zero:
            return ('Z', None)
        if mode == 'activity':
            if activity <= self.activity_q33:
                return ('L', None)
            if activity <= self.activity_q67:
                return ('M', None)
            return ('H', None)
        if mode == 'activity_dynamics':
            s = self.score(activity, dynamics)
            if s <= self.score_q33:
                return ('L', s)
            if s <= self.score_q67:
                return ('M', s)
            return ('H', s)
        raise ValueError(f'Unknown finegrain sampling mode: {mode}')

    def sampling_probabilities(self) -> dict[str, float]:
        rest = (1.0 - self.q0) / 3.0
        return {'Z': self.q0, 'L': rest, 'M': rest, 'H': rest}

    def to_json(self) -> dict:
        return {'city': self.city, 'channel': self.channel, 'p0': self.p0, 'q0': self.q0, 'activity_q33': self.activity_q33, 'activity_q67': self.activity_q67, 'score_q33': self.score_q33, 'score_q67': self.score_q67, 'positive_count': self.positive_count, 'zero_count': self.zero_count, 'total_count': self.total_count, 'sampling_probabilities': self.sampling_probabilities()}

class CandidatePool:

    def __init__(self, candidates: Iterable[CellCandidate]):
        self.by_stratum: Dict[str, list[CellCandidate]] = {s: [] for s in ['Z', 'L', 'M', 'H']}
        self.by_stratum_start: Dict[str, dict[int, list[CellCandidate]]] = {s: {} for s in ['Z', 'L', 'M', 'H']}
        for c in candidates:
            self.by_stratum[c.stratum].append(c)
            self.by_stratum_start[c.stratum].setdefault(c.start_index, []).append(c)

    def count(self, stratum: str) -> int:
        return len(self.by_stratum[stratum])

class FinegrainSampler:

    def __init__(self, active_cities: dict[str, SplitCityData], train_reference_cities: dict[str, SplitCityData], seed: int=42):
        self.active_cities = active_cities
        self.train_reference_cities = train_reference_cities
        self.rng = np.random.default_rng(seed)
        self.reference: dict[tuple[str, str], ChannelReferenceStats] = {}
        self._pools: dict[tuple[str, str, str], CandidatePool] = {}
        for city in active_cities:
            if city not in train_reference_cities:
                raise ValueError(f'Missing train sampling reference for {city}')
            for channel in CHANNEL_TO_INDEX:
                self.reference[city, channel] = self._build_reference(city, channel)

    @staticmethod
    def _window_metrics(x_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x_hist, dtype=np.float64)
        if x.shape[0] != 8:
            raise ValueError(f'Expected 8-step history, got {x.shape}')
        if np.min(x) < -1e-09:
            raise ValueError('Finegrain zero/positive strata assume nonnegative urban flow')
        x = np.maximum(x, 0.0)
        logx = np.log1p(x)
        activity = np.mean(logx, axis=0)
        dynamics = np.mean(np.abs(np.diff(logx, axis=0)), axis=0)
        zero = np.all(x == 0.0, axis=0)
        nonzero_fraction = np.mean(x > 0.0, axis=0)
        return (zero, activity, dynamics, nonzero_fraction)

    def _build_reference(self, city: str, channel: str) -> ChannelReferenceStats:
        cd = self.train_reference_cities[city]
        ch = CHANNEL_TO_INDEX[channel]
        starts = valid_window_starts(cd)
        pos_a, pos_d = ([], [])
        zero_count = 0
        total_count = 0
        for s in starts:
            zero, a, d, _ = self._window_metrics(cd.data[int(s):int(s) + 8, ch])
            total_count += int(zero.size)
            zero_count += int(zero.sum())
            positive = ~zero
            if np.any(positive):
                pos_a.append(a[positive].reshape(-1))
                pos_d.append(d[positive].reshape(-1))
        if not pos_a:
            raise ValueError(f'{city}/{channel}: train split has no positive finegrain histories')
        a_all = np.concatenate(pos_a).astype(np.float64, copy=False)
        d_all = np.concatenate(pos_d).astype(np.float64, copy=False)
        a_sorted = np.sort(a_all)
        d_sorted = np.sort(d_all)
        a33, a67 = np.quantile(a_all, [FINEGRAIN_Q_LOW, FINEGRAIN_Q_HIGH])
        rank_a = np.searchsorted(a_sorted, a_all, side='right') / len(a_sorted)
        rank_d = np.searchsorted(d_sorted, d_all, side='right') / len(d_sorted)
        score_all = FINEGRAIN_ACTIVITY_WEIGHT * rank_a + FINEGRAIN_DYNAMICS_WEIGHT * rank_d
        s33, s67 = np.quantile(score_all, [FINEGRAIN_Q_LOW, FINEGRAIN_Q_HIGH])
        p0 = float(zero_count / max(total_count, 1))
        q0 = float(min(p0, FINEGRAIN_ZERO_CAP))
        return ChannelReferenceStats(city=city, channel=channel, p0=p0, q0=q0, activity_q33=float(a33), activity_q67=float(a67), score_q33=float(s33), score_q67=float(s67), sorted_positive_activity=a_sorted, sorted_positive_dynamics=d_sorted, positive_count=int(len(a_all)), zero_count=int(zero_count), total_count=int(total_count))

    def _build_pool(self, city: str, channel: str, mode: str) -> CandidatePool:
        key = (city, channel, mode)
        if key in self._pools:
            return self._pools[key]
        cd = self.active_cities[city]
        ch = CHANNEL_TO_INDEX[channel]
        stats = self.reference[city, channel]
        candidates: list[CellCandidate] = []
        for s in valid_window_starts(cd):
            s = int(s)
            zero, a, d, nz = self._window_metrics(cd.data[s:s + 8, ch])
            H, W = a.shape
            for r in range(H):
                for c in range(W):
                    stratum, score = stats.classify(bool(zero[r, c]), float(a[r, c]), float(d[r, c]), mode)
                    candidates.append(CellCandidate(start_index=s, row=r, col=c, activity=float(a[r, c]), dynamics=float(d[r, c]), score=None if score is None else float(score), nonzero_fraction=float(nz[r, c]), stratum=stratum))
        pool = CandidatePool(candidates)
        self._pools[key] = pool
        return pool

    def _choose_stratum(self, city: str, channel: str, mode: str) -> str:
        stats = self.reference[city, channel]
        pool = self._build_pool(city, channel, mode)
        probs = stats.sampling_probabilities()
        labels, weights = ([], [])
        for s in ['Z', 'L', 'M', 'H']:
            if probs[s] <= 0:
                continue
            if pool.count(s) == 0:
                raise ValueError(f'{city}/{channel}/{mode}: required stratum {s} has zero active-split candidates; refuse to silently renormalize the frozen sampling policy')
            labels.append(s)
            weights.append(probs[s])
        if not labels:
            raise ValueError(f'{city}/{channel}/{mode}: no sampling strata with positive probability')
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        return str(self.rng.choice(labels, p=weights))

    def sample_cell(self, city: str, channel: str, mode: str) -> CellCandidate:
        stratum = self._choose_stratum(city, channel, mode)
        pool = self._build_pool(city, channel, mode).by_stratum[stratum]
        return pool[int(self.rng.integers(0, len(pool)))]

    def sample_pair(self, city: str, channel: str, mode: str) -> tuple[CellCandidate, CellCandidate]:
        stratum_a = self._choose_stratum(city, channel, mode)
        stratum_b = self._choose_stratum(city, channel, mode)
        pool = self._build_pool(city, channel, mode)
        starts_a = pool.by_stratum_start[stratum_a]
        starts_b = pool.by_stratum_start[stratum_b]
        common = []
        for start in starts_a.keys() & starts_b.keys():
            if stratum_a != stratum_b or len(starts_a[start]) >= 2:
                common.append(start)
        if not common:
            raise ValueError(f'{city}/{channel}/{mode}: no historical window can realize pair strata ({stratum_a},{stratum_b}) without replacement')
        start = int(self.rng.choice(np.asarray(common, dtype=np.int64)))
        xa = starts_a[start]
        xb = starts_b[start]
        if stratum_a == stratum_b:
            idx = self.rng.choice(len(xa), size=2, replace=False)
            return (xa[int(idx[0])], xa[int(idx[1])])
        a = xa[int(self.rng.integers(0, len(xa)))]
        b = xb[int(self.rng.integers(0, len(xb)))]
        return (a, b)

    def metadata(self, city: str, channel: str, mode: str, candidate: CellCandidate) -> dict:
        stats = self.reference[city, channel]
        return {'policy': FINEGRAIN_SAMPLING_POLICY, 'basis': 'history_only_t1_t8', 'mode': mode, 'selection_channel': channel, 'region_id': candidate.region_id, 'start_index': candidate.start_index, 'stratum': candidate.stratum, 'activity': candidate.activity, 'dynamics': candidate.dynamics, 'score': candidate.score, 'nonzero_fraction': candidate.nonzero_fraction, 'train_p0': stats.p0, 'zero_sampling_probability_q0': stats.q0}

    def reference_summary(self) -> dict:
        return {f'{city}/{channel}': stats.to_json() for (city, channel), stats in sorted(self.reference.items())}
