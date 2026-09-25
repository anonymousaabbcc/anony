from dataclasses import dataclass
from typing import Any
import hashlib
import numpy as np
from config import SOURCE_CITY_ORDER, CHANNEL_TO_INDEX, HORIZONS, TEXT_DECIMALS, RATIO_DECIMALS
from split_loader import SplitCityData as CityData, slice_window, valid_window_starts
from regions import region_ids, region_value, region_vector, dominant_region
from targets import historical_trend, normalized_temporal_change, normalized_peak_share, hotspot_movement, prediction_trend, city_argmax
from templates import TEMPLATES, QUESTION_VARIANTS, loss_mask_for_target_type
from sampling import FinegrainSampler

@dataclass
class GeneratedQASample:
    record: dict
    windows: dict
    tensor_target: Any = None

def _round_scalar(x):
    return round(float(x), TEXT_DECIMALS)

def _round_vector(x):
    return [_round_scalar(v) for v in np.asarray(x).tolist()]

def _fmt_scalar(x):
    return f'{float(x):.{TEXT_DECIMALS}f}'

def _fmt_vector(x):
    return '[' + ', '.join((_fmt_scalar(v) for v in np.asarray(x).tolist())) + ']'

def _ts_string(x):
    return str(np.datetime_as_string(np.datetime64(x), unit='h')).replace('T', ' ')

def _round_ratio(x):
    return round(float(x), RATIO_DECIMALS)

def _fmt_ratio(x):
    return f'{float(x):.{RATIO_DECIMALS}f}'

def _join_city_parts(city_order, part_fn):
    return '; '.join((part_fn(city) for city in city_order)) + '.'

def _format_p8_shapes(city_order, tensor_target):
    return _join_city_parts(city_order, lambda city: f"{city}: [{','.join((str(int(x)) for x in tensor_target[city].shape))}]")
FINE_ACTIVITY_SINGLE = {'U1', 'U4', 'U7', 'U10'}
FINE_DYNAMICS_SINGLE = {'U2', 'U3', 'U5', 'U6', 'P1', 'P3', 'P5', 'P7'}
FINE_DYNAMICS_PAIR = {'U8', 'U13', 'U14', 'U16', 'U17'}
MOVEMENT_TEMPLATES = {'U15', 'U18'}

class QAGenerator:

    def __init__(self, cities: dict[str, CityData], seed: int=42, city_order=None, finegrain_sampler: FinegrainSampler | None=None):
        self.cities = cities
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.city_order = tuple(SOURCE_CITY_ORDER if city_order is None else city_order)
        self.finegrain_sampler = finegrain_sampler
        missing = [city for city in self.city_order if city not in cities]
        if missing:
            raise ValueError(f'Generator city_order contains cities not loaded: {missing}')
        self.valid_starts = {city: valid_window_starts(cities[city]) for city in self.city_order}
        for city, starts in self.valid_starts.items():
            if len(starts) == 0:
                raise ValueError(f'{city}: no valid 12-hour windows.')
        self._active_start_cache: dict[tuple[str, str], np.ndarray] = {}

    def _sample_start(self, city):
        return int(self.rng.choice(self.valid_starts[city]))

    def _sample_windows(self, city_scope, selected_city=None):
        if city_scope == 'single_city':
            city = selected_city or str(self.rng.choice(self.city_order))
            return {city: slice_window(self.cities[city], self._sample_start(city))}
        return {city: slice_window(self.cities[city], self._sample_start(city)) for city in self.city_order}

    def _positive_history_starts(self, city: str, channel: str) -> np.ndarray:
        key = (city, channel)
        if key in self._active_start_cache:
            return self._active_start_cache[key]
        ch = CHANNEL_TO_INDEX[channel]
        cd = self.cities[city]
        keep = []
        for s in self.valid_starts[city]:
            s = int(s)
            hist = np.asarray(cd.data[s:s + 8, ch], dtype=np.float64)
            totals = hist.reshape(8, -1).sum(axis=1)
            if np.all(totals > 0):
                keep.append(s)
        out = np.asarray(keep, dtype=np.int64)
        if len(out) == 0:
            raise ValueError(f'{city}/{channel}: no history window has positive whole-city activity at all t=1..8')
        self._active_start_cache[key] = out
        return out

    def _sample_movement_windows(self, city_scope, channel: str, selected_city=None):
        if city_scope == 'single_city':
            city = selected_city or str(self.rng.choice(self.city_order))
            s = int(self.rng.choice(self._positive_history_starts(city, channel)))
            return {city: slice_window(self.cities[city], s)}
        out = {}
        for city in self.city_order:
            s = int(self.rng.choice(self._positive_history_starts(city, channel)))
            out[city] = slice_window(self.cities[city], s)
        return out

    def _hist_series(self, window, city, channel, level, rid):
        ch = CHANNEL_TO_INDEX[channel]
        return np.asarray([region_value(window['X_hist'][t, ch], city, level, rid) for t in range(8)], dtype=np.float64)

    def _future_series(self, window, city, channel, level, rid, horizons=HORIZONS):
        ch = CHANNEL_TO_INDEX[channel]
        return np.asarray([region_value(window['Y_future'][h - 1, ch], city, level, rid) for h in horizons], dtype=np.float64)

    def _region_vec(self, map_2d, city, level):
        return region_vector(map_2d, city, level)

    def _question_variant_index(self, template_id: str, sample_index: int) -> int:
        variants = QUESTION_VARIANTS[template_id]
        n = len(variants)
        group_id = sample_index // n
        position = sample_index % n
        key = f'{self.seed}|{template_id}|{group_id}'.encode('utf-8')
        digest = hashlib.blake2b(key, digest_size=8).digest()
        local_seed = int.from_bytes(digest, byteorder='big', signed=False)
        order = np.random.default_rng(local_seed).permutation(n)
        return int(order[position])

    def _apply_question_variant(self, record: dict, sample_index: int) -> None:
        tid = record['template_id']
        variant_id = self._question_variant_index(tid, sample_index)
        city = record['cities'][0] if record['cities'] else None
        context = {'city': city, 'level': record['region_level'], 'rid': record.get('region_id'), 'modality': record.get('modality'), 'time_step': record.get('time_step'), 't1': record.get('t1'), 't2': record.get('t2'), 'horizon': record.get('horizon'), 'horizon_set': record.get('horizon_set'), 'rid_a': record.get('region_id_a'), 'rid_b': record.get('region_id_b'), 'unit_phrase': record.get('unit_phrase'), 'pair_phrase': record.get('pair_phrase')}
        record['question_variant'] = variant_id
        record['question'] = QUESTION_VARIANTS[tid][variant_id].format(**context)

    def _sample_finegrain_cells(self, tid: str, cities: list[str], channel: str, pair: bool):
        if self.finegrain_sampler is None:
            raise ValueError('Finegrain selected-cell QA requires FinegrainSampler')
        mode = 'activity' if tid in FINE_ACTIVITY_SINGLE else 'activity_dynamics'
        windows, ids_by_city, meta_by_city = ({}, {}, {})
        ids_a, ids_b = ({}, {})
        for city in cities:
            if pair:
                a, b = self.finegrain_sampler.sample_pair(city, channel, mode)
                if a.start_index != b.start_index:
                    raise AssertionError('pair sampler returned different windows')
                windows[city] = slice_window(self.cities[city], a.start_index)
                ids_a[city], ids_b[city] = (a.region_id, b.region_id)
                meta_by_city[city] = {'policy': 'historical_only_zero_preserving_stratified', 'basis': 'history_only_t1_t8', 'mode': mode, 'selection_channel': channel, 'cells': [self.finegrain_sampler.metadata(city, channel, mode, a), self.finegrain_sampler.metadata(city, channel, mode, b)]}
            else:
                c = self.finegrain_sampler.sample_cell(city, channel, mode)
                windows[city] = slice_window(self.cities[city], c.start_index)
                ids_by_city[city] = c.region_id
                meta_by_city[city] = self.finegrain_sampler.metadata(city, channel, mode, c)
        if pair:
            return (windows, ids_a, ids_b, meta_by_city)
        return (windows, ids_by_city, meta_by_city)

    @staticmethod
    def _unit_phrase(level: str, cities: list[str], shared_rid=None, ids_by_city=None) -> str:
        if level != 'finegrain':
            return str(shared_rid)
        if len(cities) == 1:
            return str(ids_by_city[cities[0]])
        details = ', '.join((f'{city} {ids_by_city[city]}' for city in cities))
        return f'the specified fine-grained cell in each city ({details})'

    @staticmethod
    def _pair_phrase(level: str, cities: list[str], t1: int, t2: int, rid_a=None, rid_b=None, ids_a=None, ids_b=None) -> str:
        if level != 'finegrain':
            return f'({rid_a}, t={t1}) and ({rid_b}, t={t2})'
        if len(cities) == 1:
            city = cities[0]
            return f'({ids_a[city]}, t={t1}) and ({ids_b[city]}, t={t2})'
        return '; '.join((f'{city}: ({ids_a[city]}, t={t1}) and ({ids_b[city]}, t={t2})' for city in cities))

    def generate(self, template_id: str, sample_index: int, forced_city=None, forced_level=None) -> GeneratedQASample:
        spec = TEMPLATES[template_id]
        level = forced_level if forced_level is not None else str(self.rng.choice(spec.region_levels))
        if level not in spec.region_levels:
            raise ValueError(f'{template_id}: invalid region level {level}')
        selected_city = forced_city if spec.city_scope == 'single_city' else None
        if spec.city_scope == 'single_city' and selected_city is None:
            selected_city = str(self.rng.choice(self.city_order))
        if selected_city is not None and selected_city not in self.city_order:
            raise ValueError(f'invalid city {selected_city}')
        active_cities = [selected_city] if spec.city_scope == 'single_city' else list(self.city_order)
        modality = str(self.rng.choice(['inflow', 'outflow'])) if template_id.startswith('U') else None
        sampling_channel = modality if modality is not None else str(self.rng.choice(['inflow', 'outflow']))
        ids_by_city = None
        ids_a = ids_b = None
        sampling_meta = None
        if level == 'finegrain' and template_id in FINE_ACTIVITY_SINGLE | FINE_DYNAMICS_SINGLE:
            windows, ids_by_city, sampling_meta = self._sample_finegrain_cells(template_id, active_cities, sampling_channel, pair=False)
        elif level == 'finegrain' and template_id in FINE_DYNAMICS_PAIR:
            windows, ids_a, ids_b, sampling_meta = self._sample_finegrain_cells(template_id, active_cities, sampling_channel, pair=True)
        elif template_id in MOVEMENT_TEMPLATES:
            windows = self._sample_movement_windows(spec.city_scope, modality, selected_city)
        else:
            windows = self._sample_windows(spec.city_scope, selected_city)
        ch = CHANNEL_TO_INDEX[modality] if modality is not None else None
        time_step = int(self.rng.integers(1, 9))
        horizon = int(self.rng.integers(1, 5))
        horizon_set = [1, 2, 3, 4]
        rid = None
        rid_a = rid_b = None
        if level != 'finegrain':
            ids = region_ids(level)
            if template_id in FINE_DYNAMICS_PAIR:
                rid_a, rid_b = self.rng.choice(ids, size=2, replace=False).tolist()
            else:
                rid = str(self.rng.choice(ids))
        else:
            if ids_by_city is not None and len(active_cities) == 1:
                rid = ids_by_city[active_cities[0]]
            if ids_a is not None and len(active_cities) == 1:
                rid_a, rid_b = (ids_a[active_cities[0]], ids_b[active_cities[0]])
        t1 = t2 = None
        if template_id in {'U2', 'U5', 'U6', 'U13', 'U14', 'U16', 'U17'}:
            t1, t2 = sorted(self.rng.choice(np.arange(1, 9), size=2, replace=False).tolist())
        unit_phrase = self._unit_phrase(level, active_cities, rid, ids_by_city) if rid is not None or ids_by_city is not None else None
        pair_phrase = self._pair_phrase(level, active_cities, int(t1), int(t2), rid_a, rid_b, ids_a, ids_b) if template_id in {'U13', 'U14', 'U16', 'U17'} else None
        record = {'sample_id': f'{template_id}_{sample_index:07d}', 'template_id': template_id, 'block': spec.block, 'reasoning': spec.reasoning, 'city_scope': spec.city_scope, 'cities': list(windows.keys()), 'city_order': list(self.city_order), 'window_start_index': {city: int(w['start_index']) for city, w in windows.items()}, 'history_timestamps': {city: [_ts_string(t) for t in w['time_hist']] for city, w in windows.items()}, 'future_timestamps': {city: [_ts_string(t) for t in w['time_future']] for city, w in windows.items()}, 'region_level': level, 'region_id': rid, 'region_ids_by_city': ids_by_city, 'region_id_a': rid_a, 'region_id_b': rid_b, 'region_id_a_by_city': ids_a, 'region_id_b_by_city': ids_b, 'unit_phrase': unit_phrase, 'pair_phrase': pair_phrase, 'modality': modality, 'sampling_channel': sampling_channel if level == 'finegrain' and template_id in FINE_ACTIVITY_SINGLE | FINE_DYNAMICS_SINGLE | FINE_DYNAMICS_PAIR else None, 'sampling_meta': sampling_meta, 'movement_window_filter': 'history_only_all_t_whole_total_positive' if template_id in MOVEMENT_TEMPLATES else None, 'time_step': time_step if template_id in {'U7', 'U8', 'U9', 'U10', 'U11', 'U12'} else None, 't1': int(t1) if t1 is not None else None, 't2': int(t2) if t2 is not None else None, 'horizon': horizon if template_id in {'P2', 'P6'} else None, 'horizon_set': horizon_set if template_id in {'P1', 'P3', 'P4', 'P5', 'P7', 'P8'} else None, 'question': None, 'question_variant': None, 'answer_text': None, 'answer_labels': None, 'target_type': spec.target_type, 'numeric_target': None, 'tensor_target_ref': None, 'visual_path': None, 'loss_mask': loss_mask_for_target_type(spec.target_type)}

        def rid_for(city):
            if level == 'finegrain' and ids_by_city is not None:
                return ids_by_city[city]
            return rid

        def pair_for(city):
            if level == 'finegrain':
                return (ids_a[city], ids_b[city])
            return (rid_a, rid_b)
        tensor_target = None
        if template_id == 'U1':
            city, w = (selected_city, windows[selected_city])
            series = self._hist_series(w, city, modality, level, rid_for(city))
            record.update(answer_text=f'The 8-step sequence is {_fmt_vector(series)}.', numeric_target=_round_vector(series))
        elif template_id == 'U2':
            city, w = (selected_city, windows[selected_city])
            r = rid_for(city)
            v1 = region_value(w['X_hist'][t1 - 1, ch], city, level, r)
            v2 = region_value(w['X_hist'][t2 - 1, ch], city, level, r)
            delta = v2 - v1
            record.update(answer_text=f'The change is {_fmt_scalar(delta)}.', numeric_target=_round_scalar(delta))
        elif template_id == 'U3':
            city, w = (selected_city, windows[selected_city])
            trend = historical_trend(self._hist_series(w, city, modality, level, rid_for(city)))
            record.update(answer_text=f'The historical trend is {trend}.', answer_labels={'trend': trend})
        elif template_id == 'U4':
            series = {city: self._hist_series(w, city, modality, level, rid_for(city)) for city, w in windows.items()}
            record.update(answer_text=_join_city_parts(self.city_order, lambda c: f'{c}: {_fmt_vector(series[c])}'), numeric_target={c: _round_vector(series[c]) for c in self.city_order})
        elif template_id == 'U5':
            deltas = {}
            for city, w in windows.items():
                r = rid_for(city)
                v1 = region_value(w['X_hist'][t1 - 1, ch], city, level, r)
                v2 = region_value(w['X_hist'][t2 - 1, ch], city, level, r)
                deltas[city] = v2 - v1
            record.update(answer_text=_join_city_parts(self.city_order, lambda c: f'{c}: {_fmt_scalar(deltas[c])}'), numeric_target={c: _round_scalar(deltas[c]) for c in self.city_order})
        elif template_id == 'U6':
            scores = {}
            for city, w in windows.items():
                r = rid_for(city)
                v1 = region_value(w['X_hist'][t1 - 1, ch], city, level, r)
                v2 = region_value(w['X_hist'][t2 - 1, ch], city, level, r)
                scores[city] = normalized_temporal_change(v1, v2)
            winner = city_argmax(scores, self.city_order)
            record.update(answer_text=f'The city is {winner}.', answer_labels={'city': winner}, debug_normalized_change_scores={c: float(scores[c]) for c in self.city_order})
        elif template_id == 'U7':
            city, w = (selected_city, windows[selected_city])
            value = region_value(w['X_hist'][time_step - 1, ch], city, level, rid_for(city))
            record.update(answer_text=f'The value is {_fmt_scalar(value)}.', numeric_target=_round_scalar(value))
        elif template_id == 'U8':
            city, w = (selected_city, windows[selected_city])
            a, b = pair_for(city)
            va = region_value(w['X_hist'][time_step - 1, ch], city, level, a)
            vb = region_value(w['X_hist'][time_step - 1, ch], city, level, b)
            record['region_id_a'], record['region_id_b'] = (a, b)
            record.update(answer_text=f'The difference is {_fmt_scalar(va - vb)}.', numeric_target=_round_scalar(va - vb))
        elif template_id == 'U9':
            city, w = (selected_city, windows[selected_city])
            dom = dominant_region(w['X_hist'][time_step - 1, ch], city, level)
            record.update(answer_text=f'The dominant spatial unit is {dom}.', answer_labels={'region': dom})
        elif template_id == 'U10':
            values = {}
            for city, w in windows.items():
                values[city] = region_value(w['X_hist'][time_step - 1, ch], city, level, rid_for(city))
            record.update(answer_text=_join_city_parts(self.city_order, lambda c: f'{c}: {_fmt_scalar(values[c])}'), numeric_target={c: _round_scalar(values[c]) for c in self.city_order})
        elif template_id == 'U11':
            scores = {}
            for city, w in windows.items():
                scores[city] = normalized_peak_share(self._region_vec(w['X_hist'][time_step - 1, ch], city, level))
            record.update(answer_text=_join_city_parts(self.city_order, lambda c: f'{c}: {_fmt_ratio(scores[c])}'), numeric_target={c: _round_ratio(scores[c]) for c in self.city_order})
        elif template_id == 'U12':
            scores = {city: normalized_peak_share(self._region_vec(w['X_hist'][time_step - 1, ch], city, level)) for city, w in windows.items()}
            winner = city_argmax(scores, self.city_order)
            record.update(answer_text=f'The city is {winner}.', answer_labels={'city': winner}, debug_concentration_scores={c: float(scores[c]) for c in self.city_order})
        elif template_id == 'U13':
            city, w = (selected_city, windows[selected_city])
            a, b = pair_for(city)
            va = region_value(w['X_hist'][t1 - 1, ch], city, level, a)
            vb = region_value(w['X_hist'][t2 - 1, ch], city, level, b)
            record['region_id_a'], record['region_id_b'] = (a, b)
            vals = np.asarray([va, vb], dtype=np.float64)
            record.update(answer_text=f'The two values are {_fmt_vector(vals)}.', numeric_target=_round_vector(vals))
        elif template_id == 'U14':
            city, w = (selected_city, windows[selected_city])
            a, b = pair_for(city)
            va = region_value(w['X_hist'][t1 - 1, ch], city, level, a)
            vb = region_value(w['X_hist'][t2 - 1, ch], city, level, b)
            record['region_id_a'], record['region_id_b'] = (a, b)
            record.update(answer_text=f'The difference is {_fmt_scalar(vb - va)}.', numeric_target=_round_scalar(vb - va))
        elif template_id == 'U15':
            city, w = (selected_city, windows[selected_city])
            label, score, path = hotspot_movement(w['X_hist'][:, ch], city, level)
            record.update(answer_text=f'The hotspot movement is {label}.', answer_labels={'movement': label}, debug_hotspot_centroid_path=path, debug_movement_score=float(score))
        elif template_id == 'U16':
            vals = {}
            for city, w in windows.items():
                a, b = pair_for(city)
                va = region_value(w['X_hist'][t1 - 1, ch], city, level, a)
                vb = region_value(w['X_hist'][t2 - 1, ch], city, level, b)
                vals[city] = np.asarray([va, vb], dtype=np.float64)
            record.update(answer_text=_join_city_parts(self.city_order, lambda c: f'{c}: {_fmt_vector(vals[c])}'), numeric_target={c: _round_vector(vals[c]) for c in self.city_order})
        elif template_id == 'U17':
            deltas = {}
            for city, w in windows.items():
                a, b = pair_for(city)
                va = region_value(w['X_hist'][t1 - 1, ch], city, level, a)
                vb = region_value(w['X_hist'][t2 - 1, ch], city, level, b)
                deltas[city] = vb - va
            record.update(answer_text=_join_city_parts(self.city_order, lambda c: f'{c}: {_fmt_scalar(deltas[c])}'), numeric_target={c: _round_scalar(deltas[c]) for c in self.city_order})
        elif template_id == 'U18':
            scores, paths = ({}, {})
            for city, w in windows.items():
                _, score, path = hotspot_movement(w['X_hist'][:, ch], city, level)
                scores[city], paths[city] = (score, path)
            winner = city_argmax(scores, self.city_order)
            record.update(answer_text=f'The city is {winner}.', answer_labels={'city': winner}, debug_hotspot_centroid_paths=paths, debug_movement_scores={c: float(scores[c]) for c in self.city_order})
        elif template_id == 'P1':
            city, w = (selected_city, windows[selected_city])
            r = rid_for(city)
            trends = {channel: prediction_trend(self._future_series(w, city, channel, level, r, horizon_set)) for channel in ['inflow', 'outflow']}
            record.update(answer_text=f"Inflow: {trends['inflow']}; Outflow: {trends['outflow']}.", answer_labels=trends)
        elif template_id == 'P2':
            city, w = (selected_city, windows[selected_city])
            in_dom = dominant_region(w['Y_future'][horizon - 1, 0], city, level)
            out_dom = dominant_region(w['Y_future'][horizon - 1, 1], city, level)
            record.update(answer_text=f'Highest inflow unit: {in_dom}; Highest outflow unit: {out_dom}.', answer_labels={'inflow': in_dom, 'outflow': out_dom})
        elif template_id == 'P3':
            city, w = (selected_city, windows[selected_city])
            r = rid_for(city)
            in_vec = self._future_series(w, city, 'inflow', level, r, horizon_set)
            out_vec = self._future_series(w, city, 'outflow', level, r, horizon_set)
            record.update(answer_text=f'Inflow: {_fmt_vector(in_vec)}; Outflow: {_fmt_vector(out_vec)}.', numeric_target={'inflow': [float(x) for x in in_vec], 'outflow': [float(x) for x in out_vec]})
        elif template_id == 'P4':
            city, w = (selected_city, windows[selected_city])
            tensor_target = np.asarray(w['Y_future'], dtype=np.float32)
            H, W = tensor_target.shape[-2:]
            record.update(region_id='full_map', answer_text=f'The structured prediction target is a tensor sequence with shape [4,2,{H},{W}].')
        elif template_id == 'P5':
            city_trends = {}
            for city, w in windows.items():
                r = rid_for(city)
                city_trends[city] = {channel: prediction_trend(self._future_series(w, city, channel, level, r, horizon_set)) for channel in ['inflow', 'outflow']}
            record.update(answer_text=_join_city_parts(self.city_order, lambda c: f"{c}: in {city_trends[c]['inflow']}, out {city_trends[c]['outflow']}"), answer_labels=city_trends)
        elif template_id == 'P6':
            out = {}
            for city, w in windows.items():
                out[city] = {'inflow': dominant_region(w['Y_future'][horizon - 1, 0], city, level), 'outflow': dominant_region(w['Y_future'][horizon - 1, 1], city, level)}
            record.update(answer_text=_join_city_parts(self.city_order, lambda c: f"{c}: in {out[c]['inflow']}, out {out[c]['outflow']}"), answer_labels=out)
        elif template_id == 'P7':
            city_targets = {}
            for city, w in windows.items():
                r = rid_for(city)
                city_targets[city] = {'inflow': self._future_series(w, city, 'inflow', level, r, horizon_set), 'outflow': self._future_series(w, city, 'outflow', level, r, horizon_set)}
            record.update(answer_text=_join_city_parts(self.city_order, lambda c: f"{c}: in {_fmt_vector(city_targets[c]['inflow'])}, out {_fmt_vector(city_targets[c]['outflow'])}"), numeric_target={city: {channel: [float(x) for x in city_targets[city][channel]] for channel in ['inflow', 'outflow']} for city in self.city_order})
        elif template_id == 'P8':
            tensor_target = {city: np.asarray(w['Y_future'], dtype=np.float32) for city, w in windows.items()}
            record.update(region_id='full_map', answer_text=_format_p8_shapes(self.city_order, tensor_target))
        else:
            raise ValueError(f'Unknown template {template_id}')
        self._apply_question_variant(record, sample_index)
        return GeneratedQASample(record=record, windows=windows, tensor_target=tensor_target)
