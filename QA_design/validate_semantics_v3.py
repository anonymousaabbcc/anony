import argparse
import json
from pathlib import Path
import numpy as np
from config import SOURCE_CITY_ORDER, TARGET_CITY_ORDER, CHANNEL_TO_INDEX, TEXT_DECIMALS, RATIO_DECIMALS
from split_loader import load_all_split
from regions import region_value, region_vector, dominant_region
from targets import historical_trend, normalized_temporal_change, normalized_peak_share, hotspot_movement, prediction_trend, city_argmax

def fmt_scalar(x):
    return f'{float(x):.{TEXT_DECIMALS}f}'

def fmt_vector(x):
    return '[' + ', '.join((fmt_scalar(v) for v in np.asarray(x).tolist())) + ']'

def fmt_ratio(x):
    return f'{float(x):.{RATIO_DECIMALS}f}'

def round_scalar(x):
    return round(float(x), TEXT_DECIMALS)

def round_vector(x):
    return [round_scalar(v) for v in np.asarray(x).tolist()]

def join_city(order, fn):
    return '; '.join((fn(c) for c in order)) + '.'

def close_nested(a, b, atol=1e-06):
    if isinstance(b, dict):
        return isinstance(a, dict) and set(a) == set(b) and all((close_nested(a[k], b[k], atol) for k in b))
    if isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and len(a) == len(b) and all((close_nested(x, y, atol) for x, y in zip(a, b)))
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=0))

def rid_for(r, city):
    d = r.get('region_ids_by_city') or {}
    return d.get(city, r.get('region_id'))

def pair_for(r, city):
    da = r.get('region_id_a_by_city') or {}
    db = r.get('region_id_b_by_city') or {}
    return (da.get(city, r.get('region_id_a')), db.get(city, r.get('region_id_b')))

def window(r, data, city):
    s = int(r['window_start_index'][city])
    x = data[city].data[s:s + 8]
    y = data[city].data[s + 8:s + 12]
    return (x, y)

def hist_series(r, data, city, channel, rid):
    x, _ = window(r, data, city)
    ch = CHANNEL_TO_INDEX[channel]
    return np.asarray([region_value(x[t, ch], city, r['region_level'], rid) for t in range(8)], dtype=np.float64)

def future_series(r, data, city, channel, rid):
    _, y = window(r, data, city)
    ch = CHANNEL_TO_INDEX[channel]
    return np.asarray([region_value(y[h, ch], city, r['region_level'], rid) for h in range(4)], dtype=np.float64)

def validate_record(r, data, city_order):
    tid, level = (r['template_id'], r['region_level'])
    mod = r.get('modality')
    ch = CHANNEL_TO_INDEX[mod] if mod else None
    if tid == 'U1':
        c = r['cities'][0]
        v = hist_series(r, data, c, mod, rid_for(r, c))
        exp = round_vector(v)
        ans = f'The 8-step sequence is {fmt_vector(v)}.'
        return close_nested(r['numeric_target'], exp) and r['answer_text'] == ans
    if tid == 'U2':
        c = r['cities'][0]
        x, _ = window(r, data, c)
        rr = rid_for(r, c)
        t1, t2 = (r['t1'], r['t2'])
        d = region_value(x[t2 - 1, ch], c, level, rr) - region_value(x[t1 - 1, ch], c, level, rr)
        return close_nested(r['numeric_target'], round_scalar(d)) and r['answer_text'] == f'The change is {fmt_scalar(d)}.'
    if tid == 'U3':
        c = r['cities'][0]
        tr = historical_trend(hist_series(r, data, c, mod, rid_for(r, c)))
        return r['answer_text'] == f'The historical trend is {tr}.' and r.get('answer_labels') == {'trend': tr}
    if tid == 'U4':
        vals = {c: hist_series(r, data, c, mod, rid_for(r, c)) for c in city_order}
        ans = join_city(city_order, lambda c: f'{c}: {fmt_vector(vals[c])}')
        exp = {c: round_vector(vals[c]) for c in city_order}
        return r['answer_text'] == ans and close_nested(r['numeric_target'], exp)
    if tid == 'U5':
        vals = {}
        t1, t2 = (r['t1'], r['t2'])
        for c in city_order:
            x, _ = window(r, data, c)
            rr = rid_for(r, c)
            vals[c] = region_value(x[t2 - 1, ch], c, level, rr) - region_value(x[t1 - 1, ch], c, level, rr)
        return r['answer_text'] == join_city(city_order, lambda c: f'{c}: {fmt_scalar(vals[c])}') and close_nested(r['numeric_target'], {c: round_scalar(vals[c]) for c in city_order})
    if tid == 'U6':
        scores = {}
        t1, t2 = (r['t1'], r['t2'])
        for c in city_order:
            x, _ = window(r, data, c)
            rr = rid_for(r, c)
            v1 = region_value(x[t1 - 1, ch], c, level, rr)
            v2 = region_value(x[t2 - 1, ch], c, level, rr)
            scores[c] = normalized_temporal_change(v1, v2)
        winner = city_argmax(scores, city_order)
        return r['answer_text'] == f'The city is {winner}.' and r.get('answer_labels') == {'city': winner}
    if tid == 'U7':
        c = r['cities'][0]
        x, _ = window(r, data, c)
        v = region_value(x[r['time_step'] - 1, ch], c, level, rid_for(r, c))
        return r['answer_text'] == f'The value is {fmt_scalar(v)}.' and close_nested(r['numeric_target'], round_scalar(v))
    if tid == 'U8':
        c = r['cities'][0]
        x, _ = window(r, data, c)
        a, b = pair_for(r, c)
        t = r['time_step']
        d = region_value(x[t - 1, ch], c, level, a) - region_value(x[t - 1, ch], c, level, b)
        return r['answer_text'] == f'The difference is {fmt_scalar(d)}.' and close_nested(r['numeric_target'], round_scalar(d))
    if tid == 'U9':
        c = r['cities'][0]
        x, _ = window(r, data, c)
        dom = dominant_region(x[r['time_step'] - 1, ch], c, level)
        return r['answer_text'] == f'The dominant spatial unit is {dom}.' and r.get('answer_labels') == {'region': dom}
    if tid == 'U10':
        vals = {}
        for c in city_order:
            x, _ = window(r, data, c)
            vals[c] = region_value(x[r['time_step'] - 1, ch], c, level, rid_for(r, c))
        return r['answer_text'] == join_city(city_order, lambda c: f'{c}: {fmt_scalar(vals[c])}') and close_nested(r['numeric_target'], {c: round_scalar(vals[c]) for c in city_order})
    if tid == 'U11':
        scores = {}
        for c in city_order:
            x, _ = window(r, data, c)
            scores[c] = normalized_peak_share(region_vector(x[r['time_step'] - 1, ch], c, level))
        return r['answer_text'] == join_city(city_order, lambda c: f'{c}: {fmt_ratio(scores[c])}') and close_nested(r['numeric_target'], {c: round(float(scores[c]), RATIO_DECIMALS) for c in city_order})
    if tid == 'U12':
        scores = {}
        for c in city_order:
            x, _ = window(r, data, c)
            scores[c] = normalized_peak_share(region_vector(x[r['time_step'] - 1, ch], c, level))
        winner = city_argmax(scores, city_order)
        return r['answer_text'] == f'The city is {winner}.' and r.get('answer_labels') == {'city': winner}
    if tid in {'U13', 'U14'}:
        c = r['cities'][0]
        x, _ = window(r, data, c)
        a, b = pair_for(r, c)
        t1, t2 = (r['t1'], r['t2'])
        va = region_value(x[t1 - 1, ch], c, level, a)
        vb = region_value(x[t2 - 1, ch], c, level, b)
        if tid == 'U13':
            vals = [va, vb]
            return r['answer_text'] == f'The two values are {fmt_vector(vals)}.' and close_nested(r['numeric_target'], round_vector(vals))
        d = vb - va
        return r['answer_text'] == f'The difference is {fmt_scalar(d)}.' and close_nested(r['numeric_target'], round_scalar(d))
    if tid == 'U15':
        c = r['cities'][0]
        x, _ = window(r, data, c)
        label, _, _ = hotspot_movement(x[:, ch], c, level)
        return r['answer_text'] == f'The hotspot movement is {label}.' and r.get('answer_labels') == {'movement': label}
    if tid in {'U16', 'U17'}:
        t1, t2 = (r['t1'], r['t2'])
        vals = {}
        for c in city_order:
            x, _ = window(r, data, c)
            a, b = pair_for(r, c)
            va = region_value(x[t1 - 1, ch], c, level, a)
            vb = region_value(x[t2 - 1, ch], c, level, b)
            vals[c] = [va, vb] if tid == 'U16' else vb - va
        if tid == 'U16':
            return r['answer_text'] == join_city(city_order, lambda c: f'{c}: {fmt_vector(vals[c])}') and close_nested(r['numeric_target'], {c: round_vector(vals[c]) for c in city_order})
        return r['answer_text'] == join_city(city_order, lambda c: f'{c}: {fmt_scalar(vals[c])}') and close_nested(r['numeric_target'], {c: round_scalar(vals[c]) for c in city_order})
    if tid == 'U18':
        scores = {}
        for c in city_order:
            x, _ = window(r, data, c)
            _, score, _ = hotspot_movement(x[:, ch], c, level)
            scores[c] = score
        winner = city_argmax(scores, city_order)
        return r['answer_text'] == f'The city is {winner}.' and r.get('answer_labels') == {'city': winner}
    if tid == 'P1':
        c = r['cities'][0]
        rr = rid_for(r, c)
        tr = {chh: prediction_trend(future_series(r, data, c, chh, rr)) for chh in ['inflow', 'outflow']}
        return r['answer_text'] == f"Inflow: {tr['inflow']}; Outflow: {tr['outflow']}." and r.get('answer_labels') == tr
    if tid == 'P2':
        c = r['cities'][0]
        _, y = window(r, data, c)
        h = r['horizon']
        ind = dominant_region(y[h - 1, 0], c, level)
        outd = dominant_region(y[h - 1, 1], c, level)
        return r['answer_text'] == f'Highest inflow unit: {ind}; Highest outflow unit: {outd}.' and r.get('answer_labels') == {'inflow': ind, 'outflow': outd}
    if tid == 'P3':
        c = r['cities'][0]
        rr = rid_for(r, c)
        iv = future_series(r, data, c, 'inflow', rr)
        ov = future_series(r, data, c, 'outflow', rr)
        exp = {'inflow': [float(x) for x in iv], 'outflow': [float(x) for x in ov]}
        return r['answer_text'] == f'Inflow: {fmt_vector(iv)}; Outflow: {fmt_vector(ov)}.' and close_nested(r['numeric_target'], exp)
    if tid == 'P4':
        c = r['cities'][0]
        _, y = window(r, data, c)
        tp = Path(r['tensor_target_ref'])
        if not tp.exists():
            return False
        with np.load(tp) as f:
            actual = f['target']
        H, W = y.shape[-2:]
        return np.array_equal(actual, np.asarray(y, dtype=np.float32)) and r['answer_text'] == f'The structured prediction target is a tensor sequence with shape [4,2,{H},{W}].'
    if tid == 'P5':
        tr = {}
        for c in city_order:
            rr = rid_for(r, c)
            tr[c] = {chh: prediction_trend(future_series(r, data, c, chh, rr)) for chh in ['inflow', 'outflow']}
        ans = join_city(city_order, lambda c: f"{c}: in {tr[c]['inflow']}, out {tr[c]['outflow']}")
        return r['answer_text'] == ans and r.get('answer_labels') == tr
    if tid == 'P6':
        h = r['horizon']
        out = {}
        for c in city_order:
            _, y = window(r, data, c)
            out[c] = {'inflow': dominant_region(y[h - 1, 0], c, level), 'outflow': dominant_region(y[h - 1, 1], c, level)}
        return r['answer_text'] == join_city(city_order, lambda c: f"{c}: in {out[c]['inflow']}, out {out[c]['outflow']}") and r.get('answer_labels') == out
    if tid == 'P7':
        vals = {}
        for c in city_order:
            rr = rid_for(r, c)
            vals[c] = {'inflow': future_series(r, data, c, 'inflow', rr), 'outflow': future_series(r, data, c, 'outflow', rr)}
        ans = join_city(city_order, lambda c: f"{c}: in {fmt_vector(vals[c]['inflow'])}, out {fmt_vector(vals[c]['outflow'])}")
        exp = {c: {chh: [float(x) for x in vals[c][chh]] for chh in ['inflow', 'outflow']} for c in city_order}
        return r['answer_text'] == ans and close_nested(r['numeric_target'], exp)
    if tid == 'P8':
        tp = Path(r['tensor_target_ref'])
        if not tp.exists():
            return False
        expected = {}
        for c in city_order:
            _, y = window(r, data, c)
            expected[c] = np.asarray(y, dtype=np.float32)
        with np.load(tp) as f:
            ok = all((c.replace('-', '_') in f.files and np.array_equal(f[c.replace('-', '_')], expected[c]) for c in city_order))
        ans = join_city(city_order, lambda c: f"{c}: [{','.join((str(int(x)) for x in expected[c].shape))}]")
        return ok and r['answer_text'] == ans
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', type=Path, required=True)
    ap.add_argument('--split', choices=['train', 'valid', 'test'], required=True)
    ap.add_argument('--profile', choices=['source', 'target-dc'], default='source')
    ap.add_argument('--block', choices=['all', 'understanding', 'prediction'], default='all')
    args = ap.parse_args()
    city_order = SOURCE_CITY_ORDER if args.profile == 'source' else TARGET_CITY_ORDER
    data = load_all_split(args.split, city_order=city_order)
    counts = {}
    failures = []
    with args.jsonl.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tid = r['template_id']
            if args.block == 'understanding' and (not tid.startswith('U')):
                continue
            if args.block == 'prediction' and (not tid.startswith('P')):
                continue
            counts.setdefault(tid, {'pass': 0, 'fail': 0})
            try:
                ok = validate_record(r, data, list(city_order))
            except Exception as e:
                ok = False
                failures.append({'sample_id': r.get('sample_id'), 'template_id': tid, 'error': repr(e)})
            if ok:
                counts[tid]['pass'] += 1
            else:
                counts[tid]['fail'] += 1
                if not any((x.get('sample_id') == r.get('sample_id') for x in failures)):
                    failures.append({'sample_id': r.get('sample_id'), 'template_id': tid, 'error': 'target/answer mismatch'})
    report = {'counts': counts, 'failures': failures[:50], 'total_fail': sum((x['fail'] for x in counts.values())), 'pass': not failures}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = args.jsonl.parent / f'semantic_validation_{args.block}.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if not report['pass']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
