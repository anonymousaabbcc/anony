import argparse
import json
from pathlib import Path
import numpy as np
from config import SOURCE_CITY_ORDER, TARGET_CITY_ORDER, CHANNEL_TO_INDEX
from split_loader import load_all_split
from sampling import FinegrainSampler
from regions import parse_finegrain_id

def _iter_cells(meta):
    if not meta:
        return []
    if 'cells' in meta:
        return list(meta['cells'])
    return [meta]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', type=Path, required=True)
    ap.add_argument('--split', choices=['train', 'valid', 'test'], required=True)
    ap.add_argument('--profile', choices=['source', 'target-dc'], default='source')
    args = ap.parse_args()
    city_order = SOURCE_CITY_ORDER if args.profile == 'source' else TARGET_CITY_ORDER
    active = load_all_split(args.split, city_order=city_order)
    train = load_all_split('train', city_order=city_order)
    sampler = FinegrainSampler(active, train, seed=12345)
    errors = []
    checked = 0
    prediction_checked = 0
    with args.jsonl.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            meta_by_city = rec.get('sampling_meta') or {}
            if not meta_by_city:
                continue
            if 'future' in json.dumps(meta_by_city).lower():
                errors.append(f'line {line_no}: sampling metadata contains future-derived field')
            if rec['template_id'] in {'P1', 'P3', 'P5', 'P7'}:
                prediction_checked += 1
            for city, meta in meta_by_city.items():
                channel = meta.get('selection_channel')
                mode = meta.get('mode')
                if channel not in CHANNEL_TO_INDEX:
                    errors.append(f'line {line_no}: {city} invalid sampling channel {channel}')
                    continue
                if meta.get('basis') != 'history_only_t1_t8':
                    errors.append(f'line {line_no}: {city} sampling basis is not history-only')
                start = int(rec['window_start_index'][city])
                ch = CHANNEL_TO_INDEX[channel]
                hist = active[city].data[start:start + 8, ch]
                zero, A, D, NZ = sampler._window_metrics(hist)
                stats = sampler.reference[city, channel]
                for cell in _iter_cells(meta):
                    checked += 1
                    if int(cell['start_index']) != start:
                        errors.append(f'line {line_no}: {city} sampled start != record start')
                    r, c = parse_finegrain_id(cell['region_id'])
                    expected_stratum, expected_score = stats.classify(bool(zero[r, c]), float(A[r, c]), float(D[r, c]), mode)
                    checks = {'stratum': cell['stratum'] == expected_stratum, 'activity': np.isclose(float(cell['activity']), float(A[r, c]), atol=1e-10), 'dynamics': np.isclose(float(cell['dynamics']), float(D[r, c]), atol=1e-10), 'nonzero_fraction': np.isclose(float(cell['nonzero_fraction']), float(NZ[r, c]), atol=1e-10), 'q0': np.isclose(float(cell['zero_sampling_probability_q0']), stats.q0, atol=1e-12)}
                    if expected_score is None:
                        checks['score'] = cell.get('score') is None
                    else:
                        checks['score'] = np.isclose(float(cell['score']), float(expected_score), atol=1e-10)
                    bad = [k for k, v in checks.items() if not v]
                    if bad:
                        errors.append(f"line {line_no}: {city}/{cell['region_id']} sampling mismatch {bad}")
    report = {'checked_cells': checked, 'prediction_finegrain_records_checked': prediction_checked, 'errors': errors, 'pass': not errors}
    print(json.dumps(report, indent=2))
    out = args.jsonl.parent / 'sampling_validation_report.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if errors:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
