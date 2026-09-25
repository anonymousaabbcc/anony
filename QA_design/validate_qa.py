import argparse
import json
import math
from collections import Counter
from pathlib import Path
from templates import TEMPLATES

def _all_finite(obj):
    if obj is None:
        return True
    if isinstance(obj, dict):
        return all((_all_finite(v) for v in obj.values()))
    if isinstance(obj, (list, tuple)):
        return all((_all_finite(v) for v in obj))
    if isinstance(obj, (int, float)):
        return math.isfinite(float(obj))
    return True

def validate_jsonl(jsonl_path: Path, expected_city_order=None):
    errors = []
    ids = set()
    template_counts = Counter()
    target_counts = Counter()
    rows = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append(rec)
            sid = rec.get('sample_id')
            if sid in ids:
                errors.append(f'line {line_no}: duplicate sample_id {sid}')
            ids.add(sid)
            tid = rec.get('template_id')
            if tid not in TEMPLATES:
                errors.append(f'line {line_no}: unknown template_id {tid}')
                continue
            spec = TEMPLATES[tid]
            template_counts[tid] += 1
            if rec.get('region_level') not in spec.region_levels:
                errors.append(f"line {line_no}: {tid} invalid level={rec.get('region_level')}")
            if rec.get('target_type') != spec.target_type:
                errors.append(f'line {line_no}: {tid} target_type mismatch')
            target_counts[rec.get('target_type')] += 1
            q, a = (rec.get('question') or '', rec.get('answer_text') or '')
            if '{' in q or '}' in q or '{' in a or ('}' in a):
                errors.append(f'line {line_no}: unresolved template placeholder')
            if 'corresponding fine-grained' in q.lower():
                errors.append(f'line {line_no}: forbidden cross-city correspondence wording')
            if not _all_finite(rec.get('numeric_target')):
                errors.append(f'line {line_no}: non-finite numeric target')
            if tid in {'P1', 'P3', 'P5', 'P7'} and rec.get('region_level') == 'finegrain':
                meta = rec.get('sampling_meta') or {}
                if not meta:
                    errors.append(f'line {line_no}: missing finegrain sampling metadata')
                for city, m in meta.items():
                    if m.get('basis') != 'history_only_t1_t8':
                        errors.append(f'line {line_no}: {city} prediction cell selection not marked history-only')
            if tid in {'U15', 'U18'}:
                if rec.get('movement_window_filter') != 'history_only_all_t_whole_total_positive':
                    errors.append(f'line {line_no}: movement historical activity filter missing')
            if tid in {'P4', 'P8'} and rec.get('region_id') != 'full_map':
                errors.append(f'line {line_no}: {tid} must use full_map target')
            vp = rec.get('visual_path')
            if vp and (not Path(vp).exists()):
                errors.append(f'line {line_no}: missing visual_path {vp}')
            tp = rec.get('tensor_target_ref')
            if tp and (not Path(tp).exists()):
                errors.append(f'line {line_no}: missing tensor_target_ref {tp}')
            if expected_city_order is not None:
                expected = list(expected_city_order)
                if rec.get('city_scope') == 'multi_city' and rec.get('cities') != expected:
                    errors.append(f"line {line_no}: wrong multi-city order; got={rec.get('cities')} expected={expected}")
                elif rec.get('city_scope') == 'single_city':
                    cities = rec.get('cities') or []
                    if len(cities) != 1 or cities[0] not in expected:
                        errors.append(f'line {line_no}: single-city record outside active city set: {cities}')
            for city, stamps in (rec.get('future_timestamps') or {}).items():
                for ts in stamps:
                    if ts in q:
                        errors.append(f'line {line_no}: future timestamp leaked into question for {city}')
    return {'records': len(rows), 'unique_sample_ids': len(ids), 'template_counts': dict(sorted(template_counts.items())), 'target_type_counts': dict(sorted(target_counts.items())), 'errors': errors, 'pass': len(errors) == 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl_path', type=Path)
    args = parser.parse_args()
    report = validate_jsonl(args.jsonl_path)
    out = args.jsonl_path.parent / 'qa_validation_report.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['pass']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
