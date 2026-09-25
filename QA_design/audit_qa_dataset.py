import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

def _flatten_sampling_meta(rec):
    meta = rec.get('sampling_meta')
    if not meta:
        return []
    rows = []
    for city, m in meta.items():
        if 'cells' in m:
            for cell in m['cells']:
                rows.append((city, cell))
        else:
            rows.append((city, m))
    return rows

def _answer_keys(rec):
    labels = rec.get('answer_labels')
    if labels is None:
        return []
    if isinstance(labels, dict):
        out = []

        def walk(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f'{prefix}/{k}' if prefix else str(k), v)
            else:
                out.append(f'{prefix}={obj}')
        walk('', labels)
        return out
    return [str(labels)]

def build_audit(jsonl_path: Path, output_dir: Path | None=None):
    output_dir = jsonl_path.parent if output_dir is None else output_dir
    groups = defaultdict(lambda: {'samples': 0, 'sampling_entries': 0, 'strata': Counter(), 'nonzero_fraction': [], 'activity': [], 'dynamics': [], 'answers': Counter()})
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            tid, level = (rec['template_id'], rec['region_level'])
            for city in rec['cities']:
                g = groups[city, tid, level]
                g['samples'] += 1
                for ans in _answer_keys(rec):
                    g['answers'][ans] += 1
            for city, m in _flatten_sampling_meta(rec):
                g = groups[city, tid, level]
                g['sampling_entries'] += 1
                g['strata'][m['stratum']] += 1
                g['nonzero_fraction'].append(float(m['nonzero_fraction']))
                g['activity'].append(float(m['activity']))
                g['dynamics'].append(float(m['dynamics']))
    rows = []
    for (city, tid, level), g in sorted(groups.items()):
        n = g['sampling_entries']

        def rate(s):
            return g['strata'][s] / n if n else None

        def mean(xs):
            return sum(xs) / len(xs) if xs else None
        rows.append({'city': city, 'template': tid, 'level': level, 'samples': g['samples'], 'sampling_entries': n, 'zero_history_rate': rate('Z'), 'positive_low_rate': rate('L'), 'positive_mid_rate': rate('M'), 'positive_high_rate': rate('H'), 'mean_nonzero_fraction': mean(g['nonzero_fraction']), 'mean_activity': mean(g['activity']), 'mean_dynamics': mean(g['dynamics']), 'answer_distribution': dict(g['answers'])})
    csv_path = output_dir / 'qa_dataset_audit.csv'
    json_path = output_dir / 'qa_dataset_audit.json'
    fields = ['city', 'template', 'level', 'samples', 'sampling_entries', 'zero_history_rate', 'positive_low_rate', 'positive_mid_rate', 'positive_high_rate', 'mean_nonzero_fraction', 'mean_activity', 'mean_dynamics', 'answer_distribution']
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            r = dict(row)
            r['answer_distribution'] = json.dumps(r['answer_distribution'], ensure_ascii=False, sort_keys=True)
            w.writerow(r)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'rows': rows, 'csv': str(csv_path), 'json': str(json_path)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl', type=Path)
    args = ap.parse_args()
    result = build_audit(args.jsonl)
    print(json.dumps({'csv': result['csv'], 'json': result['json'], 'rows': len(result['rows'])}, indent=2))
if __name__ == '__main__':
    main()
