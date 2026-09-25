import argparse
import json
from pathlib import Path
from config import SOURCE_CITY_ORDER, TARGET_CITY_ORDER
from templates import TEMPLATES

def _contains_text(obj, needle):
    if isinstance(obj, dict):
        return any((_contains_text(v, needle) for v in obj.values()))
    if isinstance(obj, (list, tuple)):
        return any((_contains_text(v, needle) for v in obj))
    if isinstance(obj, str):
        return needle in obj
    return False

def validate(path: Path, profile: str):
    errors = []
    n = 0
    if profile == 'source':
        expected = list(SOURCE_CITY_ORDER)
        forbidden = list(TARGET_CITY_ORDER)
    else:
        expected = list(TARGET_CITY_ORDER)
        forbidden = list(SOURCE_CITY_ORDER)
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            r = json.loads(line)
            tid = r.get('template_id')
            spec = TEMPLATES.get(tid)
            if spec is None:
                errors.append(f'line {line_no}: unknown template {tid}')
                continue
            cities = r.get('cities') or []
            if profile == 'source':
                if spec.city_scope == 'multi_city' and cities != expected:
                    errors.append(f'line {line_no}: source multi-city order={cities}, expected={expected}')
                if spec.city_scope == 'single_city' and (len(cities) != 1 or cities[0] not in expected):
                    errors.append(f'line {line_no}: invalid source single-city record={cities}')
            else:
                if spec.city_scope != 'single_city':
                    errors.append(f'line {line_no}: held-out target contains multi-city template {tid}')
                if cities != expected:
                    errors.append(f'line {line_no}: target cities={cities}, expected={expected}')
            for city in forbidden:
                visible = {'cities': r.get('cities'), 'question': r.get('question'), 'answer_text': r.get('answer_text'), 'numeric_target': r.get('numeric_target'), 'messages': r.get('messages')}
                if _contains_text(visible, city):
                    errors.append(f'line {line_no}: forbidden city {city} leaked into {profile} QA')
            declared = r.get('city_order')
            if declared != expected:
                errors.append(f'line {line_no}: city_order={declared}, expected={expected}')
    return {'profile': profile, 'records': n, 'errors': errors, 'pass': not errors}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', type=Path, required=True)
    ap.add_argument('--profile', choices=['source', 'target-dc'], required=True)
    args = ap.parse_args()
    report = validate(args.jsonl, args.profile)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['pass']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
