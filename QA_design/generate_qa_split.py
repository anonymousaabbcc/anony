import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from config import SOURCE_CITY_ORDER, TARGET_CITY_ORDER, QA_SOURCE_OUTPUT_ROOT, QA_TARGET_OUTPUT_ROOT, SYSTEM_PROMPT
from generator import QAGenerator
from renderer import load_train_scales, render_sample_images
from split_loader import load_all_split
from templates import ALL_TEMPLATE_IDS, SINGLE_CITY_TEMPLATE_IDS, TEMPLATES
from validate_qa import validate_jsonl
from sampling import FinegrainSampler
from audit_qa_dataset import build_audit

def uri(p):
    return Path(p).resolve().as_uri()

def messages(rec, paths):
    user = []
    for city, path in zip(rec['cities'], paths):
        user.append({'type': 'text', 'text': f'{city} historical urban flow maps:'})
        user.append({'type': 'image', 'image': uri(path)})
    user.append({'type': 'text', 'text': rec['question']})
    return [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': [{'type': 'text', 'text': rec['answer_text']}]}]

def profile_config(profile):
    if profile == 'source':
        return {'city_order': list(SOURCE_CITY_ORDER), 'output_root': QA_SOURCE_OUTPUT_ROOT, 'default_templates': list(ALL_TEMPLATE_IDS), 'visual_reference': None, 'sampling_reference_note': "each source city's own train split"}
    if profile == 'target-dc':
        return {'city_order': list(TARGET_CITY_ORDER), 'output_root': QA_TARGET_OUTPUT_ROOT, 'default_templates': list(SINGLE_CITY_TEMPLATE_IDS), 'visual_reference': list(SOURCE_CITY_ORDER), 'sampling_reference_note': 'BIKEDC train histories only; do not use target QA for strict zero-shot training'}
    raise ValueError(profile)

def schedule(tid, n, city_order):
    spec = TEMPLATES[tid]
    out = []
    n_cities = len(city_order)
    levels = tuple(spec.region_levels)
    n_levels = len(levels)
    for j in range(n):
        if spec.city_scope == 'single_city':
            city_idx = j % n_cities
            block_idx = j // n_cities
            city = city_order[city_idx]
            level_idx = (city_idx + block_idx) % n_levels
            level = levels[level_idx]
        else:
            if n_cities < 2:
                raise ValueError(f'{tid} is multi_city but active profile has only {n_cities} city.')
            city = None
            level = levels[j % n_levels]
        out.append((city, level))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', choices=['source', 'target-dc'], default='source')
    ap.add_argument('--split', choices=['train', 'valid', 'test'], required=True)
    ap.add_argument('--samples-per-template', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--templates', default='all', help="Comma-separated template IDs. 'all' means all 26 for source and all single-city templates for target-dc.")
    ap.add_argument('--no-render', action='store_true')
    args = ap.parse_args()
    profile = profile_config(args.profile)
    city_order = profile['city_order']
    out = (profile['output_root'] / args.split).resolve()
    img = out / 'images'
    ten = out / 'tensor_targets'
    out.mkdir(parents=True, exist_ok=True)
    img.mkdir(exist_ok=True)
    ten.mkdir(exist_ok=True)
    tids = list(profile['default_templates']) if args.templates == 'all' else [x.strip() for x in args.templates.split(',') if x.strip()]
    bad = [x for x in tids if x not in TEMPLATES]
    if bad:
        raise ValueError(f'Unknown templates: {bad}')
    if args.profile == 'target-dc':
        invalid = [tid for tid in tids if TEMPLATES[tid].city_scope != 'single_city']
        if invalid:
            raise ValueError(f'target-dc profile only supports single-city templates; invalid={invalid}')
    active_cities = load_all_split(args.split, city_order=city_order)
    train_reference_cities = load_all_split('train', city_order=city_order)
    fine_sampler = FinegrainSampler(active_cities=active_cities, train_reference_cities=train_reference_cities, seed=args.seed + 100003)
    (out / 'finegrain_sampling_reference.json').write_text(json.dumps(fine_sampler.reference_summary(), indent=2), encoding='utf-8')
    gen = QAGenerator(active_cities, seed=args.seed, city_order=city_order, finegrain_sampler=fine_sampler)
    if args.no_render:
        scales = None
        scale_policy = 'no_render'
    else:
        scales = load_train_scales(city_order=city_order, reference_city_order=profile['visual_reference'])
        scale_policy = 'own_city_train_only' if profile['visual_reference'] is None else 'source_train_envelope_no_target_stats'
    counts = Counter()
    city_counts = Counter()
    level_counts = Counter()
    variant_counts = Counter()
    idx = 0
    path = out / f'QA_{args.split}.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for tid in tids:
            for city, level in schedule(tid, args.samples_per_template, city_order):
                sample = gen.generate(tid, idx, forced_city=city, forced_level=level)
                rec = sample.record
                rec['split'] = args.split
                rec['dataset_profile'] = args.profile
                rec['visual_scale_policy'] = scale_policy
                rec['finegrain_sampling_reference'] = profile['sampling_reference_note']
                if sample.tensor_target is not None:
                    tp = (ten / f"{rec['sample_id']}.npz").resolve()
                    if isinstance(sample.tensor_target, dict):
                        np.savez_compressed(tp, **{k.replace('-', '_'): v for k, v in sample.tensor_target.items()})
                    else:
                        np.savez_compressed(tp, target=sample.tensor_target)
                    rec['tensor_target_ref'] = str(tp)
                paths = [] if args.no_render else render_sample_images(sample, img, scales)
                rec['image_paths'] = [str(p) for p in paths]
                rec['visual_path'] = str(paths[0]) if paths else None
                rec['messages'] = messages(rec, paths) if paths else None
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                counts[tid] += 1
                level_counts[tid, level] += 1
                variant_counts[tid, int(rec['question_variant'])] += 1
                if city:
                    city_counts[tid, city] += 1
                idx += 1
    val = validate_jsonl(path, expected_city_order=city_order)
    audit = build_audit(path, out)
    summary = {'profile': args.profile, 'split': args.split, 'city_order': city_order, 'templates': tids, 'samples_per_template': args.samples_per_template, 'total_samples': idx, 'template_counts': dict(counts), 'level_counts': {f'{tid}/{level}': count for (tid, level), count in sorted(level_counts.items())}, 'single_city_counts': {f'{tid}/{city}': count for (tid, city), count in sorted(city_counts.items())}, 'question_variant_counts': {f'{tid}/v{variant}': count for (tid, variant), count in sorted(variant_counts.items())}, 'visual_scale_policy': scale_policy, 'finegrain_sampling_reference': profile['sampling_reference_note'], 'audit_csv': audit['csv'], 'audit_json': audit['json'], 'validation_pass': val['pass']}
    (out / 'generation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'qa_validation_report.json').write_text(json.dumps(val, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Output:', path)
    if not val['pass']:
        print(val['errors'][:20])
        raise SystemExit(1)
if __name__ == '__main__':
    main()
