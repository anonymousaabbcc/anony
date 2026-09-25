import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .config import SOURCE_CITY_ORDER, Stage2Config
from .qa_compat import import_qa_design
DEFAULT_STAGE1_QA_ROOT = Path(os.environ.get('URBANBIND_QA_SOURCE_ROOT', './outputs/qa/source'))
EXPECTED_STAGE1_SCALE_POLICY = 'own_city_train_only'

def _qa_jsonl(root: Path, split: str) -> Path:
    return root / split / f'QA_{split}.jsonl'

def _build_stage1_p8_index(root: Path, split: str):
    path = _qa_jsonl(root, split)
    if not path.exists():
        raise FileNotFoundError(f'Stage-1 QA JSONL not found: {path}')
    index = {city: {} for city in SOURCE_CITY_ORDER}
    duplicate_same_start = defaultdict(int)
    records = 0
    eligible_records = 0
    missing_source_images = 0
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            records += 1
            if rec.get('template_id') != 'P8':
                continue
            if rec.get('block') != 'Prediction' or rec.get('region_level') != 'finegrain':
                continue
            policy = rec.get('visual_scale_policy')
            if policy not in (None, EXPECTED_STAGE1_SCALE_POLICY):
                continue
            cities = list(rec.get('cities') or rec.get('city_order') or [])
            starts = rec.get('window_start_index') or {}
            paths = list(rec.get('image_paths') or [])
            if len(paths) != len(cities):
                raise RuntimeError(f'Malformed P8 image_paths at {path}:{line_no}: {len(paths)} paths for {len(cities)} cities')
            eligible_records += 1
            for city, image_path in zip(cities, paths):
                if city not in index or city not in starts:
                    continue
                src = Path(image_path)
                if not src.exists():
                    missing_source_images += 1
                    continue
                start = int(starts[city])
                if start in index[city]:
                    duplicate_same_start[city, start] += 1
                    continue
                index[city][start] = src.resolve()
    return (index, {'jsonl': str(path), 'records_total': records, 'eligible_p8_records': eligible_records, 'missing_source_images': missing_source_images, 'unique_p8_windows': {city: len(index[city]) for city in SOURCE_CITY_ORDER}, 'duplicate_city_start_entries': int(sum(duplicate_same_start.values()))})

def _pixel_equal(a: Path, b: Path) -> bool:
    with Image.open(a) as ia, Image.open(b) as ib:
        aa = np.asarray(ia.convert('RGBA'))
        bb = np.asarray(ib.convert('RGBA'))
    return aa.shape == bb.shape and np.array_equal(aa, bb)

def _verify_reuse_compatibility(api, scales, split, cities_data, p8_index, samples_per_city):
    if samples_per_city <= 0:
        return {'enabled': False}
    record = {'block': 'Prediction', 'region_level': 'finegrain'}
    report = {'enabled': True, 'samples_per_city': int(samples_per_city), 'cities': {}}
    with tempfile.TemporaryDirectory(prefix=f'stage2_p8_verify_{split}_') as td:
        td = Path(td)
        for city in SOURCE_CITY_ORDER:
            starts = sorted(p8_index[city].keys())
            if not starts:
                report['cities'][city] = {'checked': 0, 'pixel_equal': 0}
                continue
            take = min(samples_per_city, len(starts))
            pos = np.linspace(0, len(starts) - 1, num=take, dtype=int)
            checked = 0
            equal = 0
            for k, idx in enumerate(pos.tolist()):
                start = int(starts[idx])
                window = api['slice_window'](cities_data[city], start)
                fresh = td / f'{city}_{start:07d}_{k}.png'
                api['render_city'](record, city, window, fresh, scales)
                checked += 1
                if _pixel_equal(p8_index[city][start], fresh):
                    equal += 1
                else:
                    raise RuntimeError(f'Stage-1 P8 reuse compatibility FAILED: current renderer output differs from existing Stage-1 P8 image for {split}/{city}/start={start}. Do not mix old and newly rendered images until this is reviewed.')
            report['cities'][city] = {'checked': checked, 'pixel_equal': equal}
    return report

def _link_or_copy(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'symlink':
        os.symlink(str(src), str(dst))
    elif mode == 'copy':
        shutil.copy2(src, dst)
    else:
        raise ValueError(mode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'valid', 'test', 'all'], default='all')
    ap.add_argument('--max-windows-per-city', type=int, default=None)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--stage1-qa-root', type=Path, default=DEFAULT_STAGE1_QA_ROOT, help='Stage-1 source QA root containing <split>/QA_<split>.jsonl')
    ap.add_argument('--reuse-stage1-p8', choices=['symlink', 'copy', 'none'], default='symlink', help='Reuse exact Stage-1 P8 images when the same city/window exists.')
    ap.add_argument('--verify-reuse-samples-per-city', type=int, default=3, help='Pixel-compare this many reused P8 images per city/split against the current renderer before building.')
    args = ap.parse_args()
    cfg = Stage2Config()
    api = import_qa_design()
    scales = api['load_train_scales'](city_order=list(SOURCE_CITY_ORDER), reference_city_order=None)
    splits = ['train', 'valid', 'test'] if args.split == 'all' else [args.split]
    summary = {}
    record = {'block': 'Prediction', 'region_level': 'finegrain'}
    for split in splits:
        cities = api['load_all_split'](split)
        p8_index = {city: {} for city in SOURCE_CITY_ORDER}
        p8_meta = None
        verify_meta = {'enabled': False}
        if args.reuse_stage1_p8 != 'none':
            p8_index, p8_meta = _build_stage1_p8_index(args.stage1_qa_root, split)
            verify_meta = _verify_reuse_compatibility(api, scales, split, cities, p8_index, args.verify_reuse_samples_per_city)
            print(f'{split}: Stage-1 P8 reuse compatibility PASS')
            print(json.dumps({'p8_index': p8_meta, 'verify': verify_meta}, indent=2))
        summary[split] = {'stage1_p8': p8_meta, 'reuse_verification': verify_meta, 'cities': {}}
        for city in SOURCE_CITY_ORDER:
            starts = api['valid_window_starts'](cities[city])
            if args.max_windows_per_city is not None:
                starts = starts[:args.max_windows_per_city]
            city_dir = cfg.visual_cache_root / split / city
            city_dir.mkdir(parents=True, exist_ok=True)
            rendered = 0
            reused = 0
            skipped = 0
            for j, start in enumerate(starts):
                start = int(start)
                dst = city_dir / f'{start:07d}.png'
                if (dst.exists() or dst.is_symlink()) and (not args.overwrite):
                    skipped += 1
                    continue
                src = p8_index[city].get(start) if args.reuse_stage1_p8 != 'none' else None
                if src is not None and src.exists():
                    _link_or_copy(src, dst, args.reuse_stage1_p8)
                    reused += 1
                else:
                    window = api['slice_window'](cities[city], start)
                    api['render_city'](record, city, window, dst, scales)
                    rendered += 1
                if (j + 1) % 100 == 0:
                    print(f'{split}/{city}: {j + 1}/{len(starts)} (reuse={reused}, render={rendered}, skip={skipped})', flush=True)
            missing = []
            for start in starts:
                p = city_dir / f'{int(start):07d}.png'
                if not p.exists():
                    missing.append(int(start))
            if missing:
                raise RuntimeError(f'{split}/{city}: cache build incomplete, {len(missing)} missing; first={missing[:10]}')
            city_summary = {'requested': int(len(starts)), 'reused_stage1_p8': reused, 'rendered_missing': rendered, 'skipped_existing': skipped, 'verified_present': int(len(starts)), 'directory': str(city_dir)}
            summary[split]['cities'][city] = city_summary
            print(f'{split}/{city}: {city_summary}', flush=True)
    manifest = Path(cfg.cache_root) / 'visual_cache_manifest.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Manifest:', manifest)
if __name__ == '__main__':
    main()
