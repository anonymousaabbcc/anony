import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    jsonl = Path(args.jsonl).resolve()
    split_dir = jsonl.parent
    image_dir = split_dir / 'images'
    tensor_dir = split_dir / 'tensor_targets'
    used_images = set()
    used_tensors = set()
    with jsonl.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            for p in rec.get('image_paths', []):
                used_images.add(Path(p).resolve())
            tp = rec.get('tensor_target_ref')
            if tp:
                used_tensors.add(Path(tp).resolve())
    existing_images = set((p.resolve() for p in image_dir.glob('*.png'))) if image_dir.exists() else set()
    existing_tensors = set((p.resolve() for p in tensor_dir.glob('*.npz'))) if tensor_dir.exists() else set()
    stale_images = existing_images - used_images
    stale_tensors = existing_tensors - used_tensors
    missing_images = used_images - existing_images
    missing_tensors = used_tensors - existing_tensors
    print('JSONL:', jsonl)
    print()
    print('Images:')
    print('  referenced:', len(used_images))
    print('  existing:  ', len(existing_images))
    print('  stale:     ', len(stale_images))
    print('  missing:   ', len(missing_images))
    print()
    print('Tensors:')
    print('  referenced:', len(used_tensors))
    print('  existing:  ', len(existing_tensors))
    print('  stale:     ', len(stale_tensors))
    print('  missing:   ', len(missing_tensors))
    if missing_images:
        print('\nMissing images:')
        for p in sorted(missing_images)[:20]:
            print(' ', p)
    if missing_tensors:
        print('\nMissing tensors:')
        for p in sorted(missing_tensors)[:20]:
            print(' ', p)
    if args.dry_run:
        print('\nDRY RUN: nothing deleted.')
        return
    if missing_images or missing_tensors:
        raise RuntimeError('Referenced artifacts are missing. Refusing cleanup until this is resolved.')
    for p in stale_images:
        p.unlink()
    for p in stale_tensors:
        p.unlink()
    print()
    print('Deleted stale images:', len(stale_images))
    print('Deleted stale tensors:', len(stale_tensors))
if __name__ == '__main__':
    main()
