import argparse
import importlib
import json
import os
from pathlib import Path
import torch
from .config import EXPECTED_WINDOWS, SOURCE_CITY_ORDER, Stage2Config
from .normalization import TorchFlowNormalizer
from .qa_compat import import_qa_design

def _writable(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    p = path / '.stage2_write_test'
    p.write_text('ok', encoding='utf-8')
    p.unlink()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-vlm-load', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--check-visuals', choices=['none', 'train', 'valid', 'test', 'all'], default='none')
    args = ap.parse_args()
    cfg = Stage2Config()
    lines = []
    lines.append('STAGE-2 PREFLIGHT')
    lines.append('=================')
    lines.append(f'Stage-1 checkpoint: {cfg.stage1_checkpoint}')
    for name in ['stage1_config.json', 'stage1_auxiliary.pt', 'lora_adapter']:
        p = Path(cfg.stage1_checkpoint) / name
        if not p.exists():
            raise FileNotFoundError(p)
    lines.append('Stage-1 checkpoint files: PASS')
    for path in [cfg.output_root, cfg.checkpoint_root, cfg.cache_root, cfg.log_root]:
        _writable(path)
        lines.append(f'Writable: {path}')
    import cvxpy as cp
    import ecos
    lines.append(f'cvxpy={cp.__version__}; ecos={ecos.__version__}; solvers={cp.installed_solvers()}')
    if cp.__version__ != '1.5.4' or ecos.__version__ != '2.0.14' or 'ECOS' not in cp.installed_solvers():
        raise RuntimeError('Nash dependencies do not match the validated environment.')
    x = cp.Variable(3, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 1.0])
    prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f'ECOS solve failed: {prob.status}')
    lines.append('Nash dependencies + ECOS solve: PASS')
    api = import_qa_design()
    for split in ['train', 'valid', 'test']:
        cities = api['load_all_split'](split)
        counts = {}
        for city in SOURCE_CITY_ORDER:
            n = len(api['valid_window_starts'](cities[city]))
            counts[city] = n
            if n != EXPECTED_WINDOWS[split][city]:
                raise RuntimeError(f'{split}/{city}: {n} != {EXPECTED_WINDOWS[split][city]}')
        lines.append(f'{split} windows: {counts}')
    lines.append('Frozen chronological 8->4 window protocol: PASS')
    TorchFlowNormalizer(cfg.normalization_stats)
    lines.append(f'Stage-1 normalization reused: {cfg.normalization_stats}')
    if args.check_visuals != 'none':
        splits = ['train', 'valid', 'test'] if args.check_visuals == 'all' else [args.check_visuals]
        for split in splits:
            cities = api['load_all_split'](split)
            for city in SOURCE_CITY_ORDER:
                starts = api['valid_window_starts'](cities[city])
                missing = [str(cfg.visual_cache_root / split / city / f'{int(s):07d}.png') for s in starts if not (cfg.visual_cache_root / split / city / f'{int(s):07d}.png').exists()]
                if missing:
                    raise FileNotFoundError(f'{split}/{city}: missing {len(missing)} visual cache files; first={missing[0]}')
            lines.append(f'Visual cache {split}: PASS')
    if not args.skip_vlm_load:
        from .model import Stage2UrbanForecaster
        device = torch.device(args.device)
        model = Stage2UrbanForecaster(cfg, device)
        if model.bridge.trainable_parameter_count != 0:
            raise RuntimeError('Frozen Stage-1 unexpectedly has trainable parameters.')
        lines.append(f'Stage-1 frozen trainable params: {model.bridge.trainable_parameter_count}')
        lines.append(f'VLM hidden size: {model.bridge.hidden_size}')
        groups = model.trainable_parameter_groups()
        for name, params in groups.items():
            lines.append(f'Trainable {name}: {sum((p.numel() for p in params)):,} params')
        lines.append(f'Nash shared/reference params: {sum((p.numel() for p in model.bargaining_parameters())):,}')
        lines.append('Frozen Stage-1 bridge load: PASS')
    note = '\n'.join(lines) + '\n'
    print(note)
    note_path = Path(cfg.log_root) / 'stage2_preflight_note.txt'
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note, encoding='utf-8')
    print('Saved:', note_path)
if __name__ == '__main__':
    main()
