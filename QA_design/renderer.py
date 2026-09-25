import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from config import CHANNEL_TO_INDEX, CITY_SLUG, SPLIT_ROOT, RAW_CITY_CONFIG, SOURCE_CITY_ORDER
from regions import get_region_slices, region_value, parse_finegrain_id

def _load_city_train_stats(city):
    slug = CITY_SLUG[city]
    path = SPLIT_ROOT / slug / f'{slug}_train_stats.json'
    return json.loads(path.read_text(encoding='utf-8'))

def load_train_scales(city_order=None, reference_city_order=None):
    order = list(SOURCE_CITY_ORDER if city_order is None else city_order)
    if reference_city_order is None:
        return {city: _load_city_train_stats(city) for city in order}
    refs = [_load_city_train_stats(city) for city in reference_city_order]
    envelope = {}
    for channel in ['inflow', 'outflow']:
        envelope[channel] = {'p01': min((float(x[channel]['p01']) for x in refs)), 'p99': max((float(x[channel]['p99']) for x in refs)), 'min': min((float(x[channel]['min']) for x in refs)), 'max': max((float(x[channel]['max']) for x in refs))}
    return {city: envelope for city in order}

def _get_scale(scales, city, channel):
    stats = scales[city][channel]
    vmin, vmax = (float(stats['p01']), float(stats['p99']))
    if vmax <= vmin:
        vmin, vmax = (float(stats['min']), float(stats['max']))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return (vmin, vmax)

def _clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])

def _show_map(ax, map_2d, vmin, vmax):
    ax.imshow(map_2d, origin='upper', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    _clean_axis(ax)

def draw_region_layout(ax, city, level):
    if level not in {'coarse', 'local'}:
        raise ValueError(level)
    H, W = RAW_CITY_CONFIG[city]['grid_shape']
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('auto')
    for rid, (rs, cs) in get_region_slices(city, level).items():
        rect = Rectangle((cs.start - 0.5, rs.start - 0.5), cs.stop - cs.start, rs.stop - rs.start, fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        cx = (cs.start + cs.stop - 1) / 2.0
        cy = (rs.start + rs.stop - 1) / 2.0
        ax.text(cx, cy, rid, ha='center', va='center', fontsize=8 if level == 'local' else 10, fontweight='bold', bbox={'facecolor': 'white', 'alpha': 0.9, 'edgecolor': 'none', 'pad': 1.0})
    ax.set_title('REGION LAYOUT', fontsize=11, fontweight='bold')
    _clean_axis(ax)

def draw_region_values(ax, map_2d, city, level):
    for rid, (rs, cs) in get_region_slices(city, level).items():
        rect = Rectangle((cs.start - 0.5, rs.start - 0.5), cs.stop - cs.start, rs.stop - rs.start, fill=False, edgecolor='black', linewidth=1.0)
        ax.add_patch(rect)
        cx = (cs.start + cs.stop - 1) / 2.0
        cy = (rs.start + rs.stop - 1) / 2.0
        value = region_value(map_2d, city, level, rid)
        ax.text(cx, cy, f'{value:.1f}', ha='center', va='center', fontsize=8 if level == 'local' else 10, fontweight='bold', bbox={'facecolor': 'white', 'alpha': 0.78, 'edgecolor': 'none', 'pad': 0.8})

def _draw_finegrain_grid(ax, city):
    H, W = RAW_CITY_CONFIG[city]['grid_shape']
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which='minor', linewidth=0.25, alpha=0.55)
    ax.tick_params(which='minor', bottom=False, left=False)
    xt = np.arange(W)
    yt = np.arange(H)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([str(i + 1) for i in xt], fontsize=5)
    ax.set_yticklabels([str(i + 1) for i in yt], fontsize=5)
    ax.set_xlabel('column', fontsize=6)
    ax.set_ylabel('row', fontsize=6)

def _selected_finegrain_ids(record, city):
    out = []
    by_city = record.get('region_ids_by_city') or {}
    if city in by_city:
        out.append(by_city[city])
    for key in ['region_id_a_by_city', 'region_id_b_by_city']:
        d = record.get(key) or {}
        if city in d:
            out.append(d[city])
    if not out:
        for key in ['region_id', 'region_id_a', 'region_id_b']:
            rid = record.get(key)
            if isinstance(rid, str) and rid.startswith('F('):
                out.append(rid)
    return list(dict.fromkeys(out))

def _highlight_finegrain(ax, map_2d, city, rids, annotate=True):
    for rid in rids:
        r, c = parse_finegrain_id(rid)
        rect = Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=2.0)
        ax.add_patch(rect)
        if annotate:
            ax.text(c, r, f'{float(map_2d[r, c]):.1f}', ha='center', va='center', fontsize=6, fontweight='bold', bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none', 'pad': 0.4})

def _annotate_all_finegrain(ax, map_2d):
    H, W = map_2d.shape
    fs = 4.2 if H * W <= 200 else 3.4
    for r in range(H):
        for c in range(W):
            ax.text(c, r, f'{float(map_2d[r, c]):.1f}', ha='center', va='center', fontsize=fs, bbox={'facecolor': 'white', 'alpha': 0.52, 'edgecolor': 'none', 'pad': 0.15})

def _render_understanding_partitioned(record, city, window, output_path, scales):
    level, channel_name = (record['region_level'], record['modality'])
    channel = CHANNEL_TO_INDEX[channel_name]
    vmin, vmax = _get_scale(scales, city, channel_name)
    fig = plt.figure(figsize=(19, 7.5), layout='constrained')
    gs = fig.add_gridspec(2, 5, width_ratios=[1.55, 1, 1, 1, 1], wspace=0.08, hspace=0.18)
    draw_region_layout(fig.add_subplot(gs[:, 0]), city, level)
    for t in range(8):
        ax = fig.add_subplot(gs[t // 4, 1 + t % 4])
        m = window['X_hist'][t, channel]
        _show_map(ax, m, vmin, vmax)
        draw_region_values(ax, m, city, level)
        ax.set_title(f't={t + 1}', fontsize=10, fontweight='bold')
    fig.suptitle(f'{city} | {channel_name} | hourly history t=1..8 | level={level}', fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def _render_understanding_whole(record, city, window, output_path, scales):
    channel_name = record['modality']
    channel = CHANNEL_TO_INDEX[channel_name]
    vmin, vmax = _get_scale(scales, city, channel_name)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    for t in range(8):
        ax = axes[t // 4, t % 4]
        m = window['X_hist'][t, channel]
        _show_map(ax, m, vmin, vmax)
        value = region_value(m, city, 'whole', 'whole_city')
        ax.set_title(f't={t + 1}\nwhole_city avg = {value:.1f}', fontsize=10)
    fig.suptitle(f'{city} | {channel_name} | hourly history t=1..8 | level=whole', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def _render_native_value_detail(ax, map_2d, city, vmin, vmax, time_step):
    _show_map(ax, map_2d, vmin, vmax)
    _draw_finegrain_grid(ax, city)
    H, W = map_2d.shape
    fs = 6.8 if H * W <= 200 else 5.7
    for r in range(H):
        for c in range(W):
            ax.text(c, r, f'{float(map_2d[r, c]):.1f}', ha='center', va='center', fontsize=fs, fontweight='bold', bbox={'facecolor': 'white', 'alpha': 0.64, 'edgecolor': 'none', 'pad': 0.12})
    ax.set_title(f'QUERY-TIME NATIVE VALUES | t={time_step} | exact historical values', fontsize=11, fontweight='bold')

def _render_understanding_finegrain(record, city, window, output_path, scales):
    channel_name = record['modality']
    channel = CHANNEL_TO_INDEX[channel_name]
    vmin, vmax = _get_scale(scales, city, channel_name)
    selected = _selected_finegrain_ids(record, city)
    needs_full_values = record['template_id'] in {'U9', 'U11', 'U12'}
    if needs_full_values:
        fig = plt.figure(figsize=(19, 13), layout='constrained')
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.55])
        axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]
        detail_ax = fig.add_subplot(gs[2, :])
    else:
        fig, axarr = plt.subplots(2, 4, figsize=(19, 8.5), squeeze=False, layout='constrained')
        axes = axarr
        detail_ax = None
    for t in range(8):
        ax = axes[t // 4][t % 4]
        m = window['X_hist'][t, channel]
        _show_map(ax, m, vmin, vmax)
        _draw_finegrain_grid(ax, city)
        if selected:
            _highlight_finegrain(ax, m, city, selected, annotate=True)
        ax.set_title(f't={t + 1}', fontsize=9, fontweight='bold')
    if needs_full_values:
        q = int(record['time_step']) - 1
        _render_native_value_detail(detail_ax, window['X_hist'][q, channel], city, vmin, vmax, q + 1)
    fig.suptitle(f'{city} | {channel_name} | native-grid history t=1..8 | finegrain F(row,column)', fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=210, bbox_inches='tight')
    plt.close(fig)

def _render_prediction_partitioned(record, city, window, output_path, scales):
    level = record['region_level']
    fig = plt.figure(figsize=(19, 12.5), layout='constrained')
    gs = fig.add_gridspec(4, 5, width_ratios=[1.55, 1, 1, 1, 1], wspace=0.08, hspace=0.2)
    draw_region_layout(fig.add_subplot(gs[:, 0]), city, level)
    for ci, cname in enumerate(['inflow', 'outflow']):
        ch = CHANNEL_TO_INDEX[cname]
        vmin, vmax = _get_scale(scales, city, cname)
        for t in range(8):
            ax = fig.add_subplot(gs[ci * 2 + t // 4, 1 + t % 4])
            m = window['X_hist'][t, ch]
            _show_map(ax, m, vmin, vmax)
            draw_region_values(ax, m, city, level)
            ax.set_title(f'{cname} | t={t + 1}', fontsize=10, fontweight='bold')
    fig.suptitle(f'{city} | hourly history t=1..8 | level={level}', fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def _render_prediction_nonpartitioned(record, city, window, output_path, scales):
    level = record['region_level']
    fig, axes = plt.subplots(4, 4, figsize=(18, 13), squeeze=False, layout='constrained')
    selected = _selected_finegrain_ids(record, city) if level == 'finegrain' else []
    for ci, cname in enumerate(['inflow', 'outflow']):
        ch = CHANNEL_TO_INDEX[cname]
        vmin, vmax = _get_scale(scales, city, cname)
        for t in range(8):
            ax = axes[ci * 2 + t // 4, t % 4]
            m = window['X_hist'][t, ch]
            _show_map(ax, m, vmin, vmax)
            if level == 'whole':
                value = region_value(m, city, 'whole', 'whole_city')
                title = f'{cname} | t={t + 1}\nwhole_city avg = {value:.1f}'
            else:
                _draw_finegrain_grid(ax, city)
                if selected:
                    _highlight_finegrain(ax, m, city, selected, annotate=True)
                title = f'{cname} | t={t + 1}'
            ax.set_title(title, fontsize=9)
    fig.suptitle(f'{city} | hourly history t=1..8 | level={level}', fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220 if level == 'finegrain' else 200, bbox_inches='tight')
    plt.close(fig)

def render_city(record, city, window, output_path, scales):
    level, block = (record['region_level'], record['block'])
    if block == 'Understanding':
        if level == 'whole':
            _render_understanding_whole(record, city, window, output_path, scales)
        elif level in {'coarse', 'local'}:
            _render_understanding_partitioned(record, city, window, output_path, scales)
        elif level == 'finegrain':
            _render_understanding_finegrain(record, city, window, output_path, scales)
        else:
            raise ValueError(level)
    elif level in {'coarse', 'local'}:
        _render_prediction_partitioned(record, city, window, output_path, scales)
    elif level in {'whole', 'finegrain'}:
        _render_prediction_nonpartitioned(record, city, window, output_path, scales)
    else:
        raise ValueError(level)

def render_sample_images(sample, image_dir, scales):
    paths = []
    for city, window in sample.windows.items():
        path = (image_dir / f"{sample.record['sample_id']}_{CITY_SLUG[city]}.png").resolve()
        render_city(sample.record, city, window, path, scales)
        paths.append(path)
    return paths
