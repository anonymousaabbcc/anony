import argparse
import json
from pathlib import Path
CHANNEL_NAMES = ('inflow', 'outflow')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='./outputs/stage1/support_evidence')
    args = ap.parse_args()
    root = Path(args.output_dir)
    b_path = root / 'B_image_grounding_by_channel.json'
    c_path = root / 'C_sqa_linear_probe_by_channel.json'
    if not b_path.exists():
        raise FileNotFoundError(b_path)
    if not c_path.exists():
        raise FileNotFoundError(c_path)
    b = json.loads(b_path.read_text(encoding='utf-8'))
    c = json.loads(c_path.read_text(encoding='utf-8'))
    lines = ['STAGE-1 CHANNEL-WISE SUPPORT EVIDENCE', '=====================================', '', 'A. QA adaptation', 'Not decomposed by inflow/outflow because A is assistant-token CE/PPL over heterogeneous QA templates.', '', 'B. P8 visual grounding -- macro-city raw MAE/RMSE by channel']
    for cond in ['correct', 'shuffled_time_same_city', 'permuted_city_images', 'blank_no_information']:
        lines.append(cond)
        for ch in CHANNEL_NAMES:
            r = b['conditions'][cond]['macro_city_by_channel'][ch]
            nmse = b['conditions'][cond]['normalized_tensor_mse_by_channel'][ch]
            extra = ''
            if cond != 'correct':
                extra = f"; MAE increase={r['mae_increase_vs_correct_pct']:.2f}%"
            lines.append(f"  {ch}: MAE/RMSE={r['mae']:.6f}/{r['rmse']:.6f}; normalized Tensor MSE={nmse:.6f}{extra}")
    lines.extend(['', 'C. P8 S_QA linear probe -- same fitted 24-D probe, decomposed by channel'])
    for rep_key, label in [('base_probe', 'Base VLM'), ('stage1_probe', 'Stage-1 VLM')]:
        lines.append(label)
        for ch in CHANNEL_NAMES:
            r = c[rep_key]['test_by_channel'][ch]
            lines.append(f"  {ch}: MAE/RMSE/R2={r['mae']:.6f}/{r['rmse']:.6f}/{r['r2']:.6f}")
    lines.append('')
    lines.append('Stage-1 improvement vs Base')
    for ch in CHANNEL_NAMES:
        r = c['comparison']['by_channel'][ch]
        lines.append(f"  {ch}: MAE reduction={r['mae_reduction_pct']:.2f}%; RMSE reduction={r['rmse_reduction_pct']:.2f}%; R2 gain={r['r2_gain']:.6f}")
    out = root / 'stage1_channel_support_summary.txt'
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'\nSaved: {out}')
if __name__ == '__main__':
    main()
