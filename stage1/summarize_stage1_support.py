import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='./outputs/stage1/support_evidence')
    args = ap.parse_args()
    root = Path(args.output_dir)
    A = json.loads((root / 'A_base_vs_stage1.json').read_text())
    B = json.loads((root / 'B_image_grounding.json').read_text())
    C = json.loads((root / 'C_sqa_linear_probe.json').read_text())
    a = A['comparison']['overall']
    bc = B['conditions']['correct']['macro_city']
    lines = ['STAGE-1 SUPPORT EVIDENCE SUMMARY', '================================', '', 'A. Base VLM vs Stage-1 VLM', f"Overall held-out QA token CE: {a['base_token_ce']:.6f} -> {a['stage1_token_ce']:.6f}", f"Relative CE reduction: {a['ce_reduction_pct']:.2f}%", f"Perplexity: {a['base_perplexity']:.3f} -> {a['stage1_perplexity']:.3f}", '', 'B. P8 visual grounding', f"Correct-image macro MAE/RMSE: {bc['mae']:.6f}/{bc['rmse']:.6f}"]
    for cond in ['shuffled_time_same_city', 'permuted_city_images', 'blank_no_information']:
        r = B['conditions'][cond]['macro_city']
        lines.append(f"{cond}: MAE/RMSE={r['mae']:.6f}/{r['rmse']:.6f}; MAE increase={r['mae_increase_vs_correct_pct']:.2f}%")
    c = C['comparison']
    bprobe = C['base_probe']['test']
    sprobe = C['stage1_probe']['test']
    lines += ['', 'C. P8 S_QA linear probe', f"Base VLM probe MAE/RMSE/R2: {bprobe['mae']:.6f}/{bprobe['rmse']:.6f}/{bprobe['r2']:.6f}", f"Stage-1 probe MAE/RMSE/R2: {sprobe['mae']:.6f}/{sprobe['rmse']:.6f}/{sprobe['r2']:.6f}", f"Probe MAE reduction: {c['mae_reduction_pct']:.2f}%", f"Probe RMSE reduction: {c['rmse_reduction_pct']:.2f}%", f"Probe R2 gain: {c['r2_gain']:.6f}", '', 'Interpretation rule:', 'A supports Stage-1 urban QA adaptation; B supports dependence on visual evidence;', 'C supports that the exact multi-city P8 S_QA used by Stage 2 contains stronger forecast-relevant information.', 'Do not describe A/B/C as proof of Stage-2 latent alignment; Stage-2 alignment must be evaluated separately.']
    text = '\n'.join(lines) + '\n'
    path = root / 'stage1_support_summary.txt'
    path.write_text(text, encoding='utf-8')
    print(text)
    print('Saved:', path)
if __name__ == '__main__':
    main()
