import argparse
import json
from pathlib import Path

def avg(a, b):
    return 0.5 * (float(a) + float(b))

def read_result(path):
    p = Path(path)
    x = json.loads(p.read_text(encoding='utf-8'))
    city = x.get('cities')
    if city is None:
        raise KeyError(f'Cannot find cities metrics in {p}')
    out = {}
    for c in ('NYCTAXI', 'BIKECHI', 'NYC-BIKE'):
        m = city[c]
        out[c] = {'mae': avg(m['inflow']['mae'], m['outflow']['mae']), 'rmse': avg(m['inflow']['rmse'], m['outflow']['rmse'])}
    macro = {'mae': sum((out[c]['mae'] for c in out)) / 3.0, 'rmse': sum((out[c]['rmse'] for c in out)) / 3.0}
    return (out, macro)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', default='./outputs/stage2')
    ap.add_argument('--reference-json', default='./outputs/stage2/urbanbind_stage2_v3_2_2_fullgrad_4gpu_acc3/test_best_epoch23.json')
    ap.add_argument('--save', default=None)
    args = ap.parse_args()
    rows = []
    refs = [('Full UrbanBind', Path(args.reference_json))]
    for name in ('no_grounding', 'no_st_encoder', 'no_cross_city', 'no_dsr', 'no_nash'):
        refs.append((name, Path(args.output_root) / f'urbanbind_ablation_{name}_v322' / 'test_best.json'))
    for label, path in refs:
        if not path.exists():
            rows.append((label, None, path))
            continue
        per_city, macro = read_result(path)
        rows.append((label, (per_city, macro), path))
    print('Ablation test summary (channel-averaged MAE/RMSE)')
    print('=' * 110)
    print(f"{'Variant':<22} {'Taxi MAE':>10} {'Taxi RMSE':>11} {'CHI MAE':>10} {'CHI RMSE':>11} {'NYCB MAE':>10} {'NYCB RMSE':>11} {'Macro MAE':>11}")
    print('-' * 110)
    for label, payload, path in rows:
        if payload is None:
            print(f'{label:<22} MISSING: {path}')
            continue
        pc, macro = payload
        print(f"{label:<22} {pc['NYCTAXI']['mae']:10.3f} {pc['NYCTAXI']['rmse']:11.3f} {pc['BIKECHI']['mae']:10.3f} {pc['BIKECHI']['rmse']:11.3f} {pc['NYC-BIKE']['mae']:10.3f} {pc['NYC-BIKE']['rmse']:11.3f} {macro['mae']:11.3f}")
    print('\nLaTeX rows')
    print('-' * 80)
    labels = {'Full UrbanBind': '\\\\textbf{UrbanBind}', 'no_grounding': 'w/o QA Grounding', 'no_st_encoder': 'w/o Spatial--Temporal Encoder', 'no_cross_city': 'w/o Cross-City Attention', 'no_dsr': 'w/o Dynamic Semantic Re-entry', 'no_nash': 'w/o Nash Balancing'}
    latex_lines = []
    for label, payload, _ in rows:
        if payload is None:
            continue
        pc, _ = payload
        line = f"{labels[label]} & {pc['NYCTAXI']['mae']:.3f} & {pc['NYCTAXI']['rmse']:.3f} & {pc['BIKECHI']['mae']:.3f} & {pc['BIKECHI']['rmse']:.3f} & {pc['NYC-BIKE']['mae']:.3f} & {pc['NYC-BIKE']['rmse']:.3f} \\\\"
        latex_lines.append(line)
        print(line)
    if args.save:
        Path(args.save).write_text('\n'.join(latex_lines) + '\n', encoding='utf-8')
        print(f'Saved LaTeX: {args.save}')
if __name__ == '__main__':
    main()
