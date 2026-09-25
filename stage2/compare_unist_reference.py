import argparse
import json
from pathlib import Path
UNIST = {'NYCTAXI': {'inflow_mae': 0.17398900696295458, 'inflow_rmse': 1.2242180798039242, 'outflow_mae': 0.1566296925704546, 'outflow_rmse': 1.2894420281187065}, 'BIKECHI': {'inflow_mae': 0.6749753611855739, 'inflow_rmse': 2.604996490475859, 'outflow_mae': 0.6825512711293887, 'outflow_rmse': 2.658306713152698}, 'NYC-BIKE': {'inflow_mae': 2.6401716045621266, 'inflow_rmse': 5.72691728229845, 'outflow_mae': 2.7269342044546128, 'outflow_rmse': 5.999244151306617}, 'MACRO-CITY': {'inflow_mae': 1.163045324236885, 'inflow_rmse': 3.1853772841927444, 'outflow_mae': 1.1887050560514854, 'outflow_rmse': 3.315664297526007}}

def _urbanbind_metrics(obj, city):
    if city == 'MACRO-CITY':
        m = obj['macro_city']
        return {'inflow_mae': float(m['inflow_mae']), 'inflow_rmse': float(m['inflow_rmse']), 'outflow_mae': float(m['outflow_mae']), 'outflow_rmse': float(m['outflow_rmse'])}
    c = obj['cities'][city]
    return {'inflow_mae': float(c['inflow']['mae']), 'inflow_rmse': float(c['inflow']['rmse']), 'outflow_mae': float(c['outflow']['mae']), 'outflow_rmse': float(c['outflow']['rmse'])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urbanbind', required=True, help='UrbanBind final test_metrics.json')
    ap.add_argument('--fail-if-macro-not-beat', action='store_true')
    args = ap.parse_args()
    obj = json.loads(Path(args.urbanbind).read_text(encoding='utf-8'))
    metric_names = ('inflow_mae', 'inflow_rmse', 'outflow_mae', 'outflow_rmse')
    rows = []
    for city in ('NYCTAXI', 'BIKECHI', 'NYC-BIKE', 'MACRO-CITY'):
        got = _urbanbind_metrics(obj, city)
        ref = UNIST[city]
        for metric in metric_names:
            improvement = 100.0 * (ref[metric] - got[metric]) / ref[metric]
            rows.append((city, metric, got[metric], ref[metric], improvement, got[metric] < ref[metric]))
    print('POST-HOC ONLY -- DO NOT USE TEST REFERENCES FOR MODEL SELECTION')
    print('City        Metric          UrbanBind       UniST-ref   Improvement   Beat')
    print('----------------------------------------------------------------------------')
    for city, metric, got, ref, imp, beat in rows:
        print(f"{city:11s} {metric:15s} {got:13.6f} {ref:13.6f} {imp:10.2f}%   {('YES' if beat else 'NO')}")
    macro_rows = [r for r in rows if r[0] == 'MACRO-CITY']
    strict_macro = all((r[-1] for r in macro_rows))
    mean_mae = 0.5 * (_urbanbind_metrics(obj, 'MACRO-CITY')['inflow_mae'] + _urbanbind_metrics(obj, 'MACRO-CITY')['outflow_mae'])
    unist_mean_mae = 0.5 * (UNIST['MACRO-CITY']['inflow_mae'] + UNIST['MACRO-CITY']['outflow_mae'])
    mean_rmse = 0.5 * (_urbanbind_metrics(obj, 'MACRO-CITY')['inflow_rmse'] + _urbanbind_metrics(obj, 'MACRO-CITY')['outflow_rmse'])
    unist_mean_rmse = 0.5 * (UNIST['MACRO-CITY']['inflow_rmse'] + UNIST['MACRO-CITY']['outflow_rmse'])
    print()
    print(f'Macro mean MAE : UrbanBind={mean_mae:.6f}  UniST={unist_mean_mae:.6f}  beat={mean_mae < unist_mean_mae}')
    print(f'Macro mean RMSE: UrbanBind={mean_rmse:.6f}  UniST={unist_mean_rmse:.6f}  beat={mean_rmse < unist_mean_rmse}')
    print(f'STRICT 4-METRIC MACRO PASS: {strict_macro}')
    if args.fail_if_macro_not_beat and (not strict_macro):
        raise SystemExit(2)
if __name__ == '__main__':
    main()
