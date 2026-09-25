import json
import tempfile
from pathlib import Path
import numpy as np
from .config import CITY_ORDER, MAX_VALUE_DIM
from .normalization import FlowNormalizer, encode_value_target

def _stats_file(tmp: Path) -> Path:
    stats = {'source': 'synthetic QA-v3 compatibility test', 'cities': {'NYCTAXI': {'inflow': {'mean': 10.0, 'std': 2.0}, 'outflow': {'mean': 20.0, 'std': 4.0}}, 'BIKECHI': {'inflow': {'mean': 30.0, 'std': 5.0}, 'outflow': {'mean': 40.0, 'std': 8.0}}, 'NYC-BIKE': {'inflow': {'mean': 50.0, 'std': 10.0}, 'outflow': {'mean': 60.0, 'std': 12.0}}}}
    p = tmp / 'stats.json'
    p.write_text(json.dumps(stats), encoding='utf-8')
    return p

def _record(tid, numeric, cities, modality=None):
    return {'sample_id': f'compat_{tid}', 'template_id': tid, 'cities': list(cities), 'modality': modality, 'numeric_target': numeric, 'loss_mask': {'value_mse': 1}}

def _active(encoded):
    target, mask = encoded
    assert tuple(target.shape) == (MAX_VALUE_DIM,)
    assert tuple(mask.shape) == (MAX_VALUE_DIM,)
    return target[mask].numpy()

def main():
    with tempfile.TemporaryDirectory() as td:
        normalizer = FlowNormalizer(str(_stats_file(Path(td))))
        got = _active(encode_value_target(_record('U1', [10, 12, 14, 16, 18, 20, 22, 24], ['NYCTAXI'], 'inflow'), normalizer))
        np.testing.assert_allclose(got, [0, 1, 2, 3, 4, 5, 6, 7])
        got = _active(encode_value_target(_record('U13', [10, 14], ['NYCTAXI'], 'inflow'), normalizer))
        np.testing.assert_allclose(got, [0, 2])
        got = _active(encode_value_target(_record('U14', 6.0, ['NYCTAXI'], 'inflow'), normalizer))
        np.testing.assert_allclose(got, [3.0])
        u4 = {'NYCTAXI': [10.0] * 8, 'BIKECHI': [35.0] * 8, 'NYC-BIKE': [70.0] * 8}
        got = _active(encode_value_target(_record('U4', u4, CITY_ORDER, 'inflow'), normalizer))
        assert got.size == 24
        np.testing.assert_allclose(got[:8], 0.0)
        np.testing.assert_allclose(got[8:16], 1.0)
        np.testing.assert_allclose(got[16:24], 2.0)
        u16 = {'NYCTAXI': [10.0, 12.0], 'BIKECHI': [30.0, 35.0], 'NYC-BIKE': [50.0, 70.0]}
        got = _active(encode_value_target(_record('U16', u16, CITY_ORDER, 'inflow'), normalizer))
        np.testing.assert_allclose(got, [0, 1, 0, 1, 0, 2])
        u17 = {'NYCTAXI': 4.0, 'BIKECHI': 10.0, 'NYC-BIKE': 30.0}
        got = _active(encode_value_target(_record('U17', u17, CITY_ORDER, 'inflow'), normalizer))
        np.testing.assert_allclose(got, [2, 2, 3])
        u11 = {'NYCTAXI': 0.25, 'BIKECHI': 0.5, 'NYC-BIKE': 0.75}
        got = _active(encode_value_target(_record('U11', u11, CITY_ORDER, 'inflow'), normalizer))
        np.testing.assert_allclose(got, [0.25, 0.5, 0.75])
        p3 = {'inflow': [10, 12, 14, 16], 'outflow': [20, 24, 28, 32]}
        got = _active(encode_value_target(_record('P3', p3, ['NYCTAXI']), normalizer))
        np.testing.assert_allclose(got, [0, 1, 2, 3, 0, 1, 2, 3])
        p7 = {}
        for city in CITY_ORDER:
            if city == 'NYCTAXI':
                p7[city] = {'inflow': [10] * 4, 'outflow': [20] * 4}
            elif city == 'BIKECHI':
                p7[city] = {'inflow': [35] * 4, 'outflow': [48] * 4}
            else:
                p7[city] = {'inflow': [70] * 4, 'outflow': [84] * 4}
        got = _active(encode_value_target(_record('P7', p7, CITY_ORDER), normalizer))
        assert got.size == 24
        np.testing.assert_allclose(got[0:8], 0.0)
        np.testing.assert_allclose(got[8:16], 1.0)
        np.testing.assert_allclose(got[16:24], 2.0)
    print('PASS Stage-1 QA-v3 numerical target routing')
    print(f'PASS ValueHead width unchanged: MAX_VALUE_DIM={MAX_VALUE_DIM}')
    print('PASS U14/U17 use difference normalization')
    print('PASS U1/U4/U13/U16 vector schemas fit existing ValueHead')
    print('PASS P3/P7 two-channel horizon targets unchanged')
if __name__ == '__main__':
    main()
