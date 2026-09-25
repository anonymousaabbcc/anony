import numpy as np
from config import RAW_CITY_CONFIG
from split_loader import SplitCityData
from sampling import FinegrainSampler
from generator import QAGenerator
from templates import TEMPLATES
from targets import normalized_peak_share, tied_hotspot_centroid, hotspot_movement
from regions import dominant_region

def make_city(city, rng, T=40):
    H, W = RAW_CITY_CONFIG[city]['grid_shape']
    x = rng.poisson(1.4, size=(T, 2, H, W)).astype(np.float32)
    x[rng.random(x.shape) < 0.55] = 0
    x[:, :, 0, 0] += 1
    time = np.arange(T).astype('timedelta64[h]') + np.datetime64('2020-01-01T00', 'h')
    return SplitCityData(city, 'train', x, time)

def main():
    rng = np.random.default_rng(7)
    order = ['NYCTAXI', 'BIKECHI', 'NYC-BIKE']
    cities = {city: make_city(city, rng) for city in order}
    sampler = FinegrainSampler(cities, cities, seed=99)
    gen = QAGenerator(cities, seed=42, city_order=order, finegrain_sampler=sampler)
    n = 0
    for tid, spec in TEMPLATES.items():
        for level in spec.region_levels:
            sample = gen.generate(tid, n, forced_city='NYCTAXI' if spec.city_scope == 'single_city' else None, forced_level=level)
            r = sample.record
            assert r['question'] and r['answer_text']
            assert '{' not in r['question'] and '}' not in r['question']
            if tid in {'P1', 'P3', 'P5', 'P7'} and level == 'finegrain':
                assert all((m['basis'] == 'history_only_t1_t8' for m in r['sampling_meta'].values()))
            n += 1
    assert abs(normalized_peak_share([1, 1, 1, 1]) - 0.0) < 1e-12
    assert abs(normalized_peak_share([4, 0, 0, 0]) - 1.0) < 1e-12
    assert abs(normalized_peak_share([0, 0, 0, 0]) - 0.0) < 1e-12
    m = np.zeros((10, 20), dtype=np.float64)
    m[2, 3] = 5
    m[2, 4] = 5
    assert dominant_region(m, 'NYCTAXI', 'finegrain') == 'F(3,4)'
    center = tied_hotspot_centroid(m, 'NYCTAXI', 'finegrain')
    assert np.allclose(center, [2.5 / 10, 4.0 / 20])
    label, score, _ = hotspot_movement(np.stack([m, m]), 'NYCTAXI', 'finegrain')
    assert label == 'stationary' and score == 0.0
    H, W = RAW_CITY_CONFIG['NYCTAXI']['grid_shape']
    sparse = np.zeros((40, 2, H, W), dtype=np.float32)
    srng = np.random.default_rng(123)
    for ch in range(2):
        mask = srng.random((40, H, W)) < 0.035
        vals = srng.poisson(3.0, size=(40, H, W)).astype(np.float32) + 1
        sparse[:, ch][mask] = vals[mask]
    stime = np.arange(40).astype('timedelta64[h]') + np.datetime64('2021-01-01T00', 'h')
    scity = SplitCityData('NYCTAXI', 'train', sparse, stime)
    ss = FinegrainSampler({'NYCTAXI': scity}, {'NYCTAXI': scity}, seed=8)
    assert ss.reference['NYCTAXI', 'inflow'].p0 > 0.25
    assert ss.reference['NYCTAXI', 'inflow'].q0 == 0.25
    print(f'PASS: generated all {n} template-level combinations')
    print('PASS: normalized peak-share endpoints')
    print('PASS: row-major dominant tie rule')
    print('PASS: tied-hotspot centroid movement')
    print('PASS: prediction finegrain sampling marked history-only')
    print('PASS: natural zero-history probability is capped at 25% without deleting zeros')
if __name__ == '__main__':
    main()
