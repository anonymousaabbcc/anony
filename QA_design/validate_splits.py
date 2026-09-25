import json
import numpy as np
import pandas as pd
from config import ALL_CITY_ORDER, RAW_CITY_CONFIG, SPLIT_ROOT, WINDOW_LEN
from split_loader import load_split, valid_window_starts

def expected(city):
    total_T = RAW_CITY_CONFIG[city]['expected_shape'][0]
    assert total_T % 24 == 0, f'{city}: total timestamps {total_T} is not divisible by 24'
    total_days = total_T // 24
    train_days = total_days * 70 // 100
    valid_days = total_days * 10 // 100
    test_days = total_days - train_days - valid_days
    return {'train': (train_days, train_days * 24), 'valid': (valid_days, valid_days * 24), 'test': (test_days, test_days * 24)}

def main():
    report = {'cities': {}, 'overall_pass': True}
    for city in ALL_CITY_ORDER:
        report['cities'][city] = {}
        sets = {}
        exp = expected(city)
        for split in ['train', 'valid', 'test']:
            cd = load_split(city, split)
            days, T = exp[split]
            sets[split] = set(cd.time.astype('datetime64[ns]').astype(np.int64).tolist())
            mins = cd.time.astype('datetime64[m]').astype(np.int64)
            dates = pd.DatetimeIndex(cd.time).normalize()
            counts = pd.Series(1, index=dates).groupby(level=0).sum()
            starts = valid_window_starts(cd)
            ew = max(0, T - WINDOW_LEN + 1)
            checks = {'T_matches': cd.T == T, 'shape_ok': tuple(cd.data.shape[2:]) == tuple(RAW_CITY_CONFIG[city]['grid_shape']), 'two_channels': cd.data.shape[1] == 2, 'time_len_matches': len(cd.time) == cd.T, 'finite': bool(np.isfinite(cd.data).all()), 'nonnegative': bool(np.all(cd.data >= 0)), 'unique_timestamps': len(np.unique(cd.time)) == len(cd.time), 'hourly_continuity': bool(len(mins) <= 1 or np.all(np.diff(mins) == 60)), 'complete_24h_days': bool(len(counts) == days and (counts == 24).all()), 'window_count_matches': len(starts) == ew}
            report['cities'][city][split] = {'shape': list(cd.data.shape), 'days': days, 'valid_12h_windows': len(starts), 'checks': checks, 'pass': all(checks.values())}
            report['overall_pass'] &= all(checks.values())
        overlaps = {'train_valid': len(sets['train'] & sets['valid']), 'train_test': len(sets['train'] & sets['test']), 'valid_test': len(sets['valid'] & sets['test'])}
        report['cities'][city]['timestamp_overlap'] = overlaps
        ok = all((v == 0 for v in overlaps.values()))
        report['cities'][city]['no_timestamp_overlap'] = ok
        report['overall_pass'] &= ok
    p = SPLIT_ROOT / 'split_validation_report.json'
    p.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for city in ALL_CITY_ORDER:
        print(f'\\n{city}')
        for split in ['train', 'valid', 'test']:
            r = report['cities'][city][split]
            print(f"  {split:5s}: {('PASS' if r['pass'] else 'FAIL')} shape={tuple(r['shape'])}, windows={r['valid_12h_windows']}")
        print('  overlap:', report['cities'][city]['timestamp_overlap'])
    print('\\nOVERALL:', 'PASS' if report['overall_pass'] else 'FAIL')
    print('Report:', p)
    if not report['overall_pass']:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
