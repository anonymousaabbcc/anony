import csv
import json
from pathlib import Path
from .config import EXPECTED_WINDOWS, SOURCE_CITY_ORDER, Stage2Config
from .qa_compat import import_qa_design

def main():
    cfg = Stage2Config()
    api = import_qa_design()
    out_root = Path(cfg.cache_root) / 'window_manifests'
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ['train', 'valid', 'test']:
        cities = api['load_all_split'](split)
        summary[split] = {}
        for city in SOURCE_CITY_ORDER:
            starts = api['valid_window_starts'](cities[city])
            if len(starts) != EXPECTED_WINDOWS[split][city]:
                raise RuntimeError(f'{split}/{city}: unexpected window count {len(starts)}')
            path = out_root / f"{split}_{city.replace('-', '_')}.csv"
            with path.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['window_index', 'start_index', 'history_start', 'history_end', 'forecast_start', 'forecast_end'])
                for wi, start in enumerate(starts):
                    win = api['slice_window'](cities[city], int(start))
                    w.writerow([wi, int(start), str(win['time_hist'][0]), str(win['time_hist'][-1]), str(win['time_future'][0]), str(win['time_future'][-1])])
            summary[split][city] = {'count': int(len(starts)), 'path': str(path)}
    (out_root / 'manifest_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Frozen Stage-2 raw-window manifests:', out_root)
if __name__ == '__main__':
    main()
