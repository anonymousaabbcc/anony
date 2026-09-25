import argparse
import json
from .normalization import build_stats_from_full_train_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='./outputs/stage1_qwen25vl3b_chi_v3_finegrain/normalization_stats.json')
    args = ap.parse_args()
    stats = build_stats_from_full_train_split(args.output)
    print(json.dumps(stats, indent=2))
    print('Saved:', args.output)
if __name__ == '__main__':
    main()
