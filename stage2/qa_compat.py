import sys
from pathlib import Path

def import_qa_design():
    qa_dir = Path(__file__).resolve().parents[1] / 'QA_design'
    if not qa_dir.exists():
        raise FileNotFoundError(f'Expected frozen QA_design at {qa_dir}. Install stage2 under . so it can reuse the existing split/renderer code.')
    qa_str = str(qa_dir)
    if qa_str not in sys.path:
        sys.path.insert(0, qa_str)
    from split_loader import load_all_split, slice_window, valid_window_starts
    from renderer import load_train_scales, render_city
    from config import SYSTEM_PROMPT
    return {'load_all_split': load_all_split, 'slice_window': slice_window, 'valid_window_starts': valid_window_starts, 'load_train_scales': load_train_scales, 'render_city': render_city, 'system_prompt': SYSTEM_PROMPT}
