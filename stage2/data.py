import math
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import EXPECTED_WINDOWS, SOURCE_CITY_ORDER
from .normalization import TorchFlowNormalizer
from .qa_compat import import_qa_design

class JointMultiCityDataset(Dataset):

    def __init__(self, split: str, normalizer: TorchFlowNormalizer, visual_cache_root: str, seed: int=42, training: bool=False, max_joint_steps: int | None=None, debug_sequential: bool=False):
        if split not in EXPECTED_WINDOWS:
            raise ValueError(split)
        api = import_qa_design()
        cities = api['load_all_split'](split)
        self.slice_window = api['slice_window']
        self.city_data = cities
        self.starts: Dict[str, np.ndarray] = {}
        for city in SOURCE_CITY_ORDER:
            starts = api['valid_window_starts'](cities[city])
            expected = EXPECTED_WINDOWS[split][city]
            if len(starts) != expected:
                raise RuntimeError(f'{split}/{city}: found {len(starts)} windows, expected {expected}.')
            self.starts[city] = starts
        self.split = split
        self.normalizer = normalizer
        self.visual_cache_root = Path(visual_cache_root)
        self.seed = int(seed)
        self.training = bool(training)
        self.debug_sequential = bool(debug_sequential)
        self.full_length = max((len(self.starts[c]) for c in SOURCE_CITY_ORDER))
        self.length = min(self.full_length, int(max_joint_steps)) if max_joint_steps else self.full_length
        self.epoch = 0
        self._maps = None
        self.set_epoch(0)

    def __len__(self):
        return self.length

    def _train_map_for_city(self, city: str, epoch: int):
        n = len(self.starts[city])
        needed = self.full_length
        pieces = []
        cycle = 0
        while sum((len(x) for x in pieces)) < needed:
            rng = np.random.default_rng(self.seed + 10007 * epoch + 997 * cycle + 53 * SOURCE_CITY_ORDER.index(city))
            pieces.append(rng.permutation(n))
            cycle += 1
        return np.concatenate(pieces)[:needed]

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if self.training and (not self.debug_sequential):
            self._maps = {city: self._train_map_for_city(city, self.epoch) for city in SOURCE_CITY_ORDER}
        else:
            self._maps = {city: np.arange(self.full_length, dtype=np.int64) % len(self.starts[city]) for city in SOURCE_CITY_ORDER}

    def _image_path(self, city: str, start: int):
        return self.visual_cache_root / self.split / city / f'{int(start):07d}.png'

    def __getitem__(self, joint_index):
        out = {'joint_index': int(joint_index), 'cities': {}}
        for city in SOURCE_CITY_ORDER:
            local_window_index = int(self._maps[city][joint_index])
            start = int(self.starts[city][local_window_index])
            window = self.slice_window(self.city_data[city], start)
            x = torch.from_numpy(np.asarray(window['X_hist'], dtype=np.float32))
            y = torch.from_numpy(np.asarray(window['Y_future'], dtype=np.float32))
            x = self.normalizer.normalize(city, x)
            y = self.normalizer.normalize(city, y)
            image_path = self._image_path(city, start)
            if not image_path.exists():
                raise FileNotFoundError(f'Missing Stage-2 visual cache: {image_path}. Run `python -m stage2.build_visual_cache --split ...` first.')
            out['cities'][city] = {'x': x, 'y': y, 'time_hist': np.asarray(window['time_hist']).astype('datetime64[h]').astype(str).tolist(), 'time_future': np.asarray(window['time_future']).astype('datetime64[h]').astype(str).tolist(), 'image_path': str(image_path), 'window_index': local_window_index, 'start_index': start, 'active_once': bool(joint_index < len(self.starts[city]))}
        return out
