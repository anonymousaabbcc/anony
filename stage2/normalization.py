import json
from pathlib import Path
import torch
from .config import CHANNELS

class TorchFlowNormalizer:

    def __init__(self, stats_path: str):
        self.stats_path = str(stats_path)
        self.stats = json.loads(Path(stats_path).read_text(encoding='utf-8'))

    def mean_std(self, city: str, channel: str):
        item = self.stats['cities'][city][channel]
        return (float(item['mean']), float(item['std']))

    def normalize(self, city: str, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (4, 5):
            raise ValueError(f'{city}: expected [...,2,H,W], got {tuple(x.shape)}')
        channel_axis = 1 if x.ndim == 4 else 2
        if x.shape[channel_axis] != 2:
            raise ValueError(f'{city}: channel dimension must be 2, got {tuple(x.shape)}')
        y = x.float().clone()
        for ci, channel in enumerate(CHANNELS):
            mean, std = self.mean_std(city, channel)
            if x.ndim == 4:
                y[:, ci] = (y[:, ci] - mean) / std
            else:
                y[:, :, ci] = (y[:, :, ci] - mean) / std
        return y

    def inverse(self, city: str, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (4, 5):
            raise ValueError(f'{city}: expected [...,2,H,W], got {tuple(x.shape)}')
        channel_axis = 1 if x.ndim == 4 else 2
        if x.shape[channel_axis] != 2:
            raise ValueError(f'{city}: channel dimension must be 2, got {tuple(x.shape)}')
        y = x.float().clone()
        for ci, channel in enumerate(CHANNELS):
            mean, std = self.mean_std(city, channel)
            if x.ndim == 4:
                y[:, ci] = y[:, ci] * std + mean
            else:
                y[:, :, ci] = y[:, :, ci] * std + mean
        return y
