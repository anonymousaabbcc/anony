import json
import math
from pathlib import Path
import torch
import torch.distributed as dist
from .config import CHANNELS, FORECAST_HORIZONS, SOURCE_CITY_ORDER

def _empty_stat():
    return {'abs': 0.0, 'sq': 0.0, 'n': 0}

class RawFlowMetrics:

    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.stats = {city: {ch: _empty_stat() for ch in CHANNELS} for city in SOURCE_CITY_ORDER}
        self.by_horizon = {city: {int(h): {ch: _empty_stat() for ch in CHANNELS} for h in FORECAST_HORIZONS} for city in SOURCE_CITY_ORDER}

    @staticmethod
    def _accumulate(stat, err):
        stat['abs'] += float(err.abs().sum().cpu())
        stat['sq'] += float((err * err).sum().cpu())
        stat['n'] += int(err.numel())

    @torch.no_grad()
    def update(self, city, pred_norm, target_norm, active_mask=None):
        if pred_norm.ndim != 5 or target_norm.shape != pred_norm.shape:
            raise ValueError(f'Expected matching [B,F,2,H,W], got pred={tuple(pred_norm.shape)}, target={tuple(target_norm.shape)}')
        if pred_norm.shape[1] != len(FORECAST_HORIZONS):
            raise ValueError(f'Metric forecast length {pred_norm.shape[1]} != {len(FORECAST_HORIZONS)}')
        if active_mask is not None:
            active_mask = active_mask.to(pred_norm.device)
            pred_norm = pred_norm[active_mask]
            target_norm = target_norm[active_mask]
        if pred_norm.numel() == 0:
            return
        for ci, ch in enumerate(CHANNELS):
            mean, std = self.normalizer.mean_std(city, ch)
            pred = pred_norm[:, :, ci].float() * std + mean
            target = target_norm[:, :, ci].float() * std + mean
            err = pred - target
            self._accumulate(self.stats[city][ch], err)
            for hi, horizon in enumerate(FORECAST_HORIZONS):
                self._accumulate(self.by_horizon[city][horizon][ch], err[:, hi])

    def synchronize(self, device):
        if not (dist.is_available() and dist.is_initialized()):
            return
        for city in SOURCE_CITY_ORDER:
            for ch in CHANNELS:
                self._sync_one(self.stats[city][ch], device)
            for horizon in FORECAST_HORIZONS:
                for ch in CHANNELS:
                    self._sync_one(self.by_horizon[city][horizon][ch], device)

    @staticmethod
    def _sync_one(stat, device):
        x = torch.tensor([stat['abs'], stat['sq'], float(stat['n'])], dtype=torch.float64, device=device)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        stat['abs'] = float(x[0].cpu())
        stat['sq'] = float(x[1].cpu())
        stat['n'] = int(round(float(x[2].cpu())))

    @staticmethod
    def _finish(stat):
        if stat['n'] <= 0:
            raise RuntimeError('Metric stat has no elements')
        return {'mae': stat['abs'] / stat['n'], 'rmse': math.sqrt(stat['sq'] / stat['n']), 'n': stat['n']}

    def compute(self):
        out = {'cities': {}}
        city_in_mae, city_out_mae = ([], [])
        city_in_rmse, city_out_rmse = ([], [])
        for city in SOURCE_CITY_ORDER:
            out['cities'][city] = {}
            for ch in CHANNELS:
                out['cities'][city][ch] = self._finish(self.stats[city][ch])
            out['cities'][city]['by_horizon'] = {}
            for horizon in FORECAST_HORIZONS:
                out['cities'][city]['by_horizon'][f'h{horizon}'] = {ch: self._finish(self.by_horizon[city][horizon][ch]) for ch in CHANNELS}
            city_in_mae.append(out['cities'][city]['inflow']['mae'])
            city_out_mae.append(out['cities'][city]['outflow']['mae'])
            city_in_rmse.append(out['cities'][city]['inflow']['rmse'])
            city_out_rmse.append(out['cities'][city]['outflow']['rmse'])
        out['macro_city'] = {'inflow_mae': sum(city_in_mae) / len(city_in_mae), 'inflow_rmse': sum(city_in_rmse) / len(city_in_rmse), 'outflow_mae': sum(city_out_mae) / len(city_out_mae), 'outflow_rmse': sum(city_out_rmse) / len(city_out_rmse)}
        out['macro_city_by_horizon'] = {}
        for horizon in FORECAST_HORIZONS:
            key = f'h{horizon}'
            out['macro_city_by_horizon'][key] = {'inflow_mae': sum((out['cities'][c]['by_horizon'][key]['inflow']['mae'] for c in SOURCE_CITY_ORDER)) / len(SOURCE_CITY_ORDER), 'inflow_rmse': sum((out['cities'][c]['by_horizon'][key]['inflow']['rmse'] for c in SOURCE_CITY_ORDER)) / len(SOURCE_CITY_ORDER), 'outflow_mae': sum((out['cities'][c]['by_horizon'][key]['outflow']['mae'] for c in SOURCE_CITY_ORDER)) / len(SOURCE_CITY_ORDER), 'outflow_rmse': sum((out['cities'][c]['by_horizon'][key]['outflow']['rmse'] for c in SOURCE_CITY_ORDER)) / len(SOURCE_CITY_ORDER)}
        out['selection_macro_mae'] = 0.5 * (out['macro_city']['inflow_mae'] + out['macro_city']['outflow_mae'])
        return out

def format_metrics(metrics):
    lines = []
    lines.append('AGGREGATE OVER HORIZONS 1-4')
    lines.append('City          Inflow MAE   Inflow RMSE   Outflow MAE  Outflow RMSE')
    lines.append('-------------------------------------------------------------------')
    for city in SOURCE_CITY_ORDER:
        x = metrics['cities'][city]
        lines.append(f"{city:12s} {x['inflow']['mae']:11.6f} {x['inflow']['rmse']:13.6f} {x['outflow']['mae']:12.6f} {x['outflow']['rmse']:13.6f}")
    m = metrics['macro_city']
    lines.append('-------------------------------------------------------------------')
    lines.append(f"MACRO-CITY   {m['inflow_mae']:11.6f} {m['inflow_rmse']:13.6f} {m['outflow_mae']:12.6f} {m['outflow_rmse']:13.6f}")
    if 'macro_city_by_horizon' in metrics:
        lines.append('')
        lines.append('MACRO-CITY BY HORIZON')
        lines.append('Horizon       Inflow MAE   Inflow RMSE   Outflow MAE  Outflow RMSE')
        lines.append('-------------------------------------------------------------------')
        for horizon in FORECAST_HORIZONS:
            x = metrics['macro_city_by_horizon'][f'h{horizon}']
            lines.append(f"h{horizon:<11d} {x['inflow_mae']:11.6f} {x['inflow_rmse']:13.6f} {x['outflow_mae']:12.6f} {x['outflow_rmse']:13.6f}")
    return '\n'.join(lines)

def save_metrics(metrics, output_json, output_txt):
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    Path(output_txt).write_text(format_metrics(metrics) + '\n', encoding='utf-8')

def assert_full_window_counts(metrics, split, expected_windows, city_shapes, forecast_len=4):
    if forecast_len != len(FORECAST_HORIZONS):
        raise RuntimeError(f'Configured forecast_len={forecast_len} but metrics expect {len(FORECAST_HORIZONS)} horizons')
    for city in SOURCE_CITY_ORDER:
        h, w = city_shapes[city]
        expected_total = int(expected_windows[split][city] * forecast_len * h * w)
        expected_one = int(expected_windows[split][city] * h * w)
        for ch in CHANNELS:
            got = int(metrics['cities'][city][ch]['n'])
            if got != expected_total:
                raise RuntimeError(f'{split}/{city}/{ch}: aggregate metric n={got}, expected {expected_total}; evaluation is not the exact complete-window protocol.')
        for horizon in FORECAST_HORIZONS:
            for ch in CHANNELS:
                got = int(metrics['cities'][city]['by_horizon'][f'h{horizon}'][ch]['n'])
                if got != expected_one:
                    raise RuntimeError(f'{split}/{city}/h{horizon}/{ch}: metric n={got}, expected {expected_one}.')
