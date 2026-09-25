import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get('URBANBIND_ROOT', Path(__file__).resolve().parents[1]))
from typing import Dict, Tuple
SOURCE_CITY_ORDER = ('NYCTAXI', 'BIKECHI', 'NYC-BIKE')
TARGET_CITY_ORDER = ('BIKEDC',)
CHANNELS = ('inflow', 'outflow')
FORECAST_HORIZONS = (1, 2, 3, 4)
CITY_SHAPES: Dict[str, Tuple[int, int]] = {'NYCTAXI': (10, 20), 'BIKECHI': (15, 18), 'NYC-BIKE': (16, 8)}
EXPECTED_WINDOWS = {'train': {'NYCTAXI': 1501, 'BIKECHI': 1525, 'NYC-BIKE': 3061}, 'valid': {'NYCTAXI': 205, 'BIKECHI': 205, 'NYC-BIKE': 421}, 'test': {'NYCTAXI': 421, 'BIKECHI': 445, 'NYC-BIKE': 877}}

@dataclass
class Stage2Config:
    ablation_name: str = 'full'
    stage1_source: str = 'grounded'
    semantic_mode: str = 'dsr_reentry'
    weighting_mode: str = 'nash'
    stage1_checkpoint: str = str(Path(os.environ.get('URBANBIND_STAGE1_CHECKPOINT', PROJECT_ROOT / 'outputs' / 'stage1' / 'best')))
    model_name: str = 'Qwen/Qwen2.5-VL-3B-Instruct'
    normalization_stats: str = str(Path(os.environ.get('URBANBIND_NORMALIZATION_STATS', PROJECT_ROOT / 'outputs' / 'stage1' / 'best' / 'normalization_stats.json')))
    output_root: str = str(Path(os.environ.get('URBANBIND_STAGE2_OUTPUT', PROJECT_ROOT / 'outputs' / 'stage2')))
    checkpoint_root: str = str(Path(os.environ.get('URBANBIND_STAGE2_CHECKPOINT_ROOT', PROJECT_ROOT / 'checkpoints' / 'stage2')))
    cache_root: str = str(Path(os.environ.get('URBANBIND_STAGE2_CACHE_ROOT', PROJECT_ROOT / 'cache' / 'stage2')))
    log_root: str = str(Path(os.environ.get('URBANBIND_STAGE2_LOG_ROOT', PROJECT_ROOT / 'logs' / 'stage2')))
    history_len: int = 8
    forecast_len: int = 4
    source_city_order: tuple = SOURCE_CITY_ORDER
    encoder_dim: int = 192
    encoder_heads: int = 6
    window_size: int = 4
    spatial_depth: int = 2
    temporal_depth: int = 2
    encoder_mlp_ratio: float = 4.0
    encoder_dropout: float = 0.0
    city_attention_heads: int = 6
    city_attention_depth: int = 1
    num_residual_projectors: int = 3
    projector_hidden_dim: int = 512
    projector_dropout: float = 0.0
    decoder_heads: int = 6
    decoder_mlp_ratio: float = 2.0
    decoder_spatial_depth: int = 2
    decoder_dropout: float = 0.0
    semantic_skip_init: float = 0.5
    temporal_readout_scale_init: float = 0.25
    raw_history_init_logit: float = 5.0
    loss_l1_weight: float = 1.0
    loss_mse_weight: float = 0.2
    loss_spatial_grad_weight: float = 0.05
    nash_update_every: int = 10
    nash_normalize_mean: bool = True
    nash_optim_niter: int = 20
    nash_solver_max_iters: int = 100
    nash_eps: float = 1e-10
    nash_alpha_floor: float = 0.05
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 3
    num_workers: int = 2
    encoder_lr: float = 0.00015
    alignment_lr: float = 0.0001
    head_lr: float = 0.00075
    weight_decay: float = 0.01
    warmup_ratio: float = 0.02
    max_grad_norm: float = 1.0
    gradient_sync_bucket_mb: float = 16.0
    max_epochs: int = 50
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 5e-05
    seed: int = 42
    log_every_updates: int = 20
    checkpoint_every_updates: int = 250

    @property
    def horizon(self) -> int:
        return self.forecast_len

    @property
    def visual_cache_root(self) -> Path:
        return Path(self.cache_root) / 'visuals'

    def run_output_dir(self, run_name: str) -> Path:
        return Path(self.output_root) / run_name

    def run_checkpoint_dir(self, run_name: str) -> Path:
        return Path(self.checkpoint_root) / run_name

    def run_log_dir(self, run_name: str) -> Path:
        return Path(self.log_root) / run_name

    def to_dict(self):
        out = asdict(self)
        out['source_city_order'] = list(self.source_city_order)
        out['horizon'] = self.forecast_len
        out['architecture_version'] = 'urbanbind_v3_2_3_method_aligned_' + str(self.ablation_name)
        return out

    @classmethod
    def from_dict(cls, payload):
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in dict(payload).items() if k in valid}
        if 'source_city_order' in kwargs:
            kwargs['source_city_order'] = tuple(kwargs['source_city_order'])
        return cls(**kwargs)
