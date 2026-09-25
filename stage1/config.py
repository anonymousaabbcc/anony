import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple
SOURCE_CITY_ORDER = ('NYCTAXI', 'BIKECHI', 'NYC-BIKE')
TARGET_CITY_ORDER = ('BIKEDC',)
CITY_ORDER = SOURCE_CITY_ORDER
CITY_SHAPES: Dict[str, Tuple[int, int]] = {'NYCTAXI': (10, 20), 'BIKECHI': (15, 18), 'NYC-BIKE': (16, 8)}
CHANNELS = ('inflow', 'outflow')
MAX_VALUE_DIM = 48
PROJECT_ROOT = Path(os.environ.get('URBANBIND_ROOT', Path(__file__).resolve().parents[1]))
DEFAULT_QA_ROOT = Path(os.environ.get('URBANBIND_QA_SOURCE_ROOT', PROJECT_ROOT / 'outputs' / 'qa' / 'source'))
DEFAULT_STAGE1_OUTPUT = Path(os.environ.get('URBANBIND_STAGE1_OUTPUT', PROJECT_ROOT / 'outputs' / 'stage1'))

@dataclass
class Stage1Config:
    model_name: str = 'Qwen/Qwen2.5-VL-3B-Instruct'
    qa_root: str = str(DEFAULT_QA_ROOT)
    output_dir: str = str(DEFAULT_STAGE1_OUTPUT)
    normalization_stats: str = str(DEFAULT_STAGE1_OUTPUT / 'normalization_stats.json')
    min_pixels: int = 512 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    max_seq_length: int = 8192
    use_fast_image_processor: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_epochs: int = 5
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_workers: int = 2
    lora_lr: float = 0.0001
    merger_lr: float = 2e-05
    head_lr: float = 0.0005
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lambda_value: float = 1.0
    lambda_tensor: float = 1.0
    value_head_dim: int = 512
    tensor_head_dim: int = 1024
    seed: int = 42
    log_every_updates: int = 20
    checkpoint_every_updates: int = 250
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.001
    def train_jsonl(self) -> str:
        return str(Path(self.qa_root) / 'train' / 'QA_train.jsonl')
    def valid_jsonl(self) -> str:
        return str(Path(self.qa_root) / 'valid' / 'QA_valid.jsonl')
    def test_jsonl(self) -> str:
        return str(Path(self.qa_root) / 'test' / 'QA_test.jsonl')
    def to_dict(self):
        return asdict(self)
