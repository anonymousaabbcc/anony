import json
from pathlib import Path
from typing import Optional, Sequence
from torch.utils.data import Dataset
from .config import CITY_ORDER
from .normalization import FlowNormalizer, encode_value_target, load_normalized_tensor_targets

class Stage1QADataset(Dataset):

    def __init__(self, jsonl_path: str, normalizer: FlowNormalizer, template_ids: Optional[Sequence[str]]=None, max_samples: Optional[int]=None):
        self.path = str(jsonl_path)
        allowed = set(template_ids) if template_ids else None
        self.records = []
        with Path(jsonl_path).open('r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                unknown_cities = [city for city in rec.get('cities', []) if city not in CITY_ORDER]
                if unknown_cities:
                    raise ValueError(f"{rec.get('sample_id', '<unknown>')}: non-source cities found in Stage-1 data: {unknown_cities}. Allowed source cities: {list(CITY_ORDER)}")
                if allowed is not None and rec['template_id'] not in allowed:
                    continue
                self.records.append(rec)
                if max_samples is not None and len(self.records) >= max_samples:
                    break
        if not self.records:
            raise ValueError(f'No records loaded from {jsonl_path}')
        self.normalizer = normalizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        value_target, value_mask = encode_value_target(record, self.normalizer)
        tensor_targets = load_normalized_tensor_targets(record, self.normalizer)
        return {'record': record, 'value_target': value_target, 'value_mask': value_mask, 'tensor_targets': tensor_targets}
