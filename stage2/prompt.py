from pathlib import Path
from typing import Dict, List
from PIL import Image
import torch
from .config import SOURCE_CITY_ORDER
from .qa_compat import import_qa_design
P8_STAGE2_QUESTION = "Using each city's 8-hour history, forecast the complete fine-grained inflow and outflow maps for horizons [1, 2, 3, 4]."

def _fmt_time(x) -> str:
    return str(x).replace('T', ' ')

def build_system_prompt(sample: dict) -> str:
    base = import_qa_design()['system_prompt']
    chunks = []
    for city in SOURCE_CITY_ORDER:
        t_hist = sample['cities'][city]['time_hist']
        t_future = sample['cities'][city]['time_future']
        chunks.append(f'{city}: history {_fmt_time(t_hist[0])} to {_fmt_time(t_hist[-1])}; forecast {_fmt_time(t_future[0])} to {_fmt_time(t_future[-1])}')
    return base + ' Current Stage-2 windows are independent city histories with the same one-hour interval; their absolute calendar dates do not need to match. ' + ' | '.join(chunks) + '.'

def build_messages(sample: dict):
    user = []
    for city in SOURCE_CITY_ORDER:
        user.append({'type': 'text', 'text': f'{city} historical urban flow maps:'})
        user.append({'type': 'image', 'image': Path(sample['cities'][city]['image_path']).resolve().as_uri()})
    user.append({'type': 'text', 'text': P8_STAGE2_QUESTION})
    return [{'role': 'system', 'content': [{'type': 'text', 'text': build_system_prompt(sample)}]}, {'role': 'user', 'content': user}]

class Stage2Collator:

    def __init__(self, processor, max_seq_length: int=8192):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_seq_length = int(max_seq_length)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _load_images(paths: List[str]):
        out = []
        for p in paths:
            with Image.open(p) as im:
                out.append(im.convert('RGB').copy())
        return out

    def _encode_prompt(self, sample):
        messages = build_messages(sample)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_paths = [sample['cities'][c]['image_path'] for c in SOURCE_CITY_ORDER]
        images = self._load_images(image_paths)
        enc = self.processor(text=[text], images=images, padding=False, return_tensors='pt')
        ids = enc['input_ids'][0]
        attn = enc['attention_mask'][0]
        if ids.numel() > self.max_seq_length:
            raise ValueError(f'Stage-2 prompt length {ids.numel()} exceeds max_seq_length={self.max_seq_length}; do not silently truncate multimodal context.')
        return {'input_ids': ids, 'attention_mask': attn, 'pixel_values': enc['pixel_values'], 'image_grid_thw': enc['image_grid_thw']}

    def __call__(self, samples: List[dict]):
        encoded = [self._encode_prompt(s) for s in samples]
        bsz = len(samples)
        max_len = max((x['input_ids'].numel() for x in encoded))
        input_ids = torch.full((bsz, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        prompt_lengths = torch.zeros(bsz, dtype=torch.long)
        for i, x in enumerate(encoded):
            n = x['input_ids'].numel()
            input_ids[i, :n] = x['input_ids']
            attention_mask[i, :n] = x['attention_mask']
            prompt_lengths[i] = n
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'prompt_lengths': prompt_lengths, 'pixel_values': torch.cat([x['pixel_values'] for x in encoded], dim=0), 'image_grid_thw': torch.cat([x['image_grid_thw'] for x in encoded], dim=0), 'cities': {}, 'joint_index': torch.tensor([s['joint_index'] for s in samples], dtype=torch.long)}
        for city in SOURCE_CITY_ORDER:
            batch['cities'][city] = {'x': torch.stack([s['cities'][city]['x'] for s in samples], dim=0), 'y': torch.stack([s['cities'][city]['y'] for s in samples], dim=0), 'active_once': torch.tensor([s['cities'][city]['active_once'] for s in samples], dtype=torch.bool), 'window_index': torch.tensor([s['cities'][city]['window_index'] for s in samples], dtype=torch.long)}
        return batch
