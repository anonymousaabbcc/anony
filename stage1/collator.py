from pathlib import Path
from typing import List
import torch
from PIL import Image

class QwenStage1Collator:

    def __init__(self, processor, max_seq_length: int=8192):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_seq_length = int(max_seq_length)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        if im_end_id is None or im_end_id == self.tokenizer.unk_token_id:
            if self.tokenizer.eos_token_id is None:
                raise ValueError('Cannot resolve <|im_end|> or eos_token_id.')
            im_end_id = self.tokenizer.eos_token_id
        self.im_end_id = int(im_end_id)

    def _load_images(self, paths):
        images = []
        for p in paths:
            with Image.open(p) as im:
                images.append(im.convert('RGB').copy())
        return images

    def _encode_one(self, item):
        rec = item['record']
        prompt_messages = [m for m in rec['messages'] if m['role'] != 'assistant']
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        images = self._load_images(rec['image_paths'])
        enc = self.processor(text=[prompt_text], images=images, padding=False, return_tensors='pt')
        prompt_ids = enc['input_ids'][0]
        prompt_attention = enc['attention_mask'][0]
        answer_ids = self.tokenizer(rec['answer_text'], add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        completion_ids = torch.cat([answer_ids, torch.tensor([self.im_end_id], dtype=torch.long)], dim=0)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=0)
        attention_mask = torch.cat([prompt_attention, torch.ones_like(completion_ids)], dim=0)
        labels = torch.full_like(input_ids, -100)
        labels[prompt_ids.numel():] = completion_ids
        if input_ids.numel() > self.max_seq_length:
            raise ValueError(f"{rec['sample_id']}: sequence length {input_ids.numel()} > max_seq_length={self.max_seq_length}. Do not silently truncate multimodal training samples.")
        readout_position = prompt_ids.numel() - 1
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'readout_position': readout_position, 'pixel_values': enc['pixel_values'], 'image_grid_thw': enc['image_grid_thw'], 'value_target': item['value_target'], 'value_mask': item['value_mask'], 'tensor_targets': item['tensor_targets'], 'record': rec}

    def __call__(self, batch: List[dict]):
        encoded = [self._encode_one(item) for item in batch]
        max_len = max((x['input_ids'].numel() for x in encoded))
        bsz = len(encoded)
        input_ids = torch.full((bsz, max_len), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), fill_value=-100, dtype=torch.long)
        readout_positions = torch.zeros(bsz, dtype=torch.long)
        for i, x in enumerate(encoded):
            n = x['input_ids'].numel()
            input_ids[i, :n] = x['input_ids']
            attention_mask[i, :n] = x['attention_mask']
            labels[i, :n] = x['labels']
            readout_positions[i] = x['readout_position']
        pixel_values = torch.cat([x['pixel_values'] for x in encoded], dim=0)
        image_grid_thw = torch.cat([x['image_grid_thw'] for x in encoded], dim=0)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'pixel_values': pixel_values, 'image_grid_thw': image_grid_thw, 'readout_positions': readout_positions, 'value_target': torch.stack([x['value_target'] for x in encoded], dim=0), 'value_mask': torch.stack([x['value_mask'] for x in encoded], dim=0), 'tensor_targets': [x['tensor_targets'] for x in encoded], 'records': [x['record'] for x in encoded]}
