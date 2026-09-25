import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from .collator import QwenStage1Collator
from .config import Stage1Config
from .dataset import Stage1QADataset
from .model import build_trainable_stage1_model, trainable_parameter_report
from .normalization import FlowNormalizer

def run_one(template_id, cfg, processor, normalizer, model, device):
    ds = Stage1QADataset(cfg.train_jsonl(), normalizer, template_ids=[template_id], max_samples=1)
    collator = QwenStage1Collator(processor, max_seq_length=cfg.max_seq_length)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))
    rec = batch['records'][0]
    tensor_keys = ['input_ids', 'attention_mask', 'labels', 'pixel_values', 'image_grid_thw', 'readout_positions', 'value_target', 'value_mask']
    for key in tensor_keys:
        batch[key] = batch[key].to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'], readout_positions=batch['readout_positions'], value_target=batch['value_target'], value_mask=batch['value_mask'], tensor_targets=batch['tensor_targets'], return_predictions=False)
    out.loss.backward()
    print('=' * 80)
    print('Template:', template_id)
    print('Sample:', rec['sample_id'])
    print('Question:', rec['question'])
    print('Answer:', rec['answer_text'])
    print('input_ids:', tuple(batch['input_ids'].shape))
    print('pixel_values:', tuple(batch['pixel_values'].shape))
    print('image_grid_thw:', batch['image_grid_thw'].tolist())
    print('supervised_tokens:', int((batch['labels'] != -100).sum()))
    print('readout_position:', int(batch['readout_positions'][0]))
    print('losses:', {'total': float(out.loss.detach()), 'ce': float(out.ce_loss.detach()), 'value': float(out.value_loss.detach()), 'tensor': float(out.tensor_loss.detach())})
    print('Backward: PASS')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='./outputs/stage1/normalization_stats.json')
    args = ap.parse_args()
    cfg = Stage1Config(normalization_stats=args.stats)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    processor = AutoProcessor.from_pretrained(cfg.model_name, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels, use_fast=False)
    normalizer = FlowNormalizer(cfg.normalization_stats)
    model, merger_name = build_trainable_stage1_model(cfg)
    model.cuda()
    print('Merger:', merger_name)
    print('Trainable parameters:', json.dumps(trainable_parameter_report(model), indent=2))
    for tid in ['U1', 'P3', 'P8']:
        run_one(tid, cfg, processor, normalizer, model, torch.device('cuda'))
if __name__ == '__main__':
    main()
