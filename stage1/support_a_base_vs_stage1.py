import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .checkpoint import load_stage1_checkpoint
from .collator import QwenStage1Collator
from .normalization import FlowNormalizer
from .support_common import RecordDataset, causal_supervised_token_count, load_jsonl_records, move_multimodal_batch, write_json_and_text

class CEAccumulator:

    def __init__(self):
        self.nll = 0.0
        self.tokens = 0
        self.sample_ce_sum = 0.0
        self.samples = 0

    def update(self, loss: float, tokens: int):
        self.nll += float(loss) * int(tokens)
        self.tokens += int(tokens)
        self.sample_ce_sum += float(loss)
        self.samples += 1

    def result(self):
        token_ce = self.nll / max(self.tokens, 1)
        return {'token_ce': token_ce, 'perplexity': float(math.exp(min(token_ce, 50.0))), 'sample_mean_ce': self.sample_ce_sum / max(self.samples, 1), 'samples': self.samples, 'supervised_tokens': self.tokens}

def processor_from_checkpoint(checkpoint: Path, model_name: str, cfg: dict):
    pdir = checkpoint / 'processor'
    if pdir.exists():
        return AutoProcessor.from_pretrained(pdir, use_fast=False)
    return AutoProcessor.from_pretrained(model_name, min_pixels=int(cfg['min_pixels']), max_pixels=int(cfg['max_pixels']), use_fast=False)

@torch.no_grad()
def evaluate_vlm(vlm, processor, records, normalizer, max_seq_length, device):
    ds = RecordDataset(records, normalizer)
    collator = QwenStage1Collator(processor, max_seq_length=max_seq_length)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)
    acc = defaultdict(CEAccumulator)
    vlm.eval()
    for i, batch in enumerate(loader, start=1):
        rec = batch['records'][0]
        tid = rec['template_id']
        batch = move_multimodal_batch(batch, device)
        n_tok = causal_supervised_token_count(batch['labels'])
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = vlm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'], use_cache=False, return_dict=True)
        loss = float(out.loss.float().item())
        acc['overall'].update(loss, n_tok)
        acc[tid].update(loss, n_tok)
        acc[tid[0]].update(loss, n_tok)
        if i % 100 == 0:
            print(f'processed {i}/{len(loader)}')
    result = {k: v.result() for k, v in sorted(acc.items())}
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./outputs/stage1/best')
    ap.add_argument('--test-jsonl', default='./outputs/qa/source/test/QA_test.jsonl')
    ap.add_argument('--output-dir', default='./outputs/stage1/support_evidence')
    ap.add_argument('--max-per-template', type=int, default=None, help='Debug only. Omit for the full frozen QA test set.')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    checkpoint = Path(args.checkpoint)
    cfg = json.loads((checkpoint / 'stage1_config.json').read_text(encoding='utf-8'))
    model_name = cfg.get('model_name', 'Qwen/Qwen2.5-VL-3B-Instruct')
    device = torch.device(args.device)
    records = load_jsonl_records(args.test_jsonl, max_per_template=args.max_per_template)
    normalizer = FlowNormalizer(str(checkpoint / 'normalization_stats.json'))
    processor = processor_from_checkpoint(checkpoint, model_name, cfg)
    print('=== A1: original base VLM ===')
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    base.config.use_cache = False
    base_result = evaluate_vlm(base, processor, records, normalizer, int(cfg['max_seq_length']), device)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    print('=== A2: Stage-1 adapted VLM ===')
    stage1, _, _ = load_stage1_checkpoint(args.checkpoint, device=str(device))
    stage1.eval()
    stage1_result = evaluate_vlm(stage1.vlm, processor, records, normalizer, int(cfg['max_seq_length']), device)
    keys = sorted(set(base_result) | set(stage1_result))
    comparison = {}
    for k in keys:
        if k not in base_result or k not in stage1_result:
            continue
        b = base_result[k]
        s = stage1_result[k]
        comparison[k] = {'base_token_ce': b['token_ce'], 'stage1_token_ce': s['token_ce'], 'ce_reduction_pct': 100.0 * (b['token_ce'] - s['token_ce']) / max(b['token_ce'], 1e-12), 'base_perplexity': b['perplexity'], 'stage1_perplexity': s['perplexity'], 'samples': s['samples'], 'supervised_tokens': s['supervised_tokens']}
    payload = {'analysis': 'A_base_vlm_vs_stage1_vlm', 'metric': 'held-out assistant-token cross entropy on identical multimodal QA test inputs', 'test_jsonl': args.test_jsonl, 'max_per_template': args.max_per_template, 'base': base_result, 'stage1': stage1_result, 'comparison': comparison}
    lines = ['A. Base VLM vs Stage-1 VLM', 'Metric: held-out assistant-token CE on identical images/prompts/answers.', 'Lower is better. Full paper evidence should use max_per_template=None.', '', f"{'Group':<10} {'Base CE':>10} {'Stage1 CE':>10} {'Reduction%':>11} {'Base PPL':>10} {'Stage1 PPL':>10}"]
    order = ['overall', 'U', 'P'] + [f'U{i}' for i in range(1, 19)] + [f'P{i}' for i in range(1, 9)]
    for k in order:
        if k not in comparison:
            continue
        r = comparison[k]
        lines.append(f"{k:<10} {r['base_token_ce']:>10.6f} {r['stage1_token_ce']:>10.6f} {r['ce_reduction_pct']:>11.2f} {r['base_perplexity']:>10.3f} {r['stage1_perplexity']:>10.3f}")
    write_json_and_text(Path(args.output_dir), 'A_base_vs_stage1', payload, '\n'.join(lines))
if __name__ == '__main__':
    main()
