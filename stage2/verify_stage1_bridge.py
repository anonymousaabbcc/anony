import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import Stage2Config
from .data import JointMultiCityDataset
from .distributed import move_batch_to_device, seed_everything
from .normalization import TorchFlowNormalizer
from .prompt import Stage2Collator
from .vlm_bridge import FrozenStage1Bridge

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    cfg = Stage2Config()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    seed_everything(cfg.seed)
    norm = TorchFlowNormalizer(cfg.normalization_stats)
    bridge = FrozenStage1Bridge(cfg, device=device)
    ds = JointMultiCityDataset('train', norm, str(cfg.visual_cache_root), cfg.seed, training=True, max_joint_steps=1, debug_sequential=True)
    collator = Stage2Collator(bridge.processor, max_seq_length=int(bridge.stage1_cfg.get('max_seq_length', 8192)))
    batch = move_batch_to_device(next(iter(DataLoader(ds, batch_size=1, collate_fn=collator))), device)
    ctx = bridge.prepare_context(batch)
    bridge.stage1._captured_hidden = None
    _ = bridge.stage1.vlm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'], use_cache=False, output_hidden_states=False, return_dict=True)
    hidden = bridge.stage1._captured_hidden
    rows = torch.arange(hidden.shape[0], device=device)
    ref = hidden[rows, batch['prompt_lengths'] - 1]
    diff = (ctx.s_qa.float() - ref.float()).abs()
    report = {'max_abs_diff': float(diff.max().cpu()), 'mean_abs_diff': float(diff.mean().cpu()), 'allclose_atol_1e-4_rtol_1e-4': bool(torch.allclose(ctx.s_qa.float(), ref.float(), atol=0.0001, rtol=0.0001)), 'stage1_trainable_params': bridge.trainable_parameter_count, 'hidden_size': bridge.hidden_size, 'prompt_length': int(batch['prompt_lengths'][0].cpu())}
    if not report['allclose_atol_1e-4_rtol_1e-4']:
        raise RuntimeError('Stage-2 bridge S_QA does not reproduce the direct Stage-1 frozen forward')
    print('STAGE-1 BRIDGE EQUIVALENCE: PASS')
    print(json.dumps(report, indent=2))
    path = Path(cfg.log_root) / 'stage1_bridge_equivalence.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', path)
if __name__ == '__main__':
    main()
