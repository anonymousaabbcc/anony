import json
import os
import random
import shutil
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .model import Stage1UrbanVLM, find_unique_module, unfreeze_visual_merger

def _cpu_state_dict(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}

def _distributed():
    return dist.is_available() and dist.is_initialized()

def _rank():
    return dist.get_rank() if _distributed() else 0

def _world_size():
    return dist.get_world_size() if _distributed() else 1

def _barrier():
    if _distributed():
        dist.barrier()

def capture_rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(), 'torch_cuda': None}
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state()
    return state

def _as_cpu_byte_tensor(x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.uint8)
    return x.detach().to(device='cpu', dtype=torch.uint8).contiguous()

def restore_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(_as_cpu_byte_tensor(state['torch_cpu']))
    if torch.cuda.is_available() and state.get('torch_cuda') is not None:
        torch.cuda.set_rng_state(_as_cpu_byte_tensor(state['torch_cuda']))

def _gather_rng_states():
    local = capture_rng_state()
    if not _distributed():
        return [local]
    gathered = [None for _ in range(_world_size())]
    dist.all_gather_object(gathered, local)
    return gathered

def _write_model_payload(model: Stage1UrbanVLM, output_dir: Path, processor, cfg, normalization_stats: str, metrics: Optional[dict]=None, save_processor: bool=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / 'lora_adapter'
    model.vlm.save_pretrained(adapter_dir)
    if save_processor and processor is not None:
        processor_dir = output_dir / 'processor'
        processor.save_pretrained(processor_dir)
    _, merger = find_unique_module(model.vlm, 'visual.merger')
    torch.save({'visual_merger': _cpu_state_dict(merger), 'value_head': _cpu_state_dict(model.value_head), 'tensor_head': _cpu_state_dict(model.tensor_head)}, output_dir / 'stage1_auxiliary.pt')
    (output_dir / 'stage1_config.json').write_text(json.dumps(cfg.to_dict(), indent=2), encoding='utf-8')
    if metrics is not None:
        (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    stats_src = Path(normalization_stats)
    stats_dst = output_dir / 'normalization_stats.json'
    if stats_src.exists() and stats_src.resolve() != stats_dst.resolve():
        shutil.copyfile(stats_src, stats_dst)

def _atomic_replace_dir(tmp_dir: Path, final_dir: Path):
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(tmp_dir, final_dir)

def save_inference_checkpoint(model: Stage1UrbanVLM, processor, output_dir: str, cfg, normalization_stats: str, metrics: dict):
    _barrier()
    if _rank() == 0:
        final_dir = Path(output_dir)
        tmp_dir = final_dir.parent / f'.{final_dir.name}.tmp'
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        _write_model_payload(model=model, output_dir=tmp_dir, processor=processor, cfg=cfg, normalization_stats=normalization_stats, metrics=metrics, save_processor=True)
        _atomic_replace_dir(tmp_dir, final_dir)
    _barrier()

def save_training_checkpoint(model: Stage1UrbanVLM, optimizer, scheduler, processor, output_dir: str, cfg, normalization_stats: str, progress: dict, metrics: Optional[dict]=None):
    rng_states = _gather_rng_states()
    _barrier()
    if _rank() == 0:
        final_dir = Path(output_dir)
        tmp_dir = final_dir.parent / f'.{final_dir.name}.tmp'
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        _write_model_payload(model=model, output_dir=tmp_dir, processor=processor, cfg=cfg, normalization_stats=normalization_stats, metrics=metrics, save_processor=False)
        torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'progress': progress, 'rng_states': rng_states, 'world_size': _world_size()}, tmp_dir / 'training_state.pt')
        _atomic_replace_dir(tmp_dir, final_dir)
    _barrier()

def _load_base_with_adapter(checkpoint_dir: Path, model_name: str, torch_dtype, trainable: bool):
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype, attn_implementation='sdpa')
    base.config.use_cache = False
    vlm = PeftModel.from_pretrained(base, checkpoint_dir / 'lora_adapter', is_trainable=trainable)
    if trainable:
        unfreeze_visual_merger(vlm)
        if hasattr(vlm, 'gradient_checkpointing_enable'):
            try:
                vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            except TypeError:
                vlm.gradient_checkpointing_enable()
        if hasattr(vlm, 'enable_input_require_grads'):
            vlm.enable_input_require_grads()
    return vlm

def load_stage1_checkpoint(checkpoint_dir: str, model_name: str='Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype=torch.bfloat16, device: str='cuda', trainable: bool=False):
    ckpt = Path(checkpoint_dir)
    config = json.loads((ckpt / 'stage1_config.json').read_text(encoding='utf-8'))
    vlm = _load_base_with_adapter(checkpoint_dir=ckpt, model_name=model_name, torch_dtype=torch_dtype, trainable=trainable)
    aux = torch.load(ckpt / 'stage1_auxiliary.pt', map_location='cpu')
    _, merger = find_unique_module(vlm, 'visual.merger')
    merger.load_state_dict(aux['visual_merger'])
    model = Stage1UrbanVLM(vlm=vlm, value_head_dim=int(config['value_head_dim']), tensor_head_dim=int(config['tensor_head_dim']), lambda_value=float(config['lambda_value']), lambda_tensor=float(config['lambda_tensor']))
    model.value_head.load_state_dict(aux['value_head'])
    model.tensor_head.load_state_dict(aux['tensor_head'])
    model.to(device)
    processor_dir = ckpt / 'processor'
    if processor_dir.exists():
        processor = AutoProcessor.from_pretrained(processor_dir, use_fast=False)
    else:
        processor = AutoProcessor.from_pretrained(model_name, min_pixels=int(config['min_pixels']), max_pixels=int(config['max_pixels']), use_fast=False)
    return (model, processor, config)

def load_training_state(checkpoint_dir: str, optimizer, scheduler, device):
    ckpt = Path(checkpoint_dir)
    payload = torch.load(ckpt / 'training_state.pt', map_location=device)
    saved_world_size = int(payload.get('world_size', 1))
    current_world_size = _world_size()
    if saved_world_size != current_world_size:
        raise RuntimeError(f'Exact mid-epoch resume requires the same world size. Checkpoint world_size={saved_world_size}, current world_size={current_world_size}.')
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    rng_states = payload.get('rng_states')
    if rng_states:
        restore_rng_state(rng_states[_rank()])
    return payload['progress']
