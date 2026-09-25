import json
import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

def _module(model):
    return model.module if hasattr(model, 'module') else model

def _distributed():
    return dist.is_available() and dist.is_initialized()

def _rank():
    return dist.get_rank() if _distributed() else 0

def _world_size():
    return dist.get_world_size() if _distributed() else 1

def _barrier():
    if _distributed():
        dist.barrier()

def _cpu_state(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}

def _rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(), 'torch_cuda': None}
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state()
    return state

def _gather_rng_states():
    local = _rng_state()
    if not _distributed():
        return [local]
    gathered = [None for _ in range(_world_size())]
    dist.all_gather_object(gathered, local)
    return gathered

def _cpu_byte(x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.uint8)
    return x.detach().to(device='cpu', dtype=torch.uint8).contiguous()

def _restore_rng(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(_cpu_byte(state['torch_cpu']))
    if torch.cuda.is_available() and state.get('torch_cuda') is not None:
        torch.cuda.set_rng_state(_cpu_byte(state['torch_cuda']))

def save_checkpoint(path, model, optimizer, scheduler, nash, cfg, progress, metrics=None):
    rng_states = _gather_rng_states()
    _barrier()
    if _rank() == 0:
        path = Path(path)
        tmp = path.parent / f'.{path.name}.tmp'
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        m = _module(model)
        torch.save({'multi_city': _cpu_state(m.multi_city), 'alignment': _cpu_state(m.alignment), 'regression': _cpu_state(m.regression)}, tmp / 'stage2_model.pt')
        torch.save({'optimizer': optimizer.state_dict() if optimizer is not None else None, 'scheduler': scheduler.state_dict() if scheduler is not None else None, 'nash': nash.state_dict() if nash is not None else None, 'progress': progress, 'rng_states': rng_states, 'world_size': _world_size()}, tmp / 'training_state.pt')
        meta = {'config': cfg.to_dict(), 'stage1_checkpoint_dependency': cfg.stage1_checkpoint, 'metrics': metrics}
        (tmp / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)
    _barrier()

def load_stage2_weights(path, model, strict=True):
    payload = torch.load(Path(path) / 'stage2_model.pt', map_location='cpu')
    m = _module(model)
    m.multi_city.load_state_dict(payload['multi_city'], strict=strict)
    m.alignment.load_state_dict(payload['alignment'], strict=strict)
    m.regression.load_state_dict(payload['regression'], strict=strict)

def load_training_state(path, optimizer, scheduler, nash):
    payload = torch.load(Path(path) / 'training_state.pt', map_location='cpu')
    saved_world = int(payload.get('world_size', 1))
    if saved_world != _world_size():
        raise RuntimeError(f'Exact resume requires same world size: saved={saved_world}, current={_world_size()}')
    if optimizer is not None and payload.get('optimizer') is not None:
        optimizer.load_state_dict(payload['optimizer'])
    if scheduler is not None and payload.get('scheduler') is not None:
        scheduler.load_state_dict(payload['scheduler'])
    if nash is not None and payload.get('nash') is not None:
        nash.load_state_dict(payload['nash'])
    rng_states = payload.get('rng_states')
    if rng_states:
        _restore_rng(rng_states[_rank()])
    return payload.get('progress', {})
