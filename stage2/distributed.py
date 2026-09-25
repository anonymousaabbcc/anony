import os
import random
import numpy as np
import torch
import torch.distributed as dist

def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        return (rank, world, local_rank)
    return (0, 1, 0)

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def rank():
    return dist.get_rank() if is_distributed() else 0

def world_size():
    return dist.get_world_size() if is_distributed() else 1

def barrier():
    if is_distributed():
        dist.barrier()

def cleanup():
    if is_distributed():
        dist.destroy_process_group()

def seed_everything(seed, rank_offset=0):
    s = int(seed) + int(rank_offset)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def _copy_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _copy_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_copy_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple((_copy_to_device(v, device) for v in obj))
    return obj

def copy_batch_to_device(batch, device):
    return _copy_to_device(batch, device)

def move_batch_to_device(batch, device):
    return copy_batch_to_device(batch, device)
