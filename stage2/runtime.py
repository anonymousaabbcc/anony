import json
import time
from pathlib import Path
import torch
import torch.distributed as dist

def _distributed():
    return dist.is_available() and dist.is_initialized()

def _world_size():
    return dist.get_world_size() if _distributed() else 1

def _rank():
    return dist.get_rank() if _distributed() else 0

def synchronize_for_timing(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    if _distributed():
        dist.barrier()

def start_wall_timer(device):
    synchronize_for_timing(device)
    return time.perf_counter()

def stop_wall_timer(start_time, device):
    synchronize_for_timing(device)
    local = time.perf_counter() - float(start_time)
    x = torch.tensor([local], dtype=torch.float64, device=device)
    if _distributed():
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return float(x.cpu())

def reset_peak_memory(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

def peak_memory_across_ranks(device):
    if device.type != 'cuda':
        return {'peak_allocated_gb': 0.0, 'peak_reserved_gb': 0.0}
    vals = torch.tensor([torch.cuda.max_memory_allocated(device) / 1024 ** 3, torch.cuda.max_memory_reserved(device) / 1024 ** 3], dtype=torch.float64, device=device)
    if _distributed():
        dist.all_reduce(vals, op=dist.ReduceOp.MAX)
    return {'peak_allocated_gb': float(vals[0].cpu()), 'peak_reserved_gb': float(vals[1].cpu())}

def save_runtime_summary(summary, path):
    if _rank() != 0:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

def runtime_definition():
    return {'timing': 'Synchronized wall-clock time. CUDA is synchronized and DDP ranks are barriered at phase boundaries; reported time is the maximum across ranks.', 'training_wall_seconds': 'Sum of completed epoch wall times, including training, validation, checkpoint writing, and epoch-level reporting; model initialization is excluded.', 'training_only_seconds': 'Optimizer/Nash training loop wall time, including data loading/collation and all training forwards/backwards; validation and checkpoint writing are excluded.', 'validation_seconds': 'Full validation evaluation wall time accumulated across completed epochs.', 'final_test_seconds': 'Full test evaluation wall time after warm-up; model initialization/checkpoint loading and warm-up are excluded. Data loading/collation, frozen VLM inference, UrbanBind inference, and metric accumulation are included.', 'memory': 'Peak CUDA allocated/reserved memory is the maximum observed across DDP ranks.'}
