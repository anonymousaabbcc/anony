import os
import time
import torch
import torch.distributed as dist
from .exact_accum import _all_reduce_flat_in_buckets_
TOTAL_NUMEL = 25346518
BUCKET_MB = 16.0

def main():
    if 'RANK' not in os.environ:
        raise RuntimeError('Launch with torchrun')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world < 2:
        raise RuntimeError(f'Expected distributed world_size >= 2, got {world}')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group('nccl')
    try:
        flat = torch.full((TOTAL_NUMEL,), float(rank + 1), dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)
        dist.barrier()
        t0 = time.perf_counter()
        n_buckets = _all_reduce_flat_in_buckets_(flat, bucket_mb=BUCKET_MB)
        torch.cuda.synchronize(device)
        dist.barrier()
        elapsed = time.perf_counter() - t0
        expected = float(sum(range(1, world + 1)))
        probe_idx = [0, flat.numel() - 1]
        bucket_elems = int(BUCKET_MB * 1024 * 1024 // flat.element_size())
        for k in range(1, n_buckets):
            idx = k * bucket_elems
            if idx < flat.numel():
                probe_idx.extend([idx - 1, idx])
        idx = torch.tensor(sorted(set(probe_idx)), device=device, dtype=torch.long)
        probe = flat.index_select(0, idx)
        if not torch.all(probe == expected):
            raise RuntimeError(f'Bucketed NCCL SUM mismatch on rank {rank}: expected={expected}, got={probe.detach().cpu().tolist()}')
        elapsed_t = torch.tensor([elapsed], device=device, dtype=torch.float64)
        dist.all_reduce(elapsed_t, op=dist.ReduceOp.MAX)
        if rank == 0:
            mib = TOTAL_NUMEL * 4 / 1024 ** 2
            print('URBANBIND BUCKETED NCCL SYNC: PASS')
            print(f'world_size: {world}')
            print(f'total_numel: {TOTAL_NUMEL} ({mib:.2f} MiB FP32)')
            print(f'bucket_mb: {BUCKET_MB}')
            print(f'buckets: {n_buckets}')
            print(f'max_elapsed_seconds: {float(elapsed_t.item()):.4f}')
    finally:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
