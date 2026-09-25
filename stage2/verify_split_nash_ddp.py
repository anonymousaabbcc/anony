import copy
import os
import socket
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from .config import SOURCE_CITY_ORDER
from .exact_accum import exact_nash_accumulated_update

class Tiny(nn.Module):

    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        self.shared = nn.Linear(6, 2)
        self.head = nn.ModuleList([nn.Linear(4, 1) for _ in range(3)])

    def bargaining_parameters(self):
        return list(self.shared.parameters())

    def private_parameters_by_city(self):
        return {c: list(self.enc[i].parameters()) + list(self.head[i].parameters()) for i, c in enumerate(SOURCE_CITY_ORDER)}

    def assert_gradient_routing_partition(self):
        shared = self.bargaining_parameters()
        private = self.private_parameters_by_city()
        routed = shared + [p for c in SOURCE_CITY_ORDER for p in private[c]]
        trainable = [p for p in self.parameters() if p.requires_grad]
        if len({id(p) for p in routed}) != len(routed):
            raise RuntimeError('overlap')
        if {id(p) for p in routed} != {id(p) for p in trainable}:
            raise RuntimeError('partition mismatch')
        return {'shared': shared, 'private_by_city': private}

    def forward(self, batch):
        toks = [self.enc[i](batch['x'][:, i]) for i in range(3)]
        shared = self.shared(torch.cat(toks, dim=-1))
        losses = []
        for i in range(3):
            pred = self.head[i](torch.cat([toks[i], shared], dim=-1)).squeeze(-1)
            losses.append((pred - batch['y'][:, i]).square().mean())
        return SimpleNamespace(city_losses=torch.stack(losses))

class FixedNash:
    alpha_floor = 0.05
    normalize_mean = True

    def __init__(self):
        self.step = 0
        self.alpha = torch.tensor([1.7, 1.25, 0.05])

    def should_update_now(self):
        return False

    def cached_weights_and_advance(self, device, dtype=torch.float32):
        self.step += 1
        return self.alpha.to(device=device, dtype=dtype)

def _batch(seed):
    g = torch.Generator().manual_seed(seed)
    return {'x': torch.randn(1, 3, 2, generator=g), 'y': torch.randn(1, 3, generator=g)}

def _state_vector(model):
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])

def _worker(rank, world, port, initial_state, return_dict):
    dist.init_process_group(backend='gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=world)
    try:
        base = Tiny()
        base.load_state_dict(initial_state)
        model = DDP(base)
        opt = torch.optim.SGD(model.parameters(), lr=0.02)
        all_batches = [_batch(100 + i) for i in range(4)]
        local = all_batches[rank * 2:(rank + 1) * 2]
        exact_nash_accumulated_update(model=model, cpu_batches=local, device=torch.device('cpu'), nash=FixedNash(), optimizer=opt, scheduler=None, trainable_params=[p for p in model.parameters() if p.requires_grad], max_grad_norm=1000000000.0)
        vec = _state_vector(model.module)
        if rank == 0:
            return_dict['distributed'] = vec.tolist()
        dist.barrier()
    finally:
        dist.destroy_process_group()

def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def main():
    torch.manual_seed(123)
    initial = Tiny()
    initial_state = copy.deepcopy(initial.state_dict())
    ref = Tiny()
    ref.load_state_dict(initial_state)
    opt = torch.optim.SGD(ref.parameters(), lr=0.02)
    exact_nash_accumulated_update(model=ref, cpu_batches=[_batch(100 + i) for i in range(4)], device=torch.device('cpu'), nash=FixedNash(), optimizer=opt, scheduler=None, trainable_params=[p for p in ref.parameters() if p.requires_grad], max_grad_norm=1000000000.0)
    ref_vec = _state_vector(ref)
    world = 2
    port = _free_port()
    manager = mp.Manager()
    result = manager.dict()
    mp.spawn(_worker, args=(world, port, initial_state, result), nprocs=world, join=True)
    ddp_vec = torch.tensor(result['distributed'], dtype=ref_vec.dtype)
    max_diff = float((ref_vec - ddp_vec).abs().max().item())
    if max_diff > 2e-07:
        raise RuntimeError(f'Distributed split-gradient mean mismatch: {max_diff}')
    print('SPLIT-NASH DDP GLOBAL-MEAN CHECK: PASS')
    print('max parameter difference vs single-process global batch:', max_diff)
if __name__ == '__main__':
    main()
