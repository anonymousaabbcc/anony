from dataclasses import dataclass
from typing import Any, Dict, List
import torch
import torch.distributed as dist
from .config import SOURCE_CITY_ORDER
from .distributed import copy_batch_to_device, is_distributed

@dataclass
class TorchRNGSnapshot:
    cpu: torch.Tensor
    cuda: torch.Tensor | None

def capture_torch_rng(device: torch.device) -> TorchRNGSnapshot:
    cuda_state = None
    if device.type == 'cuda':
        cuda_state = torch.cuda.get_rng_state(device)
    return TorchRNGSnapshot(cpu=torch.get_rng_state(), cuda=cuda_state)

def restore_torch_rng(snapshot: TorchRNGSnapshot, device: torch.device):
    torch.set_rng_state(snapshot.cpu)
    if device.type == 'cuda' and snapshot.cuda is not None:
        torch.cuda.set_rng_state(snapshot.cuda, device)

def _unwrap(model):
    return model.module if hasattr(model, 'module') else model

def _assert_no_grads(params, where: str):
    bad = [i for i, p in enumerate(params) if p.grad is not None]
    if bad:
        raise RuntimeError(f'Parameter gradients must be empty {where}; found {len(bad)} populated tensors.')

def _zeros_like_params(params):
    return [torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format) for p in params]

def _accumulate_(acc, grads, scale=1.0):
    if len(acc) != len(grads):
        raise RuntimeError('Gradient accumulator length mismatch')
    scale = float(scale)
    for a, g in zip(acc, grads):
        if g is None:
            raise RuntimeError('Unexpected unused parameter in split gradient routing')
        a.add_(g.detach().float(), alpha=scale)

def _all_reduce_flat_in_buckets_(flat: torch.Tensor, *, bucket_mb: float):
    if flat.ndim != 1 or not flat.is_contiguous():
        raise ValueError('flat routed gradient must be a contiguous 1-D tensor')
    if bucket_mb <= 0:
        raise ValueError('bucket_mb must be > 0')
    if not is_distributed():
        return 0
    bytes_per_elem = flat.element_size()
    max_elems = max(1, int(float(bucket_mb) * 1024 * 1024 // bytes_per_elem))
    n_buckets = 0
    for start in range(0, flat.numel(), max_elems):
        chunk = flat.narrow(0, start, min(max_elems, flat.numel() - start))
        dist.all_reduce(chunk, op=dist.ReduceOp.SUM)
        n_buckets += 1
    return n_buckets

def _global_mean_and_assign_(param_groups, grad_groups, *, micro_steps: int, bucket_mb: float=16.0):
    params = [p for group in param_groups for p in group]
    grads = [g for group in grad_groups for g in group]
    if len(params) != len(grads) or not params:
        raise RuntimeError('Invalid routed gradient groups')
    if micro_steps <= 0:
        raise RuntimeError('micro_steps must be positive')
    flat = torch.cat([g.reshape(-1) for g in grads], dim=0).contiguous()
    if is_distributed():
        signature = torch.tensor([flat.numel(), len(params)], device=flat.device, dtype=torch.int64)
        sig_min = signature.clone()
        sig_max = signature.clone()
        dist.all_reduce(sig_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(sig_max, op=dist.ReduceOp.MAX)
        if not torch.equal(sig_min, sig_max):
            raise RuntimeError(f'Routed-gradient synchronization signature differs across ranks: local={signature.tolist()}, min={sig_min.tolist()}, max={sig_max.tolist()}')
    world = dist.get_world_size() if is_distributed() else 1
    _all_reduce_flat_in_buckets_(flat, bucket_mb=bucket_mb)
    flat.div_(float(world * int(micro_steps)))
    offset = 0
    for p in params:
        n = p.numel()
        view = flat[offset:offset + n].view_as(p)
        p.grad = view.to(dtype=p.dtype).clone()
        offset += n
    if offset != flat.numel():
        raise RuntimeError('Routed gradient flatten/unflatten size mismatch')

def _grad_norm(params):
    sq = None
    for p in params:
        if p.grad is None:
            continue
        term = p.grad.detach().float().square().sum()
        sq = term if sq is None else sq + term
    if sq is None:
        device = params[0].device if params else torch.device('cpu')
        return torch.zeros((), device=device)
    return torch.sqrt(sq)

@dataclass
class ExactNashUpdateReport:
    alpha: torch.Tensor
    probe_city_losses_local_mean: torch.Tensor
    update_city_losses_local_mean: torch.Tensor
    weighted_loss_local_mean: torch.Tensor
    private_loss_local_mean: torch.Tensor
    grad_norm: torch.Tensor
    shared_grad_norm: torch.Tensor
    private_grad_norms: torch.Tensor
    micro_steps: int
    nash_solves: int
    used_nash_probe: bool
    optimizer_steps: int
    scheduler_steps: int
    max_probe_update_city_loss_abs_diff: float
    routing_mode: str = 'shared_nash_private_own_city'

def exact_nash_accumulated_update(*, model, cpu_batches: List[Dict[str, Any]], device: torch.device, nash, optimizer, scheduler, trainable_params, max_grad_norm: float) -> ExactNashUpdateReport:
    if not cpu_batches:
        raise ValueError('cpu_batches cannot be empty')
    base = _unwrap(model)
    routing = base.assert_gradient_routing_partition()
    shared_params = list(routing['shared'])
    private_by_city = routing['private_by_city']
    trainable_params = list(trainable_params)
    micro_steps = len(cpu_batches)
    routed_ids = {id(p) for p in shared_params}
    for city in SOURCE_CITY_ORDER:
        routed_ids.update((id(p) for p in private_by_city[city]))
    if routed_ids != {id(p) for p in trainable_params}:
        raise RuntimeError("trainable_params do not match the model's routing partition")
    optimizer.zero_grad(set_to_none=True)
    _assert_no_grads(trainable_params, 'at transaction start')
    do_probe = bool(nash.should_update_now())
    rng_snapshots: List[TorchRNGSnapshot] = []
    probe_city_loss_mean = None
    nash_solves = 0
    if do_probe:
        accumulated_task_grads = None
        probe_city_loss_sum = None
        for cpu_batch in cpu_batches:
            rng_snapshots.append(capture_torch_rng(device))
            batch = copy_batch_to_device(cpu_batch, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                out = base(batch)
            task_grads = nash.local_task_gradients(out.city_losses, shared_params)
            accumulated_task_grads = nash.accumulate_task_gradients_(accumulated_task_grads, task_grads)
            current_losses = out.city_losses.detach().float()
            probe_city_loss_sum = current_losses.clone() if probe_city_loss_sum is None else probe_city_loss_sum + current_losses
            del out, task_grads, batch
        _assert_no_grads(trainable_params, 'after Nash probe pass')
        global_task_grads = nash.global_mean_accumulated_task_gradients(accumulated_task_grads, local_micro_steps=micro_steps)
        alpha = nash.weights_from_task_gradients(global_task_grads, device=device, dtype=torch.float32)
        nash_solves = 1
        del accumulated_task_grads, global_task_grads
        probe_city_loss_mean = probe_city_loss_sum / float(micro_steps)
    else:
        alpha = nash.cached_weights_and_advance(device=device, dtype=torch.float32)
    if float(alpha.min().detach().cpu()) + 1e-07 < float(nash.alpha_floor):
        raise RuntimeError(f'Applied Nash alpha violates floor: min={float(alpha.min())}, floor={nash.alpha_floor}')
    if nash.normalize_mean and abs(float(alpha.mean().detach().cpu()) - 1.0) > 1e-05:
        raise RuntimeError('Applied Nash alpha must have mean 1 after floor projection')
    shared_acc = _zeros_like_params(shared_params)
    alpha_scales = [float(x) for x in alpha.detach().float().cpu().tolist()]
    private_acc = {city: _zeros_like_params(private_by_city[city]) for city in SOURCE_CITY_ORDER}
    update_city_loss_sum = None
    shared_weighted_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    private_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    for micro_idx, cpu_batch in enumerate(cpu_batches):
        if do_probe:
            restore_torch_rng(rng_snapshots[micro_idx], device)
        batch = copy_batch_to_device(cpu_batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out = base(batch)
        for city_idx, city in enumerate(SOURCE_CITY_ORDER):
            private_params = private_by_city[city]
            requested = shared_params + private_params
            grads = torch.autograd.grad(out.city_losses[city_idx], requested, retain_graph=city_idx < len(SOURCE_CITY_ORDER) - 1, create_graph=False, allow_unused=False)
            n_shared = len(shared_params)
            _accumulate_(shared_acc, grads[:n_shared], scale=alpha_scales[city_idx])
            _accumulate_(private_acc[city], grads[n_shared:], scale=1.0)
        current_losses = out.city_losses.detach().float()
        update_city_loss_sum = current_losses.clone() if update_city_loss_sum is None else update_city_loss_sum + current_losses
        shared_weighted_loss_sum += torch.sum(alpha * current_losses)
        private_loss_sum += torch.sum(current_losses)
        del out, batch
    param_groups = [shared_params] + [private_by_city[c] for c in SOURCE_CITY_ORDER]
    grad_groups = [shared_acc] + [private_acc[c] for c in SOURCE_CITY_ORDER]
    sync_bucket_mb = float(getattr(getattr(base, 'cfg', None), 'gradient_sync_bucket_mb', 16.0))
    _global_mean_and_assign_(param_groups, grad_groups, micro_steps=micro_steps, bucket_mb=sync_bucket_mb)
    bad_grads = []
    for param_name, p in base.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            g = p.grad.detach()
            bad_grads.append({'name': param_name, 'nan': int(torch.isnan(g).sum().item()), 'posinf': int(torch.isposinf(g).sum().item()), 'neginf': int(torch.isneginf(g).sum().item())})
    if bad_grads:
        raise RuntimeError(f'Non-finite Stage-2 gradients before optimizer.step(); bad_tensors={len(bad_grads)}, first={bad_grads[:12]}')
    shared_grad_norm = torch.nn.utils.clip_grad_norm_(shared_params, max_grad_norm)
    private_grad_norm_list = []
    for city in SOURCE_CITY_ORDER:
        private_grad_norm_list.append(torch.nn.utils.clip_grad_norm_(private_by_city[city], max_grad_norm))
    private_grad_norms = torch.stack([x.detach().float().to(device) if torch.is_tensor(x) else torch.tensor(float(x), device=device) for x in private_grad_norm_list])
    grad_norm = torch.sqrt(shared_grad_norm.detach().float().square() + private_grad_norms.detach().float().square().sum())
    if not torch.isfinite(grad_norm):
        raise RuntimeError('Non-finite Stage-2 routed gradient norm before optimizer.step()')
    optimizer.step()
    optimizer_steps = 1
    scheduler_steps = 0
    if scheduler is not None:
        scheduler.step()
        scheduler_steps = 1
    optimizer.zero_grad(set_to_none=True)
    _assert_no_grads(trainable_params, 'after optimizer update and zero_grad')
    update_city_loss_mean = update_city_loss_sum / float(micro_steps)
    if probe_city_loss_mean is None:
        probe_city_loss_mean = update_city_loss_mean.detach().clone()
        loss_diff = 0.0
    else:
        loss_diff = float(torch.max(torch.abs(update_city_loss_mean - probe_city_loss_mean)).detach().cpu())
    return ExactNashUpdateReport(alpha=alpha.detach(), probe_city_losses_local_mean=probe_city_loss_mean.detach(), update_city_losses_local_mean=update_city_loss_mean.detach(), weighted_loss_local_mean=(shared_weighted_loss_sum / float(micro_steps)).detach(), private_loss_local_mean=(private_loss_sum / float(micro_steps)).detach(), grad_norm=grad_norm.detach(), shared_grad_norm=shared_grad_norm.detach().float(), private_grad_norms=private_grad_norms.detach(), micro_steps=micro_steps, nash_solves=nash_solves, used_nash_probe=do_probe, optimizer_steps=optimizer_steps, scheduler_steps=scheduler_steps, max_probe_update_city_loss_abs_diff=loss_diff)
