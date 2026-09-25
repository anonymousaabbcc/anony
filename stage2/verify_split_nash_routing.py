import copy
from types import SimpleNamespace
import torch
import torch.nn as nn
from .config import SOURCE_CITY_ORDER
from .exact_accum import exact_nash_accumulated_update
from .routing import project_nash_alpha

class CrossCoupledTiny(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([nn.Linear(3, 2) for _ in range(3)])
        self.shared = nn.Sequential(nn.Linear(6, 4), nn.Tanh(), nn.Linear(4, 2))
        self.heads = nn.ModuleList([nn.Linear(4, 1) for _ in range(3)])

    def bargaining_parameters(self):
        return list(self.shared.parameters())

    def private_parameters_by_city(self):
        return {city: list(self.encoders[i].parameters()) + list(self.heads[i].parameters()) for i, city in enumerate(SOURCE_CITY_ORDER)}

    def assert_gradient_routing_partition(self):
        shared = self.bargaining_parameters()
        private = self.private_parameters_by_city()
        routed = shared + [p for c in SOURCE_CITY_ORDER for p in private[c]]
        trainable = [p for p in self.parameters() if p.requires_grad]
        ids = [id(p) for p in routed]
        if len(ids) != len(set(ids)):
            raise RuntimeError('routing overlap')
        if set(ids) != {id(p) for p in trainable}:
            raise RuntimeError('routing partition mismatch')
        return {'shared': shared, 'private_by_city': private}

    def forward(self, batch):
        toks = [enc(batch['x'][:, i]) for i, enc in enumerate(self.encoders)]
        cross = self.shared(torch.cat(toks, dim=-1))
        losses = []
        for i, head in enumerate(self.heads):
            pred = head(torch.cat([toks[i], cross], dim=-1)).squeeze(-1)
            err = pred - batch['y'][:, i]
            losses.append(err.square().mean() + 0.1 * err.abs().mean())
        return SimpleNamespace(city_losses=torch.stack(losses))

class FixedNash:

    def __init__(self, alpha):
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha_floor = 0.05
        self.normalize_mean = True
        self.step = 0

    def should_update_now(self):
        return False

    def cached_weights_and_advance(self, device, dtype=torch.float32):
        self.step += 1
        return self.alpha.to(device=device, dtype=dtype)

def _clone_private(model):
    p = model.private_parameters_by_city()
    return {c: [x.detach().clone() for x in p[c]] for c in SOURCE_CITY_ORDER}

def _changed(before, after_params):
    return [b - p.detach() for b, p in zip(before, after_params)]

def main():
    torch.manual_seed(7)
    base = CrossCoupledTiny()
    model_a = copy.deepcopy(base)
    model_b = copy.deepcopy(base)
    batch = {'x': torch.randn(5, 3, 3), 'y': torch.randn(5, 3)}
    alpha_a = [2.9, 0.05, 0.05]
    alpha_b = [0.05, 0.05, 2.9]
    before_a = _clone_private(model_a)
    before_b = _clone_private(model_b)
    shared_a0 = [p.detach().clone() for p in model_a.bargaining_parameters()]
    shared_b0 = [p.detach().clone() for p in model_b.bargaining_parameters()]
    for model, alpha in ((model_a, alpha_a), (model_b, alpha_b)):
        opt = torch.optim.SGD(model.parameters(), lr=0.03)
        tx = exact_nash_accumulated_update(model=model, cpu_batches=[batch], device=torch.device('cpu'), nash=FixedNash(alpha), optimizer=opt, scheduler=None, trainable_params=[p for p in model.parameters() if p.requires_grad], max_grad_norm=0.05)
        if tx.routing_mode != 'shared_nash_private_own_city':
            raise RuntimeError('wrong routing mode')
    pa = model_a.private_parameters_by_city()
    pb = model_b.private_parameters_by_city()
    max_private_delta_diff = 0.0
    for city in SOURCE_CITY_ORDER:
        da = _changed(before_a[city], pa[city])
        db = _changed(before_b[city], pb[city])
        for xa, xb in zip(da, db):
            max_private_delta_diff = max(max_private_delta_diff, float((xa - xb).abs().max().item()))
    if max_private_delta_diff > 1e-07:
        raise RuntimeError(f'Private updates changed with Nash alpha: max diff={max_private_delta_diff}')
    shared_delta_diff = 0.0
    for a0, b0, pa_now, pb_now in zip(shared_a0, shared_b0, model_a.bargaining_parameters(), model_b.bargaining_parameters()):
        da = a0 - pa_now.detach()
        db = b0 - pb_now.detach()
        shared_delta_diff = max(shared_delta_diff, float((da - db).abs().max().item()))
    if shared_delta_diff <= 1e-08:
        raise RuntimeError('Shared update did not respond to Nash alpha')
    projected = project_nash_alpha(torch.tensor([2.95, 0.04, 0.01]), normalize_mean=True, alpha_floor=0.05)
    if float(projected.min()) < 0.05 - 1e-07:
        raise RuntimeError(f'Nash floor failed: {projected.tolist()}')
    if abs(float(projected.mean()) - 1.0) > 1e-06:
        raise RuntimeError(f'Nash mean-one failed: {projected.tolist()}')
    print('SPLIT-NASH ROUTING REGRESSION: PASS')
    print('private update max difference across opposite Nash vectors:', max_private_delta_diff)
    print('shared update max difference across opposite Nash vectors:', shared_delta_diff)
    print('floor projection:', projected.tolist())
if __name__ == '__main__':
    main()
