from types import SimpleNamespace
import torch
import torch.nn as nn
from .exact_accum import exact_nash_accumulated_update
from .nash import DistributedNashMTL

class TinyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.city = nn.ModuleList([nn.Linear(4, 1) for _ in range(3)])

    def bargaining_parameters(self):
        return list(self.shared.parameters())

    def private_parameters_by_city(self):
        names = ('NYCTAXI', 'BIKECHI', 'NYC-BIKE')
        return {name: list(self.city[i].parameters()) for i, name in enumerate(names)}

    def assert_gradient_routing_partition(self):
        shared = self.bargaining_parameters()
        private = self.private_parameters_by_city()
        routed = shared + [p for name in ('NYCTAXI', 'BIKECHI', 'NYC-BIKE') for p in private[name]]
        trainable = [p for p in self.parameters() if p.requires_grad]
        if len({id(p) for p in routed}) != len(routed):
            raise RuntimeError('TinyModel routing overlap')
        if {id(p) for p in routed} != {id(p) for p in trainable}:
            raise RuntimeError('TinyModel routing partition mismatch')
        return {'shared': shared, 'private_by_city': private}

    def forward(self, batch):
        x = self.shared(batch['x'])
        losses = []
        for i, head in enumerate(self.city):
            pred = head(x).squeeze(-1)
            target = batch['y'][:, i]
            losses.append((pred - target).abs().mean() + 0.1 * (pred - target).square().mean())
        return SimpleNamespace(city_losses=torch.stack(losses))

def main():
    torch.manual_seed(0)
    device = torch.device('cpu')
    model = TinyModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    nash = DistributedNashMTL(n_tasks=3, update_weights_every=2, normalize_mean=True, optim_niter=3, solver_max_iters=50, alpha_floor=0.05)
    batches = [{'x': torch.randn(2, 4), 'y': torch.randn(2, 3)}, {'x': torch.randn(2, 4), 'y': torch.randn(2, 3)}]
    trainable = [p for p in model.parameters() if p.requires_grad]
    r1 = exact_nash_accumulated_update(model=model, cpu_batches=batches, device=device, nash=nash, optimizer=opt, scheduler=sch, trainable_params=trainable, max_grad_norm=1.0)
    r2 = exact_nash_accumulated_update(model=model, cpu_batches=batches, device=device, nash=nash, optimizer=opt, scheduler=sch, trainable_params=trainable, max_grad_norm=1.0)
    if not (r1.used_nash_probe and r1.nash_solves == 1):
        raise RuntimeError('First periodic update must refresh Nash')
    if r2.used_nash_probe or r2.nash_solves != 0:
        raise RuntimeError('Second periodic update must reuse cached Nash weights')
    if abs(float(r1.alpha.mean()) - 1.0) > 1e-05 or abs(float(r2.alpha.mean()) - 1.0) > 1e-05:
        raise RuntimeError('Mean-one Nash rescaling failed')
    if float(r1.alpha.min()) < 0.05 - 1e-07 or float(r2.alpha.min()) < 0.05 - 1e-07:
        raise RuntimeError('Nash alpha floor failed')
    if nash.step != 2:
        raise RuntimeError(f'Expected Nash step 2, got {nash.step}')
    print('PERIODIC NASH FAST CHECK: PASS')
    print('refresh alpha:', r1.alpha.tolist())
    print('cached alpha :', r2.alpha.tolist())
if __name__ == '__main__':
    main()
