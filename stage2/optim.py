import math
import torch

def _split_decay(named_params):
    decay, no_decay = ([], [])
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return (decay, no_decay)

def build_optimizer(model, cfg):
    m = model.module if hasattr(model, 'module') else model
    modules = [('encoder', m.multi_city, cfg.encoder_lr), ('alignment', m.alignment, cfg.alignment_lr), ('heads', m.regression, cfg.head_lr)]
    groups = []
    seen = set()
    for prefix, module, lr in modules:
        named = []
        for name, p in module.named_parameters():
            if id(p) in seen:
                raise RuntimeError(f'Duplicate optimizer parameter: {prefix}.{name}')
            seen.add(id(p))
            named.append((f'{prefix}.{name}', p))
        decay, no_decay = _split_decay(named)
        if decay:
            groups.append({'params': decay, 'lr': lr, 'weight_decay': cfg.weight_decay, 'tag': prefix + '/decay'})
        if no_decay:
            groups.append({'params': no_decay, 'lr': lr, 'weight_decay': 0.0, 'tag': prefix + '/no_decay'})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-08)

def build_cosine_scheduler(optimizer, total_updates: int, warmup_ratio: float):
    total_updates = max(1, int(total_updates))
    warmup = int(total_updates * float(warmup_ratio))

    def scale(step):
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if total_updates <= warmup:
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / float(total_updates - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return (torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale), warmup)
