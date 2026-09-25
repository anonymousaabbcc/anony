import torch

def project_nash_alpha(alpha: torch.Tensor, *, normalize_mean: bool, alpha_floor: float, eps: float=1e-10) -> torch.Tensor:
    floor = float(alpha_floor)
    if floor < 0:
        raise ValueError('alpha_floor must be >= 0')
    if normalize_mean and floor >= 1.0:
        raise ValueError('alpha_floor must be < 1 when normalize_mean=True')
    alpha = torch.nan_to_num(alpha, nan=1.0, posinf=1.0, neginf=1.0)
    alpha = alpha.clamp_min(float(eps))
    if normalize_mean:
        alpha = alpha / alpha.mean().clamp_min(float(eps))
    if floor <= 0.0:
        return alpha
    if not normalize_mean:
        return alpha.clamp_min(floor)
    n = alpha.numel()
    floor_t = alpha.new_tensor(floor)
    surplus = (alpha - floor_t).clamp_min(0.0)
    budget = alpha.new_tensor(float(n) * (1.0 - floor))
    surplus_sum = surplus.sum()
    if float(surplus_sum.detach().cpu()) <= float(eps):
        return torch.ones_like(alpha)
    return floor_t + surplus * (budget / surplus_sum.clamp_min(float(eps)))
