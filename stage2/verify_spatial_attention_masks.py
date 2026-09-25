import argparse
import json
import math
import torch
from .config import CITY_SHAPES
from .spatial import WindowSelfAttention, _padding_key_mask, _shift_mask

def _combined_mask(h0, w0, ws, shift, device):
    hp = math.ceil(h0 / ws) * ws
    wp = math.ceil(w0 / ws) * ws
    sm = _shift_mask(hp, wp, ws, shift, device)
    pm = _padding_key_mask(h0, w0, hp, wp, ws, shift, device)
    if sm is None:
        mask = pm
    elif pm is None:
        mask = sm
    else:
        mask = sm + pm
    return (mask, hp, wp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--window-size', type=int, default=4)
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    rows = {}
    ok = True
    for city, (h0, w0) in CITY_SHAPES.items():
        city_rows = {}
        for shift in (0, args.window_size // 2):
            mask, hp, wp = _combined_mask(h0, w0, args.window_size, shift, device)
            if mask is None:
                all_masked = 0
                nwin = hp // args.window_size * (wp // args.window_size)
            else:
                all_masked = int(torch.isneginf(mask).all(dim=-1).sum().item())
                nwin = int(mask.shape[0])
            city_rows[f'shift_{shift}'] = {'padded_hw': [hp, wp], 'windows': nwin, 'all_masked_query_rows': all_masked}
            ok = ok and all_masked == 0
            n = args.window_size * args.window_size
            x = torch.randn(nwin, n, args.dim, device=device, requires_grad=True)
            attn = WindowSelfAttention(args.dim, args.heads, args.window_size).to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                y = attn(x, mask)
                loss = y.float().square().mean()
            loss.backward()
            finite_input = bool(torch.isfinite(x.grad).all().item())
            finite_params = all((p.grad is None or torch.isfinite(p.grad).all().item() for p in attn.parameters()))
            city_rows[f'shift_{shift}']['sdpa_input_grad_finite'] = finite_input
            city_rows[f'shift_{shift}']['sdpa_param_grads_finite'] = finite_params
            ok = ok and finite_input and finite_params
        rows[city] = city_rows
    print('STAGE-2 SPATIAL MASK NUMERICS:', 'PASS' if ok else 'FAIL')
    print(json.dumps(rows, indent=2))
    if not ok:
        raise SystemExit(2)
if __name__ == '__main__':
    main()
