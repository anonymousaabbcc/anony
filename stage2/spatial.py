import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _pad_hw(x, window_size: int):
    b, h, w, d = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    return (x, h, w)

def _window_partition(x, window_size: int):
    b, h, w, d = x.shape
    ws = window_size
    x = x.view(b, h // ws, ws, w // ws, ws, d)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, d)

def _window_reverse(windows, window_size: int, b: int, h: int, w: int, d: int):
    ws = window_size
    x = windows.view(b, h // ws, w // ws, ws, ws, d)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, d)

def _shift_mask(hp: int, wp: int, ws: int, shift: int, device):
    if shift == 0:
        return None
    img_mask = torch.zeros((1, hp, wp, 1), device=device)
    h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    cnt = 0
    for hs in h_slices:
        for wslice in w_slices:
            img_mask[:, hs, wslice, :] = cnt
            cnt += 1
    mask_windows = _window_partition(img_mask, ws).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, 0.0)
    return attn_mask

def _padding_key_mask(h0: int, w0: int, hp: int, wp: int, ws: int, shift: int, device):
    if h0 == hp and w0 == wp:
        return None
    valid = torch.zeros((1, hp, wp, 1), device=device, dtype=torch.float32)
    valid[:, :h0, :w0, :] = 1.0
    if shift:
        valid = torch.roll(valid, shifts=(-shift, -shift), dims=(1, 2))
    win = _window_partition(valid, ws).squeeze(-1) > 0.5
    mask = torch.zeros((win.shape[0], win.shape[1], win.shape[1]), device=device)
    mask = mask.masked_fill((~win).unsqueeze(1), float('-inf'))
    idx = torch.arange(win.shape[1], device=device)
    mask[:, idx, idx] = 0.0
    return mask

class WindowSelfAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int, window_size: int, dropout: float=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = int(window_size)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def forward(self, windows: torch.Tensor, attn_mask=None):
        bw, n, d = windows.shape
        qkv = self.qkv(windows).view(bw, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        mask = None
        if attn_mask is not None:
            nwin = attn_mask.shape[0]
            if bw % nwin != 0:
                raise RuntimeError(f'window batch {bw} not divisible by mask windows {nwin}')
            repeat = bw // nwin
            mask = attn_mask.unsqueeze(0).repeat(repeat, 1, 1, 1).view(bw, 1, n, n)
            mask = mask.to(dtype=q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(bw, n, d)
        return self.proj(out)

class ShiftedWindowBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.window_size = int(window_size)
        self.shift_size = self.window_size // 2
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = WindowSelfAttention(dim, num_heads, self.window_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = WindowSelfAttention(dim, num_heads, self.window_size, dropout)
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def _attend(self, x, norm, attn, shift):
        residual = x
        x = norm(x)
        x, h0, w0 = _pad_hw(x, self.window_size)
        b, hp, wp, d = x.shape
        if shift:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
        windows = _window_partition(x, self.window_size)
        shift_mask = _shift_mask(hp, wp, self.window_size, shift, x.device)
        pad_mask = _padding_key_mask(h0, w0, hp, wp, self.window_size, shift, x.device)
        if shift_mask is None:
            mask = pad_mask
        elif pad_mask is None:
            mask = shift_mask
        else:
            mask = shift_mask + pad_mask
        windows = attn(windows, mask)
        x = _window_reverse(windows, self.window_size, b, hp, wp, d)
        if shift:
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
        x = x[:, :h0, :w0]
        return residual + x

    def forward(self, x):
        x = self._attend(x, self.norm1, self.attn1, shift=0)
        x = self._attend(x, self.norm2, self.attn2, shift=self.shift_size)
        x = x + self.mlp(self.norm3(x))
        return x
