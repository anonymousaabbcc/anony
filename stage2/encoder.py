import torch
import torch.nn as nn
from .config import CITY_SHAPES, SOURCE_CITY_ORDER
from .spatial import ShiftedWindowBlock

class TemporalBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        q = self.norm1(x)
        y, _ = self.attn(q, q, q, need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))

class LearnedQueryPool(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        b = tokens.shape[0]
        q = self.query.expand(b, -1, -1)
        kv = self.norm(tokens)
        y, _ = self.attn(q, kv, kv, need_weights=False)
        return self.out_norm(q + y)[:, 0]

class CitySpecificEncoder(nn.Module):

    def __init__(self, city: str, dim: int, heads: int, window_size: int, spatial_depth: int, temporal_depth: int, history_len: int, mlp_ratio: float, dropout: float):
        super().__init__()
        h, w = CITY_SHAPES[city]
        self.city = city
        self.h = h
        self.w = w
        self.input_proj = nn.Conv2d(2, dim, kernel_size=1)
        self.pos2d = nn.Parameter(torch.zeros(1, h, w, dim))
        nn.init.trunc_normal_(self.pos2d, std=0.02)
        self.spatial = nn.ModuleList([ShiftedWindowBlock(dim, heads, window_size, mlp_ratio, dropout) for _ in range(spatial_depth)])
        self.temporal_pos = nn.Parameter(torch.zeros(1, history_len, dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.temporal = nn.ModuleList([TemporalBlock(dim, heads, mlp_ratio, dropout) for _ in range(temporal_depth)])
        self.pool = LearnedQueryPool(dim, heads, dropout)

    def forward(self, x):
        b, t, c, h, w = x.shape
        if (h, w) != (self.h, self.w) or c != 2:
            raise ValueError(f'{self.city}: bad input {tuple(x.shape)}')
        y = self.input_proj(x.view(b * t, c, h, w)).permute(0, 2, 3, 1)
        y = y + self.pos2d
        for block in self.spatial:
            y = block(y)
        d = y.shape[-1]
        y = y.view(b, t, h, w, d)
        y = y.permute(0, 2, 3, 1, 4).contiguous().view(b * h * w, t, d)
        if t != self.temporal_pos.shape[1]:
            raise ValueError(f'{self.city}: temporal length {t} != configured {self.temporal_pos.shape[1]}')
        y = y + self.temporal_pos
        for block in self.temporal:
            y = block(y)
        y = y.view(b, h, w, t, d).permute(0, 3, 1, 2, 4).contiguous()
        tokens = y.view(b, t * h * w, d)
        return (self.pool(tokens), y)

class CityAttentionBlock(nn.Module):

    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim * 2)
        hidden = int(dim * mlp_ratio)
        self.combine = nn.Sequential(nn.Linear(dim * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, city_tokens):
        q = self.norm1(city_tokens)
        cross, _ = self.attn(q, q, q, need_weights=False)
        delta = self.combine(self.norm2(torch.cat([city_tokens, cross], dim=-1)))
        return city_tokens + delta

class MultiCityEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.city_encoders = nn.ModuleDict({city.replace('-', '_'): CitySpecificEncoder(city=city, dim=cfg.encoder_dim, heads=cfg.encoder_heads, window_size=cfg.window_size, spatial_depth=cfg.spatial_depth, temporal_depth=cfg.temporal_depth, history_len=cfg.history_len, mlp_ratio=cfg.encoder_mlp_ratio, dropout=cfg.encoder_dropout) for city in SOURCE_CITY_ORDER})
        self.city_blocks = nn.ModuleList([CityAttentionBlock(cfg.encoder_dim, cfg.city_attention_heads, cfg.encoder_mlp_ratio, cfg.encoder_dropout) for _ in range(cfg.city_attention_depth)])

    @staticmethod
    def _key(city):
        return city.replace('-', '_')

    def forward(self, city_inputs):
        e = []
        maps = {}
        for city in SOURCE_CITY_ORDER:
            token, token_map = self.city_encoders[self._key(city)](city_inputs[city])
            e.append(token)
            maps[city] = token_map
        city_tokens = torch.stack(e, dim=1)
        for block in self.city_blocks:
            city_tokens = block(city_tokens)
        global_h = city_tokens.reshape(city_tokens.shape[0], -1)
        return (global_h, city_tokens, maps)

    def bargaining_parameters(self):
        params = []
        for block in self.city_blocks:
            params.extend((p for p in block.parameters() if p.requires_grad))
        return params
