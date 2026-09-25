import torch
import torch.nn as nn
from .config import CITY_SHAPES, SOURCE_CITY_ORDER

class SpatialRefineBlock(nn.Module):

    def __init__(self, dim: int, mlp_ratio: float=2.0, dropout: float=0.0):
        super().__init__()
        hidden = max(dim, int(dim * mlp_ratio))
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Sequential(nn.Conv2d(dim, hidden, kernel_size=1), nn.GELU(), nn.Dropout2d(dropout), nn.Conv2d(hidden, dim, kernel_size=1), nn.Dropout2d(dropout))
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        residual = x
        y = self.norm(x).permute(0, 3, 1, 2).contiguous()
        y = self.depthwise(y)
        y = self.pointwise(y).permute(0, 2, 3, 1).contiguous()
        return residual + self.scale * y

class CityDenseHorizonHead(nn.Module):

    def __init__(self, *, city: str, dim: int, vlm_dim: int, heads: int, forecast_len: int, history_len: int, spatial_depth: int, mlp_ratio: float, dropout: float, temporal_scale_init: float, raw_history_init_logit: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f'decoder dim={dim} must be divisible by heads={heads}')
        self.city = city
        self.dim = int(dim)
        self.forecast_len = int(forecast_len)
        self.history_len = int(history_len)
        self.h, self.w = CITY_SHAPES[city]
        self.raw_history_logits = nn.Parameter(torch.zeros(forecast_len, 2, history_len))
        with torch.no_grad():
            self.raw_history_logits[..., -1] = float(raw_history_init_logit)
        self.temporal_norm = nn.LayerNorm(dim)
        self.horizon_queries = nn.Parameter(torch.zeros(1, forecast_len, dim))
        nn.init.trunc_normal_(self.horizon_queries, std=0.02)
        self.city_query_proj = nn.Linear(dim, dim, bias=False)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.temporal_scale = nn.Parameter(torch.tensor(float(temporal_scale_init)))
        self.temporal_out_norm = nn.LayerNorm(dim)
        self.semantic_proj = nn.Linear(vlm_dim, dim)
        self.city_cond_proj = nn.Linear(dim, dim)
        self.horizon_cond = nn.Parameter(torch.zeros(1, forecast_len, dim))
        nn.init.trunc_normal_(self.horizon_cond, std=0.02)
        self.cond_norm = nn.LayerNorm(dim)
        self.film = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.normal_(self.film[-1].weight, std=0.001)
        nn.init.zeros_(self.film[-1].bias)
        self.spatial_refine = nn.ModuleList([SpatialRefineBlock(dim, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(int(spatial_depth))])
        self.out_norm = nn.LayerNorm(dim)
        self.delta_head = nn.Conv2d(dim, 2, kernel_size=1)
        nn.init.normal_(self.delta_head.weight, std=0.001)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, dense_history, city_context, semantic_context, x_hist):
        if dense_history.ndim != 5:
            raise ValueError(f'{self.city}: dense history must be [B,T,H,W,D]')
        b, t, h, w, d = dense_history.shape
        if (h, w, d) != (self.h, self.w, self.dim):
            raise ValueError(f'{self.city}: bad dense shape {tuple(dense_history.shape)}, expected [B,T,{self.h},{self.w},{self.dim}]')
        if x_hist.shape != (b, t, 2, h, w):
            raise ValueError(f'{self.city}: x_hist shape mismatch {tuple(x_hist.shape)}')
        kv = dense_history.permute(0, 2, 3, 1, 4).contiguous().view(b * h * w, t, d)
        kv = self.temporal_norm(kv)
        q0 = self.horizon_queries + self.city_query_proj(city_context).unsqueeze(1)
        q = q0[:, None, None, :, :].expand(b, h, w, self.forecast_len, d)
        q = q.contiguous().view(b * h * w, self.forecast_len, d)
        temporal, _ = self.temporal_attn(q, kv, kv, need_weights=False)
        last_feat = dense_history[:, -1]
        temporal = temporal.view(b, h, w, self.forecast_len, d).permute(0, 3, 1, 2, 4)
        feat = last_feat.unsqueeze(1) + self.temporal_scale * temporal
        feat = self.temporal_out_norm(feat)
        cond = self.semantic_proj(semantic_context).unsqueeze(1) + self.city_cond_proj(city_context).unsqueeze(1) + self.horizon_cond
        gamma, beta = self.film(self.cond_norm(cond)).chunk(2, dim=-1)
        gamma = 0.5 * torch.tanh(gamma)
        feat = feat * (1.0 + gamma[:, :, None, None, :]) + beta[:, :, None, None, :]
        y = feat.contiguous().view(b * self.forecast_len, h, w, d)
        for block in self.spatial_refine:
            y = block(y)
        y = self.out_norm(y).permute(0, 3, 1, 2).contiguous()
        delta = self.delta_head(y).view(b, self.forecast_len, 2, h, w)
        history_weights = torch.softmax(self.raw_history_logits, dim=-1)
        base = torch.einsum('btchw,kct->bkchw', x_hist, history_weights)
        return base + delta

class MultiCityRegressionHeads(nn.Module):

    def __init__(self, vlm_dim, cfg):
        super().__init__()
        self.semantic_skip_scale = nn.Parameter(torch.tensor(float(cfg.semantic_skip_init)))
        self.semantic_norm = nn.LayerNorm(vlm_dim)
        self.heads = nn.ModuleDict({city.replace('-', '_'): CityDenseHorizonHead(city=city, dim=cfg.encoder_dim, vlm_dim=vlm_dim, heads=cfg.decoder_heads, forecast_len=cfg.forecast_len, history_len=cfg.history_len, spatial_depth=cfg.decoder_spatial_depth, mlp_ratio=cfg.decoder_mlp_ratio, dropout=cfg.decoder_dropout, temporal_scale_init=cfg.temporal_readout_scale_init, raw_history_init_logit=cfg.raw_history_init_logit) for city in SOURCE_CITY_ORDER})

    @staticmethod
    def _key(city):
        return city.replace('-', '_')

    def semantic_context(self, r_hidden, z):
        scale = torch.tanh(self.semantic_skip_scale)
        return self.semantic_norm(r_hidden + scale * z)

    def forward(self, r_hidden, z, dense_maps, city_tokens, city_inputs):
        semantic = self.semantic_context(r_hidden, z)
        out = {}
        for idx, city in enumerate(SOURCE_CITY_ORDER):
            out[city] = self.heads[self._key(city)](dense_history=dense_maps[city], city_context=city_tokens[:, idx], semantic_context=semantic, x_hist=city_inputs[city])
        return out
