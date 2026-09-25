import torch
import torch.nn as nn

class ResidualProjector(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim))
        self.rho = nn.Parameter(torch.tensor(0.1))

    def forward(self, base, normalized):
        return base + self.rho * self.net(normalized)

class DynamicMultiProjectorBond(nn.Module):

    def __init__(self, input_dim, vlm_dim, num_projectors, hidden_dim, dropout=0.0):
        super().__init__()
        self.vlm_dim = int(vlm_dim)
        self.num_projectors = int(num_projectors)
        self.p0 = nn.Linear(input_dim, vlm_dim)
        self.h0_norm = nn.LayerNorm(vlm_dim)
        self.projectors = nn.ModuleList([ResidualProjector(vlm_dim, hidden_dim, dropout) for _ in range(num_projectors)])
        fusion_dim = vlm_dim * 3
        self.selector = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_projectors))
        self.gate = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vlm_dim))
        self.out_norm = nn.LayerNorm(vlm_dim)

    def forward(self, h, s):
        if h.ndim != 2 or s.ndim != 2:
            raise ValueError(f'Expected h/s rank-2, got h={tuple(h.shape)}, s={tuple(s.shape)}')
        if s.shape[-1] != self.vlm_dim:
            raise ValueError(f'Expected QA-grounded VLM representation dim {self.vlm_dim}, got {s.shape[-1]}')
        h0 = self.p0(h)
        hn = self.h0_norm(h0)
        candidates = torch.stack([p(h0, hn) for p in self.projectors], dim=1)
        selector_feat = torch.cat([s, h0, s * h0], dim=-1)
        pi = torch.softmax(self.selector(selector_feat), dim=-1)
        h_tilde = torch.sum(pi.unsqueeze(-1) * candidates, dim=1)
        gate_feat = torch.cat([s, h_tilde, s * h_tilde], dim=-1)
        g = torch.sigmoid(self.gate(gate_feat))
        z = self.out_norm((1.0 - g) * s + g * h_tilde)
        return (z, pi, g)

class DirectConcatSemanticBond(nn.Module):

    def __init__(self, input_dim: int, vlm_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.vlm_dim = int(vlm_dim)
        self.norm = nn.LayerNorm(self.input_dim + self.vlm_dim)
        self.proj = nn.Linear(self.input_dim + self.vlm_dim, self.vlm_dim)

    def forward(self, global_h: torch.Tensor, s_qa: torch.Tensor):
        if global_h.ndim != 2 or s_qa.ndim != 2:
            raise ValueError('DirectConcatSemanticBond expects [B,D] tensors')
        if global_h.shape[0] != s_qa.shape[0]:
            raise ValueError('H_MC/S_QA batch mismatch')
        fused = torch.cat([s_qa, global_h.to(dtype=s_qa.dtype)], dim=-1)
        z = self.proj(self.norm(fused))
        pi = torch.ones((z.shape[0], 1), device=z.device, dtype=z.dtype)
        gates = torch.zeros_like(z)
        return (z, pi, gates)
