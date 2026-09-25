from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .alignment import DirectConcatSemanticBond, DynamicMultiProjectorBond
from .config import SOURCE_CITY_ORDER
from .encoder import MultiCityEncoder
from .heads import MultiCityRegressionHeads
from .vlm_bridge import FrozenStage1Bridge

@dataclass
class Stage2Output:
    city_losses: torch.Tensor
    predictions: Dict[str, torch.Tensor]
    loss_components: Dict[str, torch.Tensor]
    s_qa: torch.Tensor
    city_tokens: torch.Tensor
    projector_weights: torch.Tensor
    gates: torch.Tensor
    z: torch.Tensor
    r_hidden: torch.Tensor

def _spatial_gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 5 or target.shape != pred.shape:
        raise ValueError('spatial-gradient loss expects matching [B,K,2,H,W] tensors')
    loss = pred.new_zeros(())
    terms = 0
    if pred.shape[-2] > 1:
        p = pred[..., 1:, :] - pred[..., :-1, :]
        t = target[..., 1:, :] - target[..., :-1, :]
        loss = loss + F.l1_loss(p.float(), t.float(), reduction='mean')
        terms += 1
    if pred.shape[-1] > 1:
        p = pred[..., :, 1:] - pred[..., :, :-1]
        t = target[..., :, 1:] - target[..., :, :-1]
        loss = loss + F.l1_loss(p.float(), t.float(), reduction='mean')
        terms += 1
    return loss / max(1, terms)

class Stage2UrbanForecaster(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.bridge = FrozenStage1Bridge(cfg, device=device)
        self.multi_city = MultiCityEncoder(cfg)
        global_dim = len(SOURCE_CITY_ORDER) * cfg.encoder_dim
        if cfg.semantic_mode == 'dsr_reentry':
            self.alignment = DynamicMultiProjectorBond(input_dim=global_dim, vlm_dim=self.bridge.hidden_size, num_projectors=cfg.num_residual_projectors, hidden_dim=cfg.projector_hidden_dim, dropout=cfg.projector_dropout)
        elif cfg.semantic_mode == 'direct_concat':
            self.alignment = DirectConcatSemanticBond(input_dim=global_dim, vlm_dim=self.bridge.hidden_size)
        else:
            raise ValueError(f'Unknown semantic_mode={cfg.semantic_mode!r}')
        self.regression = MultiCityRegressionHeads(vlm_dim=self.bridge.hidden_size, cfg=cfg)
        self.to(device)
        self.bridge._assert_frozen()
        self.assert_gradient_routing_partition()

    def train(self, mode: bool=True):
        super().train(mode)
        self.bridge.stage1.eval()
        self.bridge.qwen.eval()
        return self

    def bargaining_parameters(self):
        params = []
        params.extend(self.multi_city.bargaining_parameters())
        params.extend((p for p in self.alignment.parameters() if p.requires_grad))
        params.extend((p for p in self.regression.semantic_norm.parameters() if p.requires_grad))
        if self.regression.semantic_skip_scale.requires_grad:
            params.append(self.regression.semantic_skip_scale)
        return params

    def private_parameters_by_city(self):
        out = {}
        for city in SOURCE_CITY_ORDER:
            key = city.replace('-', '_')
            params = []
            params.extend((p for p in self.multi_city.city_encoders[key].parameters() if p.requires_grad))
            params.extend((p for p in self.regression.heads[key].parameters() if p.requires_grad))
            out[city] = params
        return out

    def assert_gradient_routing_partition(self):
        shared = list(self.bargaining_parameters())
        private = self.private_parameters_by_city()
        routed = shared + [p for city in SOURCE_CITY_ORDER for p in private[city]]
        routed_ids = [id(p) for p in routed]
        if len(routed_ids) != len(set(routed_ids)):
            raise RuntimeError('Shared/private Stage-2 gradient-routing groups overlap')
        trainable = [p for p in self.parameters() if p.requires_grad]
        missing = set(map(id, trainable)) - set(routed_ids)
        extra = set(routed_ids) - set(map(id, trainable))
        if missing or extra:
            raise RuntimeError(f'Invalid Stage-2 routing partition: missing={len(missing)}, extra={len(extra)}')
        return {'shared': shared, 'private_by_city': private}

    def trainable_parameter_groups(self):
        city_encoder_params = [p for p in self.multi_city.city_encoders.parameters() if p.requires_grad]
        city_attention_params = [p for p in self.multi_city.city_blocks.parameters() if p.requires_grad]
        return {'encoder': city_encoder_params + city_attention_params, 'alignment': [p for p in self.alignment.parameters() if p.requires_grad], 'heads': [p for p in self.regression.parameters() if p.requires_grad]}

    def _city_forecast_loss(self, pred, target):
        l1 = F.l1_loss(pred.float(), target.float(), reduction='mean')
        mse = F.mse_loss(pred.float(), target.float(), reduction='mean')
        spatial = _spatial_gradient_l1(pred, target)
        total = self.cfg.loss_l1_weight * l1 + self.cfg.loss_mse_weight * mse + self.cfg.loss_spatial_grad_weight * spatial
        return (total, l1, mse, spatial)

    def forward(self, batch):
        city_inputs = {city: batch['cities'][city]['x'] for city in SOURCE_CITY_ORDER}
        context = self.bridge.prepare_context(batch)
        global_h, city_tokens, maps = self.multi_city(city_inputs)
        z, pi, gates = self.alignment(global_h, context.s_qa)
        if self.cfg.semantic_mode == 'dsr_reentry':
            r = self.bridge.reenter(context, z)
        elif self.cfg.semantic_mode == 'direct_concat':
            r = z
        else:
            raise ValueError(f'Unknown semantic_mode={self.cfg.semantic_mode!r}')
        preds = self.regression(r_hidden=r, z=z, dense_maps=maps, city_tokens=city_tokens, city_inputs=city_inputs)
        losses = []
        l1s, mses, spatials = ([], [], [])
        for city in SOURCE_CITY_ORDER:
            target = batch['cities'][city]['y']
            pred = preds[city]
            if pred.shape != target.shape:
                raise RuntimeError(f'{city}: prediction {tuple(pred.shape)} != target {tuple(target.shape)}')
            total, l1, mse, spatial = self._city_forecast_loss(pred, target)
            losses.append(total)
            l1s.append(l1)
            mses.append(mse)
            spatials.append(spatial)
        return Stage2Output(city_losses=torch.stack(losses), predictions=preds, loss_components={'l1': torch.stack(l1s), 'mse': torch.stack(mses), 'spatial_grad_l1': torch.stack(spatials)}, s_qa=context.s_qa, city_tokens=city_tokens, projector_weights=pi, gates=gates, z=z, r_hidden=r)
