from pathlib import Path
from .config import SOURCE_CITY_ORDER

def count_params(module, trainable_only=False):
    return sum((p.numel() for p in module.parameters() if p.requires_grad or not trainable_only))

def architecture_note(model, cfg, extra=None):
    m = model.module if hasattr(model, 'module') else model
    groups = m.trainable_parameter_groups()
    lines = ['URBANBIND STAGE-2 -- CONTROLLED ABLATION FRAMEWORK', '=================================================', f'Ablation: {cfg.ablation_name}', f'Stage-1 source: {cfg.stage1_source}', f'VLM--dynamics fusion mode: {cfg.semantic_mode}', f'Shared weighting mode: {cfg.weighting_mode}', f'Stage-1 checkpoint dependency: {cfg.stage1_checkpoint}', f'Stage-1 trainable params: {m.bridge.trainable_parameter_count}', f'Stage-1 VLM hidden size: {m.bridge.hidden_size}', 'Source cities: ' + ' -> '.join(SOURCE_CITY_ORDER), f'Protocol: {cfg.history_len} historical hours -> direct {cfg.forecast_len}-step forecast', 'City windows remain independent; no artificial cross-city timestamp alignment', f'Encoder: dim={cfg.encoder_dim}, spatial_depth={cfg.spatial_depth}, temporal_depth={cfg.temporal_depth}', f'Cross-city attention depth: {cfg.city_attention_depth}', 'Dense [B,T,H,W,D] native-grid path continues to the city-specific decoder', f'VLM-conditioned fusion: {cfg.semantic_mode}', f'Dense decoder: {cfg.forecast_len} horizon queries, spatial_refine_depth={cfg.decoder_spatial_depth}', f'Loss = {cfg.loss_l1_weight:g}*L1 + {cfg.loss_mse_weight:g}*MSE + {cfg.loss_spatial_grad_weight:g}*SpatialGradientL1', f'Shared weighting: {cfg.weighting_mode}', 'Private routing: Encoder_c + Head_c receive ONLY L_c with coefficient 1', 'Gradient clipping is route-separated for shared and each city-private branch', 'Numerical normalization: exact source-train-only city x channel statistics', 'Checkpoint selection: validation raw macro MAE only', f'Schedule: max_epochs={cfg.max_epochs}, patience={cfg.early_stopping_patience}, warmup_ratio={cfg.warmup_ratio}', f'LRs: encoder={cfg.encoder_lr:g}, alignment={cfg.alignment_lr:g}, head={cfg.head_lr:g}', f"Trainable encoder params: {sum((p.numel() for p in groups['encoder'])):,}", f"Trainable alignment params: {sum((p.numel() for p in groups['alignment'])):,}", f"Trainable head params: {sum((p.numel() for p in groups['heads'])):,}", f'Shared routed params: {sum((p.numel() for p in m.bargaining_parameters())):,}', 'Private routed params by city: ' + ', '.join((f'{city}={sum((p.numel() for p in params)):,}' for city, params in m.private_parameters_by_city().items()))]
    if extra:
        lines.extend((str(x) for x in extra))
    return '\n'.join(lines) + '\n'

def save_note(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
