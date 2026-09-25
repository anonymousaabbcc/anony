from dataclasses import replace
from .config import Stage2Config
ABLATION_PROFILES = {'full': {}, 'no_grounding': {'stage1_source': 'base_pretrained'}, 'no_st_encoder': {'spatial_depth': 0, 'temporal_depth': 0}, 'no_cross_city': {'city_attention_depth': 0}, 'no_dsr': {'semantic_mode': 'direct_concat'}, 'no_nash': {'weighting_mode': 'equal'}}

def ablation_names(include_full: bool=True):
    names = list(ABLATION_PROFILES)
    return names if include_full else [x for x in names if x != 'full']

def make_ablation_config(name: str) -> Stage2Config:
    if name not in ABLATION_PROFILES:
        raise ValueError(f"Unknown ablation {name!r}; choose from {', '.join(ABLATION_PROFILES)}")
    cfg = Stage2Config()
    changes = dict(ABLATION_PROFILES[name])
    changes['ablation_name'] = name
    return replace(cfg, **changes)

def describe_ablation(name: str) -> str:
    descriptions = {'full': 'Full UrbanBind v3.2.2 reference architecture.', 'no_grounding': 'w/o QA Grounding: frozen base Qwen2.5-VL replaces the QA-grounded Stage-1 checkpoint; Stage-2 is otherwise unchanged.', 'no_st_encoder': 'w/o ST Encoder: removes Stage-2 city-specific spatial and temporal attention blocks while retaining native-grid projection/readout.', 'no_cross_city': 'w/o Cross-City Attention: per-city tokens are directly concatenated without city-token self-attention.', 'no_dsr': 'w/o DSR: removes the VLM-conditioned multi-projector/fusion and frozen-backbone contextualization; uses direct [S_QA; H_MC] linear fusion.', 'no_nash': 'w/o Nash: shared gradients use equal alpha=[1,1,1]; private routing remains city-specific with coefficient 1.'}
    return descriptions[name]
