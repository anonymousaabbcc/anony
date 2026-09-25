import torch
from .alignment import DynamicMultiProjectorBond
from .config import CITY_SHAPES, SOURCE_CITY_ORDER, Stage2Config
from .encoder import MultiCityEncoder
from .heads import MultiCityRegressionHeads

def main():
    torch.manual_seed(0)
    cfg = Stage2Config(encoder_dim=48, encoder_heads=6, spatial_depth=1, temporal_depth=1, city_attention_heads=6, num_residual_projectors=2, projector_hidden_dim=64, decoder_heads=6, decoder_spatial_depth=1)
    b, d_vlm = (2, 96)
    x = {city: torch.randn(b, cfg.history_len, 2, *CITY_SHAPES[city]) for city in SOURCE_CITY_ORDER}
    enc = MultiCityEncoder(cfg)
    global_h, city_tokens, maps = enc(x)
    assert global_h.shape == (b, len(SOURCE_CITY_ORDER) * cfg.encoder_dim)
    assert city_tokens.shape == (b, len(SOURCE_CITY_ORDER), cfg.encoder_dim)
    for city in SOURCE_CITY_ORDER:
        h, w = CITY_SHAPES[city]
        assert maps[city].shape == (b, cfg.history_len, h, w, cfg.encoder_dim)
    align = DynamicMultiProjectorBond(input_dim=global_h.shape[-1], vlm_dim=d_vlm, num_projectors=cfg.num_residual_projectors, hidden_dim=cfg.projector_hidden_dim)
    s = torch.randn(b, d_vlm)
    z, pi, gate = align(global_h, s)
    assert z.shape == (b, d_vlm)
    assert pi.shape == (b, cfg.num_residual_projectors)
    assert gate.shape == (b, d_vlm)
    heads = MultiCityRegressionHeads(d_vlm, cfg)
    r = torch.randn_like(z)
    pred = heads(r, z, maps, city_tokens, x)
    for city in SOURCE_CITY_ORDER:
        assert pred[city].shape == (b, cfg.forecast_len, 2, *CITY_SHAPES[city])
    loss = sum((v.square().mean() for v in pred.values()))
    loss.backward()
    touched = sum((1 for p in list(enc.parameters()) + list(align.parameters()) + list(heads.parameters()) if p.grad is not None and torch.isfinite(p.grad).all()))
    if touched == 0:
        raise RuntimeError('No finite gradients in UrbanBind split-Nash dense-horizon core test')
    print('URBANBIND DENSE-HORIZON CORE SHAPE/AUTOGRAD: PASS')
    print('global_h:', tuple(global_h.shape))
    print('city_tokens:', tuple(city_tokens.shape))
    print('Z:', tuple(z.shape))
    for city in SOURCE_CITY_ORDER:
        print(city, 'dense', tuple(maps[city].shape), 'pred', tuple(pred[city].shape))
if __name__ == '__main__':
    main()
