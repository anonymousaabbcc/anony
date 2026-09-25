from .ablations import make_ablation_config

def main():
    full = make_ablation_config('full')
    checks = []
    g = make_ablation_config('no_grounding')
    checks.append((g.stage1_source == 'base_pretrained', 'no_grounding uses base pretrained VLM'))
    checks.append((g.semantic_mode == full.semantic_mode, 'no_grounding keeps DSR'))
    checks.append((g.weighting_mode == full.weighting_mode, 'no_grounding keeps Nash'))
    st = make_ablation_config('no_st_encoder')
    checks.append((st.spatial_depth == 0 and st.temporal_depth == 0, 'no_st_encoder removes both ST encoder blocks'))
    checks.append((st.city_attention_depth == full.city_attention_depth, 'no_st_encoder keeps cross-city attention'))
    cc = make_ablation_config('no_cross_city')
    checks.append((cc.city_attention_depth == 0, 'no_cross_city removes city-token attention'))
    checks.append((cc.spatial_depth == full.spatial_depth and cc.temporal_depth == full.temporal_depth, 'no_cross_city keeps ST encoder'))
    dsr = make_ablation_config('no_dsr')
    checks.append((dsr.semantic_mode == 'direct_concat', 'no_dsr uses direct concatenation fusion'))
    checks.append((dsr.stage1_source == 'grounded', 'no_dsr keeps Stage-1 grounding'))
    nw = make_ablation_config('no_nash')
    checks.append((nw.weighting_mode == 'equal', 'no_nash uses equal shared weighting'))
    checks.append((nw.nash_alpha_floor == full.nash_alpha_floor, 'no_nash leaves unrelated config unchanged'))
    bad = [msg for ok, msg in checks if not ok]
    if bad:
        raise RuntimeError('ABLATION PROFILE VERIFY FAILED: ' + '; '.join(bad))
    print('URBANBIND ABLATION PROFILE VERIFY: PASS')
    for _, msg in checks:
        print('- ' + msg)
if __name__ == '__main__':
    main()
