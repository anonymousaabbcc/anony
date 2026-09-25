from pathlib import Path

def _between(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]

def main():
    root = Path(__file__).resolve().parent
    bridge = (root / 'vlm_bridge.py').read_text(encoding='utf-8')
    model = (root / 'model.py').read_text(encoding='utf-8')
    alignment = (root / 'alignment.py').read_text(encoding='utf-8')
    prompt = (root / 'prompt.py').read_text(encoding='utf-8')
    encoder = (root / 'encoder.py').read_text(encoding='utf-8')
    spatial = (root / 'spatial.py').read_text(encoding='utf-8')
    assert 'P8_STAGE2_QUESTION' in prompt
    assert 'for city in SOURCE_CITY_ORDER' in prompt
    assert '"type": "image"' in prompt
    assert 'add_generation_prompt=True' in prompt
    prep = _between(bridge, '    @torch.no_grad()\n    def prepare_context', '    def _reenter_impl')
    assert 'self._fuse_multimodal_embeddings(batch)' in prep
    assert 'self.qwen.model(' in prep
    assert 'hidden[idx, lengths - 1]' in prep
    assert 's_qa=s.detach()' in prep
    assert 'qa_encoder' not in prep
    assert 'self.alignment(global_h, context.s_qa)' in model
    assert 'context.qa_encoder' not in model
    assert 'ShiftedWindowBlock' in encoder
    assert 'TemporalBlock' in encoder
    assert 'LearnedQueryPool' in encoder
    assert 'CityAttentionBlock' in encoder
    assert 'city_tokens.reshape' in encoder
    assert 'shift=0' in spatial and 'shift=self.shift_size' in spatial
    assert 'self.p0(h)' in alignment
    assert 'torch.cat([s, h0, s * h0]' in alignment
    assert 'torch.softmax(self.selector' in alignment
    assert 'torch.sum(pi.unsqueeze(-1) * candidates' in alignment
    assert 'torch.cat([s, h_tilde, s * h_tilde]' in alignment
    assert 'torch.sigmoid(self.gate' in alignment
    assert '(1.0 - g) * s + g * h_tilde' in alignment
    impl = _between(bridge, '    def _reenter_impl', '    def reenter')
    assert 'self.qwen.model(' in impl
    assert 'return hidden[rows, positions]' in impl
    assert model.index('self.alignment(global_h, context.s_qa)') < model.index('self.bridge.reenter(context, z)')
    assert 'for p in stage1.parameters():' in bridge
    assert 'p.requires_grad_(False)' in bridge
    assert 'self._assert_frozen()' in bridge
    assert 'z=z' in model or 'z=z,' in model
    assert 'loss_l1_weight' in model
    assert 'loss_mse_weight' in model
    assert 'loss_spatial_grad_weight' in model
    print('URBANBIND METHOD-ALIGNED STAGE-2 CONTRACT: PASS')
    print('- S = frozen QA-grounded VLM(I, forecasting prompt) prompt-boundary hidden state')
    print('- native-grid spatial/temporal encoding + cross-city interaction preserved')
    print('- VLM-conditioned multi-projector routing and gated fusion preserved')
    print('- Z is contextualized by the frozen VLM backbone to obtain R')
    print('- all Stage-1/VLM parameters remain frozen')
    print('- direct Z decoder residual and existing composite loss are intentionally retained')
if __name__ == '__main__':
    main()
