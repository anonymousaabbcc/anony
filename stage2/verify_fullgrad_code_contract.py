from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    cfg_src = (root / 'config.py').read_text(encoding='utf-8')
    bridge_src = (root / 'vlm_bridge.py').read_text(encoding='utf-8')
    heads_src = (root / 'heads.py').read_text(encoding='utf-8')
    if 'stop_gradient_reentry:' in cfg_src:
        raise RuntimeError('Formal v3.2 config must not expose a stop-gradient re-entry switch')
    reenter_block = bridge_src.split('    def reenter(self, context: FrozenContext, z: torch.Tensor):', 1)[1]
    forbidden = ['with torch.no_grad()', 'z.detach()', '_reenter_impl(context, z.detach())']
    hits = [x for x in forbidden if x in reenter_block]
    if hits:
        raise RuntimeError(f'Full-gradient reenter contains forbidden stop-gradient code: {hits}')
    if 'return self._reenter_impl(context, z)' not in reenter_block:
        raise RuntimeError('Formal reenter does not call _reenter_impl(context, z) directly')
    impl_block = bridge_src.split('    def _reenter_impl(self, context: FrozenContext, z: torch.Tensor):', 1)[1]
    impl_block = impl_block.split('    def reenter(', 1)[0]
    for bad in ('z.detach()', 'hidden.detach()', 'out.last_hidden_state.detach()'):
        if bad in impl_block:
            raise RuntimeError(f'Full-gradient re-entry implementation contains {bad!r}')
    semantic_block = heads_src.split('    def semantic_context(self, r_hidden, z):', 1)[1]
    semantic_block = semantic_block.split('    def forward(', 1)[0]
    if 'r_hidden.detach' in semantic_block or 'z.detach' in semantic_block:
        raise RuntimeError('Semantic residual interface detaches R or Z')
    if 'r_hidden + scale * z' not in semantic_block:
        raise RuntimeError('Expected residual semantic interface R + scale*Z not found')
    print('FULL-GRAD SEMANTIC CODE CONTRACT: PASS')
    print('- no formal stop-gradient config switch')
    print('- reenter(Z) has no no_grad/detach')
    print('- _reenter_impl preserves Z -> FrozenQwen -> R autograd path')
    print('- semantic_context preserves both R and direct-Z gradient paths')
if __name__ == '__main__':
    main()
