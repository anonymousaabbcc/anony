from __future__ import annotations
import ast
from pathlib import Path
COLLECTIVE_CALLS = {'_mean_across_ranks', '_max_across_ranks', 'peak_memory_across_ranks', 'barrier', 'save_checkpoint', 'start_wall_timer', 'stop_wall_timer'}

def _is_rank0_test(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], ast.Eq):
        return False
    a, b = (node.left, node.comparators[0])

    def is_r(x):
        return isinstance(x, ast.Name) and x.id == 'r'

    def is_zero(x):
        return isinstance(x, ast.Constant) and x.value == 0
    return is_r(a) and is_zero(b) or (is_zero(a) and is_r(b))

def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None

def main():
    path = Path(__file__).with_name('train.py')
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    bad: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):

        def __init__(self):
            self.rank0_depth = 0

        def visit_If(self, node: ast.If):
            is_rank0 = _is_rank0_test(node.test)
            if is_rank0:
                self.rank0_depth += 1
                for child in node.body:
                    self.visit(child)
                self.rank0_depth -= 1
                for child in node.orelse:
                    self.visit(child)
            else:
                self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            name = _call_name(node)
            if self.rank0_depth and name in COLLECTIVE_CALLS:
                bad.append((node.lineno, name))
            self.generic_visit(node)
    Visitor().visit(tree)
    if bad:
        raise RuntimeError(f'Rank-0-only collective call(s) found: {bad}')
    text = path.read_text(encoding='utf-8')
    required = ['private_loss_mean = _mean_across_ranks(report.private_loss_local_mean)', '"private_unweighted_loss": float(private_loss_mean.cpu())']
    missing = [x for x in required if x not in text]
    if missing:
        raise RuntimeError(f'Logging collective fix contract missing: {missing}')
    print('URBANBIND DISTRIBUTED COLLECTIVE LOGGING CONTRACT: PASS')
    print('- all logging reductions execute on every rank')
    print('- rank 0 only formats/prints already-reduced values')
if __name__ == '__main__':
    main()
