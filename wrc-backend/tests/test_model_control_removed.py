"""Dependency-free regression checks; no model/GPU or vehicle I/O.

Stage 2 moved the three classes these checks assert on out of ``app.py``:
``ModelResult`` and ``AlpamayoRunner`` into ``inference/alpamayo.py``,
``PipelineRunner`` into ``pipeline.py``.  The checks now parse whichever module
owns the class, and the "no custom takeover parser" check scans every module
rather than one - a moved function must not be able to leave a second copy
behind in the file it came from.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]

#: One entry per module the AST checks below parse.  Not a search path: each
#: check names the module that is *supposed* to own the class, so a class that
#: drifts to a third home fails rather than being found anywhere.
MODULES = {
    'app': ROOT / 'app.py',
    'pipeline': ROOT / 'pipeline.py',
    'inference.alpamayo': ROOT / 'inference' / 'alpamayo.py',
}


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == name)


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _all_sources() -> list[tuple[str, str]]:
    """Every production .py under the backend, minus tests and dot-dirs."""
    out: list[tuple[str, str]] = []
    for path in sorted(ROOT.rglob('*.py')):
        rel = path.relative_to(ROOT)
        if any(part.startswith('.') or part in ('tests', '__pycache__')
               for part in rel.parts):
            continue
        out.append((str(rel).replace('\\', '/'), path.read_text(encoding='utf-8')))
    return out


class ModelControlRemovedTests(unittest.TestCase):
    def setUp(self):
        self.trees = {
            key: ast.parse(path.read_text(encoding='utf-8'))
            for key, path in MODULES.items()
        }

    def test_official_prompt_is_used_and_custom_parser_is_absent(self):
        runner = _class(self.trees['inference.alpamayo'], 'AlpamayoRunner')
        infer = _method(runner, 'infer')
        calls = [ast.unparse(n.func) for n in ast.walk(infer)
                 if isinstance(n, ast.Call)]
        self.assertIn('helper.create_message', calls)
        self.assertFalse(any('takeover' in name for name in calls))

    def test_no_module_defines_a_takeover_parser(self):
        # Repo-wide, because the earlier version of this check only looked at
        # app.py: after Stage 2 a reintroduced parser could sit in any module.
        forbidden = {'create_takeover_messages', 'parse_model_control_flag',
                     'classify_takeover_from_cot'}
        found: dict[str, set[str]] = {}
        for rel, text in _all_sources():
            tree = ast.parse(text)
            names = {n.name for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            hit = names & forbidden
            if hit:
                found[rel] = hit
        self.assertEqual(found, {}, f"a custom takeover parser came back: {found}")

    def test_model_result_has_no_control_fields(self):
        result = _class(self.trees['inference.alpamayo'], 'ModelResult')
        fields = {n.target.id for n in result.body if isinstance(n, ast.AnnAssign)}
        self.assertFalse(fields & {'takeover_flag', 'flag_parsed', 'control_flag'})
        self.assertTrue({'positions', 'rotations', 'cot'} <= fields)

    def test_independent_control_ignores_model_text(self):
        # Execute the actual tiny control functions, without importing app's GPU
        # dependencies.  These two stayed in app.py: they are the HTTP-facing
        # manual override, not pipeline state.
        tree = self.trees['app']
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('set_control_flag', 'resolve_control_flag')]
        namespace = {'_CONTROL_FLAG_OVERRIDE': None}
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0)] + functions, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), 'control_subset', 'exec'), namespace)
        settings = SimpleNamespace(control_flag=0)
        result = SimpleNamespace(cot='takeover=1 obstacle pedestrian avoid')
        self.assertEqual(namespace['resolve_control_flag'](settings, result=result), 0)
        settings.control_flag = 1
        self.assertEqual(namespace['resolve_control_flag'](settings, result=result), 1)
        namespace['set_control_flag'](0)
        self.assertEqual(namespace['resolve_control_flag'](settings, result=result), 0)
        namespace['set_control_flag'](None)
        self.assertEqual(namespace['resolve_control_flag'](settings), 1)

    def test_pipeline_uses_trajectory_decision_control(self):
        # ``_handle_result`` is where the control flag is decided for both the
        # inline and the pooled path.  ``run`` is only a dispatcher between them
        # now, so asserting on ``run`` would assert on two lines of ``if``.
        handler = _method(_class(self.trees['pipeline'], 'PipelineRunner'), '_handle_result')
        source = ast.unparse(handler)
        self.assertIn('decision = self._trajectory_control_decision(result)', source)
        self.assertIn('control = decision.control_flag', source)
        self.assertNotIn('resolve_control_flag(', source)
        self.assertNotIn('result.takeover_flag', source)

    def test_the_control_flag_is_computed_in_exactly_one_module(self):
        # The send path must have a single author.  app.py still holds the
        # manual override, which is a *input* to the decision layer, not a
        # second copy of this assignment.
        holders = [
            rel for rel, text in _all_sources()
            if 'control = decision.control_flag' in text
        ]
        self.assertEqual(holders, ['pipeline.py'], holders)

    def test_offline_harnesses_do_not_read_model_control(self):
        offline = (ROOT / 'test_backend.py').read_text(encoding='utf-8')
        self.assertNotIn('result.takeover_flag', offline)
        self.assertNotIn('result.flag_parsed', offline)


if __name__ == '__main__':
    unittest.main()
