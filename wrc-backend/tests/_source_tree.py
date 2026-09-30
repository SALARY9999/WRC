"""Read a production method's AST from its source file, without importing it.

``inspect.getsource(app.PipelineRunner.run)`` and a parse of ``pipeline.py``
yield the same tree, but the first reaches it by importing ``app`` - and
therefore ``cv2``, ``torch`` and the entire camera stack.  Three test modules
assert on the *shape* of ``PipelineRunner``'s code rather than its behaviour, and
for those the import is pure cost plus an import side effect (see
``_cv2_isolation.py`` for the side effects this suite has already been bitten
by).

Parsing the file is not a weaker check: it reads the same bytes the interpreter
would have compiled, and ``pathlib`` resolving ``__file__`` means it cannot pick
up a stale copy from elsewhere on ``sys.path``.
"""

from __future__ import annotations

import ast
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def pipeline_method_tree(method: str, class_name: str = "PipelineRunner") -> ast.FunctionDef:
    """The AST of ``<class_name>.<method>`` in ``pipeline.py``."""
    path = BACKEND_ROOT / "pipeline.py"
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method:
                    return child
    raise AssertionError(f"{class_name}.{method} is gone from {path}")


def pipeline_result_tree() -> ast.FunctionDef:
    """The AST of ``PipelineRunner._handle_result`` - everything after a model call.

    The takeover decision, the latency crop, the local-to-WGS84 transform, the
    send and the inference package all live in that one method, and the guards
    in ``test_waypoint_trimming``/``test_replay_tool`` assert on their order
    because the order is the safety property.

    They used to read ``run``.  There are two ways to obtain a model result now
    - inline, and a worker through the pool - and both funnel through this
    method so the two cannot drift apart, which means ``run`` is a two-line
    dispatcher and the sequence moved.  Retargeting the guards here rather than
    in each test file keeps "where does the post-inference code live" a single
    answer: if it moves again, one line changes.
    """
    return pipeline_method_tree("_handle_result")
