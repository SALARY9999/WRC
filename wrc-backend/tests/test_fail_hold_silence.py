"""Rejected input and a failed inference must not send a release frame.

Fail-hold is the vehicle policy, pinned in ``test_planned_route_control``:
``TakeoverDecisionEngine.invalid`` keeps the last ``control_flag``.  On the
wire, ``count=0 control=0`` is a release and would drop that takeover.  Silence
leaves the vehicle on the last datagram it accepted.

The empty-trim path is the exception, and it is pinned separately: a usable
trajectory with nothing left to execute DOES send ``count=0 control=0``.
These checks read ``pipeline.py`` as text so they do not import cv2.
"""

from __future__ import annotations

import ast
import unittest

from _source_tree import pipeline_method_tree


def _sends(node: ast.AST) -> list[str]:
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and "sender.send" in ast.unparse(child.func):
            found.append(ast.unparse(child))
    return found


def _calls_invalid(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call) and ".invalid" in ast.unparse(child.func)
        for child in ast.walk(node)
    )


class _SilenceCase(unittest.TestCase):
    def assert_silent(self, node: ast.AST, *, label: str) -> None:
        self.assertEqual(
            _sends(node),
            [],
            f"{label} must not send a release frame",
        )
        self.assertTrue(_calls_invalid(node), f"{label} must record invalid()")


class RejectedInputSilenceTests(_SilenceCase):
    def test_both_loops_stay_silent_when_the_input_is_rejected(self) -> None:
        for method in ("_run_inline", "_run_multi_process"):
            tree = pipeline_method_tree(method)
            rejected = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Attribute)
                and node.test.operand.attr == "accepted"
            ]
            self.assertEqual(len(rejected), 1, method)
            self.assert_silent(rejected[0], label=f"{method} rejected input")

    def test_both_loops_stay_silent_when_the_cycle_raises(self) -> None:
        for method in ("_run_inline", "_run_multi_process"):
            tree = pipeline_method_tree(method)
            handlers = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ExceptHandler)
            ]
            self.assertEqual(len(handlers), 1, method)
            self.assert_silent(handlers[0], label=f"{method} exception")


class InvalidTrajectorySilenceTests(_SilenceCase):
    def test_an_unusable_trajectory_returns_before_any_send(self) -> None:
        tree = pipeline_method_tree("_handle_result")
        invalid = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and node.test.operand.id == "valid"
        ]
        self.assertEqual(len(invalid), 1)
        guard = invalid[0]
        self.assert_silent(guard, label="invalid trajectory")
        returns = [node for node in guard.body if isinstance(node, ast.Return)]
        self.assertTrue(returns, "the invalid trajectory must return")
        self.assertIsInstance(returns[-1].value, ast.Constant)
        self.assertIs(returns[-1].value.value, False)
