"""The cv2 stub must leave the interpreter exactly as it found it.

Every other test module that needs cv2 depends on this one, so the contract is
asserted here rather than described in a comment.  It has been described in a
comment before, and the comment was wrong: a bare ``del sys.modules["cv2"]``
reads like a full undo, but ``camera.sync`` and ``pipeline`` bind cv2 to their
own globals while the stub is installed, and nothing reachable through
``sys.modules`` can undo that.  The result was that ``test_camera_sei`` reported

    AttributeError: module 'cv2' has no attribute 'resize'

- a message that names cv2, faults the wrong thing, and depends on which test
file the interpreter imported first.

Everything here runs in a child interpreter, because the property under test is
"this process is unchanged" and an in-process check would have to import the
very modules it is asserting about.

Those children are also given a machine without cv2, via
``CV2_ABSENT_PRELUDE``.  Asserting that ``cv2_stub()`` installed something is
only meaningful where cv2 cannot be imported, and the deployment server has it;
left to the machine, this file would pass on a developer box and fail on the
server.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from _cv2_isolation import CV2_ABSENT_PRELUDE

TESTS_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = TESTS_DIR.parent


def _run(body: str, prelude: bool = True) -> str:
    script = (CV2_ABSENT_PRELUDE if prelude else "") + textwrap.dedent(body)
    env = dict(os.environ, PYTHONPATH=str(BACKEND_ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(TESTS_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise AssertionError(f"child failed (exit {result.returncode}):\n{result.stderr}")
    return result.stdout.strip()


class TheCv2AbsentPreludeTests(unittest.TestCase):
    """The premise the rest of this file rests on, asserted rather than assumed.

    Every assertion below is about what ``cv2_stub()`` does on a machine without
    cv2.  If the prelude stopped working on a machine that has cv2, those
    assertions would quietly stop measuring anything - so it is checked here
    first, on whichever machine the suite happens to be running.
    """

    def test_the_prelude_makes_cv2_unimportable(self):
        self.assertEqual(
            _run(
                """
                try:
                    import cv2
                except ModuleNotFoundError as exc:
                    print("absent", str(exc))
                else:
                    print("present", cv2.__file__)
                """
            ),
            "absent No module named 'cv2'",
        )

    def test_the_prelude_drops_an_already_imported_cv2(self):
        """A cv2 the child already holds is dropped, not merely shadowed.

        Faked through ``sys.modules`` so the assertion is the same on a machine
        that has cv2 and one that does not.
        """
        self.assertEqual(
            _run(
                """
                import sys, types
                sys.modules["cv2"] = types.ModuleType("cv2")   # as if already imported

                from _cv2_isolation import force_cv2_absent
                force_cv2_absent()
                print("cv2" in sys.modules)
                """,
                prelude=False,
            ),
            "False",
        )


class StubIsTornDownCompletelyTests(unittest.TestCase):
    def test_modules_imported_under_the_stub_are_evicted_with_it(self):
        """The stub, and everything that captured it, is gone afterwards.

        ``config.loader`` is imported *before* the block and must survive: the
        tear-down drops what the stub reached, not everything in sight.
        """
        observed = _run(
            """
            import sys
            from _cv2_isolation import cv2_stub

            import config.loader                      # imported before the stub

            with cv2_stub() as installed:
                import app
                print(
                    installed,
                    "cv2" in sys.modules,
                    "app" in sys.modules,
                    "camera.sync" in sys.modules,
                )
            print(
                installed,
                "cv2" in sys.modules,
                "app" in sys.modules,
                "camera.sync" in sys.modules,
                "pipeline" in sys.modules,
                "config.loader" in sys.modules,
            )
            """
        ).splitlines()
        inside, after = observed[0].split(), observed[1].split()
        self.assertEqual(inside, ["True", "True", "True", "True"])
        self.assertEqual(
            after,
            ["True", "False", "False", "False", "False", "True"],
            "the stub or a module holding it outlived the block",
        )

    def test_a_real_cv2_is_never_replaced_and_nothing_is_evicted(self):
        """On a machine that has cv2 the block is a no-op.

        Otherwise the deployment server - which does have cv2 - would run a
        different import graph from the developer box, and this file's whole
        premise is that the two agree.
        """
        observed = _run(
            """
            import sys, types
            sys.modules["cv2"] = types.ModuleType("cv2")   # stand-in for the real one

            from _cv2_isolation import cv2_stub

            with cv2_stub() as installed:
                import app
                print(installed, "cv2" in sys.modules, "app" in sys.modules)
            print(installed, "cv2" in sys.modules, "app" in sys.modules)
            """
        ).splitlines()
        inside, after = observed[0].split(), observed[1].split()
        self.assertEqual(inside, ["False", "True", "True"])
        self.assertEqual(
            after,
            ["False", "True", "True"],
            "a present cv2 must not be swapped out or evicted",
        )


class TheSuiteIsOrderIndependentTests(unittest.TestCase):
    """The property the whole exercise is for, stated end to end.

    Importing ``camera.sync`` must fail with "cv2 is not installed" no matter
    which neighbouring test module the interpreter imported first.  Before this
    fix the answer depended on the file name, so the suite's error count was
    not reproducible.
    """

    def _sync_import_errors(self, first: str) -> str:
        env = dict(os.environ, PYTHONPATH=str(BACKEND_ROOT))
        script = CV2_ABSENT_PRELUDE + textwrap.dedent(
            f"""
            import {first}
            try:
                import camera.sync
            except ModuleNotFoundError as exc:
                print("ModuleNotFoundError:", exc)
            else:
                print("imported camera.sync")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(TESTS_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return "\n".join(
            line
            for line in (result.stdout + result.stderr).splitlines()
            if "ModuleNotFoundError" in line or "AttributeError" in line or line.startswith("imported")
        )

    def test_a_neighbour_import_cannot_change_how_camera_sync_fails(self):
        for first in (
            "test_waypoint_trimming",
            "test_http_source_split",
            "test_shutdown",
            "test_postprocess_equivalence",
            "test_protocol_golden",
            "test_replay_tool",
        ):
            with self.subTest(imported_first=first):
                errors = self._sync_import_errors(first)
                self.assertIn("ModuleNotFoundError: No module named 'cv2'", errors)
                self.assertNotIn(
                    "AttributeError: module 'cv2' has no attribute",
                    errors,
                    f"importing {first} left a cv2 stub reachable from camera.sync",
                )


if __name__ == "__main__":
    unittest.main()
