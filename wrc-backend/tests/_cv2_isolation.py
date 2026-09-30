"""Import cv2-dependent backend modules without contaminating the interpreter.

Why this module exists
----------------------
``camera.sync.py`` and ``pipeline.py`` both ``import cv2`` at module level, and
cv2 is absent from lightweight developer machines.  Several test modules
therefore need a stub to get ``import app`` to work at all::

    sys.modules["cv2"] = types.ModuleType("cv2")
    import app
    del sys.modules["cv2"]          # <- looks like it undoes the stub

It does not.  ``camera.sync`` bound the stub to its *own* global while it was
imported, and deleting the ``sys.modules`` entry cannot reach that binding.  For
the rest of the process ``camera.sync.cv2`` is then an empty module, so the next
test module sees

    AttributeError: module 'cv2' has no attribute 'resize'

for what is really "cv2 is not installed on this machine".  Worse, the failure
is silent in the other direction too: with a ``MagicMock`` in place the same
call returns a mock and the test that should have failed outright passes.

Undoing the stub properly means dropping every backend module the stub reached,
so a later ``import camera.sync`` re-executes against the real (here: absent)
cv2 and fails honestly::

    with cv2_stub():
        import app
        import camera.sync
    # no stub in sys.modules, and nothing left holding one

Only when cv2 is genuinely missing.  A machine that has cv2 gets the real one
and this context manager does nothing at all - so a developer box and the
deployment server run the same import statements.

Usage rule: every cv2-dependent import goes INSIDE the ``with`` block, at module
level.  An import performed at test time, after the block has closed, re-executes
the module and raises ``ModuleNotFoundError: No module named 'cv2'``.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _backend_top_level_names() -> frozenset[str]:
    """Every importable top-level name this checkout provides.

    Used to decide what to evict: a backend module imported while the stub was
    installed may hold a reference to it, a third-party one cannot.
    """
    names = set()
    for entry in BACKEND_ROOT.iterdir():
        if entry.is_dir():
            if (entry / "__init__.py").exists():
                names.add(entry.name)
        elif entry.suffix == ".py":
            names.add(entry.stem)
    return frozenset(names)


BACKEND_TOP_LEVEL = _backend_top_level_names()


# --------------------------------------------------------------------------- #
# Taking cv2 away from a child interpreter
# --------------------------------------------------------------------------- #
# ``cv2_stub()`` only does anything where cv2 cannot be imported, and the
# deployment server *does* have cv2.  A test that calls it and assumes a stub
# was installed therefore passes on a developer box and fails on the server -
# the one machine the suite most needs to run on.  These children take cv2 away
# from themselves instead, so both machines walk the same path and neither
# result depends on what the machine happens to have installed.


class _Cv2Blocker:
    """A finder that makes ``import cv2`` fail even where cv2 is installed."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2" or fullname.startswith("cv2."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def force_cv2_absent() -> None:
    """Make cv2 unimportable for the rest of this interpreter's life.

    Deliberately irreversible.  The callers are throwaway child processes, and
    a way to put cv2 back could leave a suite in exactly the state this module
    exists to prevent.
    """
    for name in [n for n in sys.modules if n == "cv2" or n.startswith("cv2.")]:
        del sys.modules[name]
    sys.meta_path.insert(0, _Cv2Blocker())


#: Prepend to a child's source to give that child a machine without cv2.
CV2_ABSENT_PRELUDE = (
    "from _cv2_isolation import force_cv2_absent\n"
    "force_cv2_absent()\n"
)


@contextmanager
def cv2_stub():
    """Yield ``True`` if a throwaway stub was installed, ``False`` if cv2 is real."""
    if "cv2" in sys.modules:
        yield False
        return
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        yield False
        return

    before = set(sys.modules)
    sys.modules["cv2"] = types.ModuleType("cv2")
    try:
        yield True
    finally:
        sys.modules.pop("cv2", None)
        for name in set(sys.modules) - before:
            if name.split(".")[0] in BACKEND_TOP_LEVEL:
                del sys.modules[name]
