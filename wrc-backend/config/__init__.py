"""Backend configuration package.

Holds the JSON document (``config.json``) and the single loader that turns it,
plus the existing environment variables, into one validated ``AppConfig``.

NOTE ON IMPORT NAME: adding this directory makes ``config`` importable as a
top-level package.  ``run_backend_real.sh`` puts ``$BACKEND`` ahead of
``alpamayo1.5/src`` on ``PYTHONPATH``, so this package wins for every
``import config`` in the process.  Before deploying, confirm the Alpamayo
checkout does not itself rely on a top-level ``config`` module::

    python -c "import alpamayo1_5.helper as h; print(h.__file__)"

This module intentionally has no import side effects.
"""
