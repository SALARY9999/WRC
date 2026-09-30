"""Trajectory geometry: coordinate frames, trimming and takeover decisions.

Three modules, lowest first:

``trajectory.coordinates``
    The frame transforms (ENU, ego, local xy <-> lat/lon) and the constants
    they are built on.  Everything else here depends on it; it depends on
    nothing but numpy.
``trajectory.postprocess``
    The latency-driven crop of the trajectory that actually gets sent.
``trajectory.takeover``
    The takeover policy: the deviation primitives, the planned-route data
    model, and the per-cycle decision layer that runs the gates.

``postprocess`` and ``takeover`` are siblings, not a chain, and they are
allowed to disagree within one cycle: the decision measures deviation on the
time-compensated trajectory while the send path crops the raw model output.
That is deliberate - see each module's docstring.
"""
