"""How a model camera index is named to a human.

Two layers need this vocabulary and they must not disagree:

* the model prompt writes ``Front camera: frame 0 <image> ...`` into the chat
  message, so the text the model reads and the text recorded in the inference
  package describe the same camera;
* the inference package names its JPEGs ``left_t0.jpg`` and records
  ``camera_label``/``display`` per frame.

That is why the tables live in one file instead of one per module.  This module
deliberately imports nothing, because ``inference.input_builder`` reaches it and
that module must stay importable without the model stack.

The index space is Alpamayo's, not the deployment's: the four deployed cameras
map to indices ``1, 2, 0, 4`` (see ``camera_indices``), so indices 3 and 5..6
are named but unused, and an unknown index falls back to ``idx<N>`` rather
than raising - a package with a mislabelled camera is still worth reading.
"""

from __future__ import annotations

_CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}

# Human-readable file labels for the deployed four-camera mapping
# (left=0, front=1, right=2, back=4 in Alpamayo camera-index space).
_CAMERA_PACKAGE_LABELS = {0: "left", 1: "front", 2: "right", 4: "back"}


def _camera_package_label(camera_index: int) -> str:
    return _CAMERA_PACKAGE_LABELS.get(
        int(camera_index), f"idx{int(camera_index)}"
    )


__all__ = [
    "_CAMERA_DISPLAY_NAMES",
    "_CAMERA_PACKAGE_LABELS",
    "_camera_package_label",
]
