"""Model-facing layer: input assembly and the Alpamayo inference wrapper.

``input_builder`` turns a camera window plus the pose history into the tensors
the model consumes; ``alpamayo`` owns the checkpoint, the prompt and the
sampling call; ``camera_labels`` holds the camera naming both of them, and the
inference package writer, need.

Neither module in this package imports ``app``, ``pipeline`` or any of the
takeover/sender machinery, so the model call can be exercised (and, for the
input builder, tested) without the HTTP layer or the vehicle link.
"""

__all__: list[str] = []
