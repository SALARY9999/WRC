"""JPEG task bodies shared by PGX-1 and the network workers.

The wire carries the original JPEGs and the small ego tensors.  It does not
carry the decoded image tensor.  A network slot pushes an
:class:`InferenceResult` onto the scheduler's existing result queue; this
module is how that result is built from the HTTP body.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Mapping

from inference.tasks import GpuMemory, InferenceResult, InferenceTask
from vehicle.models import PoseSample


@dataclass(frozen=True)
class NetworkFault:
    """The HTTP call failed or timed out for one task.

    Delivered on the result queue so only the pipeline thread changes slot
    state.  A fault for a task the slot is no longer running is ignored.
    """

    worker_id: int
    task_seq: int
    error: str


@dataclass(frozen=True)
class NetworkHealth:
    """One ``GET /health`` answer, as the scheduler is allowed to use it.

    ``ready`` is true only when the worker said both ``ready`` and
    ``state="ready"``.  Busy, loading and failed answers stay false.
    """

    worker_id: int
    ready: bool
    state: str
    error: str | None = None


def health_is_ready(body: Mapping[str, Any]) -> bool:
    """True only for an explicit idle-and-loaded health answer."""
    return bool(body.get("ready")) and str(body.get("state") or "") == "ready"


def build_infer_body(task: InferenceTask) -> dict[str, Any]:
    """The ``POST /infer`` body.  JPEG bytes, never the image tensor."""
    window = task.window
    jpegs = getattr(window, "jpeg_bytes", None)
    if not jpegs:
        raise ValueError("network dispatch requires jpeg_bytes on the camera window")
    frame_ids = window.frame_ids
    frame_times = window.frame_timestamps_us
    if frame_ids is not None and len(frame_ids) != len(jpegs):
        raise ValueError("jpeg_bytes and frame_ids disagree on camera count")
    encoded: list[list[str | None]] = []
    for index, row in enumerate(jpegs):
        if frame_ids is not None and len(frame_ids[index]) != len(row):
            raise ValueError(
                f"jpeg_bytes camera row {index} does not match frame_ids"
            )
        if frame_times is not None and len(frame_times[index]) != len(row):
            raise ValueError(
                f"jpeg_bytes camera row {index} does not match frame_timestamps_us"
            )
        encoded.append(
            [
                base64.b64encode(bytes(item)).decode("ascii") if item else None
                for item in row
            ]
        )
    frames = window.frames
    return {
        "task_seq": int(task.task_seq),
        "capture_timestamp_us": int(task.capture_timestamp_us),
        "cycle_id": int(task.cycle_id),
        "camera_indices": [int(index) for index in window.camera_indices],
        "timestamps_us": [int(value) for value in window.timestamps_us],
        "frame_ids": frame_ids,
        "frame_timestamps_us": frame_times,
        "jpeg_b64": encoded,
        "image_height": int(frames.shape[-2]),
        "image_width": int(frames.shape[-1]),
        "ego_xyz": _jsonable(task.ego_xyz),
        "ego_rot": _jsonable(task.ego_rot),
        "reference_pose": _pose_dict(task.reference_pose),
    }


def to_inference_result(
    body: Mapping[str, Any],
    *,
    worker_id: int,
    device: int,
    task_seq: int,
) -> InferenceResult:
    """Turn a worker JSON body into the result the existing queue already carries.

    ``task_seq`` is the sequence this call was made for.  The body may echo a
    different one; that echo is kept so a late answer cannot be relabelled as
    the slot's current task.
    """
    echoed = int(body.get("task_seq", task_seq))
    ok = bool(body.get("ok"))
    error = body.get("error")
    return InferenceResult(
        task_seq=echoed,
        worker_id=int(worker_id),
        device=int(device),
        ok=ok,
        model_result=_model_result(body) if ok else None,
        error=None if error is None else str(error),
        error_kind=str(body.get("error_kind") or ""),
        infer_ms=float(body.get("infer_ms") or 0.0),
        finished_monotonic_ns=int(body.get("finished_monotonic_ns") or 0),
        memory=_memory(body.get("memory") or {}),
    )


def _pose_dict(pose: PoseSample) -> dict[str, float | int]:
    return {
        "timestamp_us": int(pose.timestamp_us),
        "latitude": float(pose.latitude),
        "longitude": float(pose.longitude),
        "altitude": float(pose.altitude),
        "heading_deg": float(pose.heading_deg),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        return value.tolist()
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (str, bytes)):
        return tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _memory(raw: Any) -> GpuMemory:
    if not isinstance(raw, Mapping):
        return GpuMemory()
    def number(name: str) -> int:
        try:
            return int(raw.get(name) or 0)
        except (TypeError, ValueError):
            return 0
    return GpuMemory(
        allocated_bytes=number("allocated_bytes"),
        reserved_bytes=number("reserved_bytes"),
        max_allocated_bytes=number("max_allocated_bytes"),
        max_reserved_bytes=number("max_reserved_bytes"),
    )


def _model_result(body: Mapping[str, Any]) -> Any:
    import numpy as np

    pose = body.get("ref_pose") or {}
    if not isinstance(pose, Mapping):
        pose = {}
    ref = PoseSample(
        timestamp_us=int(pose.get("timestamp_us") or body.get("t0_us") or 0),
        latitude=float(pose.get("latitude") or 0.0),
        longitude=float(pose.get("longitude") or 0.0),
        altitude=float(pose.get("altitude") or 0.0),
        heading_deg=float(pose.get("heading_deg") or 0.0),
    )
    positions = np.asarray(body.get("positions"), dtype=np.float64)
    rotations = np.asarray(body.get("rotations"), dtype=np.float64)
    cot = str(body.get("cot") or "")
    t0_us = int(body.get("t0_us") or 0)
    try:
        from inference.alpamayo import ModelResult
    except Exception:  # noqa: BLE001 - a CPU test box may not import the model stack
        return _StandIn(
            cot=cot,
            positions=positions,
            rotations=rotations,
            t0_us=t0_us,
            ref_pose=ref,
        )
    return ModelResult(
        cot=cot,
        positions=positions,
        rotations=rotations,
        t0_us=t0_us,
        ref_pose=ref,
    )


@dataclass
class _StandIn:
    cot: str
    positions: Any
    rotations: Any
    t0_us: int
    ref_pose: PoseSample


__all__ = [
    "NetworkFault",
    "NetworkHealth",
    "build_infer_body",
    "health_is_ready",
    "to_inference_result",
]
