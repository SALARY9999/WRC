"""One Alpamayo model behind ``GET /health`` and ``POST /infer``.

This is the process PGX-2, PGX-3 and PGX-4 run.  It loads one checkpoint,
accepts one inference at a time, and returns the model output.  It does not
schedule, post-process a trajectory, or send UDP to the vehicle.

``GET /health`` reports ``ready`` only when the model is loaded and no
inference is running.  A second ``POST /infer`` while one is in progress is
rejected with HTTP 409 and is not queued.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Callable

logger = logging.getLogger("app")

_MAX_BODY_BYTES = 64 * 1024 * 1024


@dataclass
class _CameraWindow:
    t0_us: int
    frames: Any
    camera_indices: list[int]
    timestamps_us: list[int]
    frame_ids: Any
    frame_timestamps_us: Any
    jpeg_bytes: list[list[bytes]]


@dataclass
class _PoseSample:
    timestamp_us: int
    latitude: float
    longitude: float
    altitude: float
    heading_deg: float


class InferenceService:
    """The worker's state machine: loading, ready, busy, failed.

    The lock is the concurrency limit.  ``health`` reads it from another
    request thread, which is why the server is a ``ThreadingHTTPServer``.
    """

    def __init__(
        self,
        *,
        decode: Callable[[bytes, tuple[int, int]], Any] | None = None,
        to_rgb: Callable[[Any], Any] | None = None,
    ) -> None:
        self._decode = decode or _decode_bgr
        self._to_rgb = to_rgb or _bgr_to_rgb
        self._lock = threading.Lock()
        self._busy = False
        self._runner: Any = None
        self._error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._runner is not None and self._error is None

    def set_runner(self, runner: Any) -> None:
        self._runner = runner
        self._error = None

    def fail(self, error: str) -> None:
        self._error = error
        self._runner = None

    def health(self) -> dict[str, Any]:
        if self._error:
            return {"ready": False, "state": "failed", "error": self._error}
        if not self.loaded:
            return {"ready": False, "state": "loading"}
        if self._busy or self._lock.locked():
            return {"ready": False, "state": "busy"}
        return {"ready": True, "state": "ready"}

    def infer(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if self._error:
            return 503, {
                "ok": False,
                "ready": False,
                "state": "failed",
                "error": self._error,
            }
        if not self.loaded:
            return 503, {
                "ok": False,
                "ready": False,
                "state": "loading",
                "error": "model is loading",
            }
        if not self._lock.acquire(blocking=False):
            return 409, {
                "ok": False,
                "ready": False,
                "state": "busy",
                "error": "inference already running",
            }
        self._busy = True
        started = time.monotonic()
        task_seq = int(body.get("task_seq") or 0)
        try:
            window = self._window(body)
            ego_xyz, ego_rot, ref_pose = self._inputs(body)
            result = self._runner.infer(window, ego_xyz, ego_rot, ref_pose)
            return 200, _ok_body(result, task_seq, (time.monotonic() - started) * 1000.0)
        except Exception as exc:  # noqa: BLE001 - a model error is a result, not a crash
            return 200, {
                "ok": False,
                "task_seq": task_seq,
                "error": f"{type(exc).__name__}: {exc}",
                "error_kind": "worker_exception",
                "infer_ms": (time.monotonic() - started) * 1000.0,
            }
        finally:
            self._busy = False
            self._lock.release()

    def _window(self, body: dict[str, Any]) -> Any:
        import numpy as np
        import torch

        size = (int(body["image_width"]), int(body["image_height"]))
        jpeg_rows: list[list[bytes]] = []
        tensor_rows = []
        for row in body["jpeg_b64"]:
            raws: list[bytes] = []
            frames = []
            for item in row:
                if not item:
                    raise ValueError("network inference is missing a JPEG")
                raw = base64.b64decode(item)
                raws.append(raw)
                decoded = self._decode(raw, size)
                if decoded is None:
                    raise ValueError("JPEG decode failed")
                rgb = np.ascontiguousarray(self._to_rgb(decoded))
                frames.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
            jpeg_rows.append(raws)
            tensor_rows.append(torch.stack(frames, dim=0))
        return _CameraWindow(
            t0_us=int(body.get("capture_timestamp_us") or 0),
            frames=torch.stack(tensor_rows, dim=0),
            camera_indices=[int(index) for index in body.get("camera_indices") or []],
            timestamps_us=[int(value) for value in body.get("timestamps_us") or []],
            frame_ids=body.get("frame_ids"),
            frame_timestamps_us=body.get("frame_timestamps_us"),
            jpeg_bytes=jpeg_rows,
        )

    def _inputs(self, body: dict[str, Any]) -> tuple[Any, Any, Any]:
        import torch

        pose = body.get("reference_pose") or {}
        ref = _PoseSample(
            timestamp_us=int(pose.get("timestamp_us") or 0),
            latitude=float(pose.get("latitude") or 0.0),
            longitude=float(pose.get("longitude") or 0.0),
            altitude=float(pose.get("altitude") or 0.0),
            heading_deg=float(pose.get("heading_deg") or 0.0),
        )
        return (
            torch.tensor(body["ego_xyz"], dtype=torch.float32),
            torch.tensor(body["ego_rot"], dtype=torch.float32),
            ref,
        )


def _decode_bgr(jpeg: bytes, size: tuple[int, int]) -> Any:
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    width, height = size
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def _bgr_to_rgb(frame: Any) -> Any:
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _ok_body(result: Any, task_seq: int, infer_ms: float) -> dict[str, Any]:
    import numpy as np

    pose = result.ref_pose
    return {
        "ok": True,
        "task_seq": int(task_seq),
        "infer_ms": float(infer_ms),
        "finished_monotonic_ns": time.monotonic_ns(),
        "cot": str(getattr(result, "cot", "") or ""),
        "positions": np.asarray(result.positions).tolist(),
        "rotations": np.asarray(result.rotations).tolist(),
        "t0_us": int(result.t0_us),
        "ref_pose": {
            "timestamp_us": int(pose.timestamp_us),
            "latitude": float(pose.latitude),
            "longitude": float(pose.longitude),
            "altitude": float(pose.altitude),
            "heading_deg": float(pose.heading_deg),
        },
    }


class WorkerHandler(BaseHTTPRequestHandler):
    """``GET /health`` and ``POST /infer`` only."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path.split("?", 1)[0] != "/health":
            self._send(404, {"ok": False, "error": "not found"})
            return
        self._send(200, self.server.service.health())  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path.split("?", 1)[0] != "/infer":
            self._send(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"ok": False, "error": "bad content length"})
            return
        if length <= 0 or length > _MAX_BODY_BYTES:
            self._send(400, {"ok": False, "error": "body size rejected"})
            return
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._send(400, {"ok": False, "error": "body is not JSON"})
            return
        if not isinstance(body, dict):
            self._send(400, {"ok": False, "error": "body must be an object"})
            return
        status, payload = self.server.service.infer(body)  # type: ignore[attr-defined]
        self._send(status, payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("network worker: " + fmt, *args)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def make_server(
    service: InferenceService,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), WorkerHandler)
    server.service = service  # type: ignore[attr-defined]
    return server


def _load_model(service: InferenceService, settings: Any) -> None:
    try:
        from inference.alpamayo import AlpamayoRunner

        runner = AlpamayoRunner(settings, device="cuda:0")
        runner.ensure_loaded()
        service.set_runner(runner)
        logger.info("network worker model loaded")
    except Exception as exc:  # noqa: BLE001 - health must be able to say why
        logger.exception("network worker failed to load")
        service.fail(f"{type(exc).__name__}: {exc}")


def serve(settings: Any, host: str, port: int) -> None:
    """Listen, then load the checkpoint on a side thread so health can answer."""
    service = InferenceService()
    threading.Thread(
        target=_load_model,
        args=(service, settings),
        name="alpamayo-network-load",
        daemon=True,
    ).start()
    server = make_server(service, host, port)
    logger.info("network worker listening on %s:%s", host, port)
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Alpamayo network inference worker")
    parser.add_argument(
        "--host",
        default=os.environ.get("ALPAMAYO_WORKER_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ALPAMAYO_WORKER_PORT", "8092")),
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "ALPAMAYO_CHECKPOINT",
            "/data/users_data/wanghaohao/701/model_weights/Alpamayo-1.5-10B",
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    settings = SimpleNamespace(
        checkpoint=args.checkpoint,
        frames_per_camera=int(os.environ.get("ALPAMAYO_FRAMES_PER_CAMERA", "4")),
        navigation_text=os.environ.get("ALPAMAYO_NAVIGATION_TEXT", ""),
        num_traj_samples=int(os.environ.get("ALPAMAYO_NUM_TRAJ_SAMPLES", "1")),
        top_p=float(os.environ.get("ALPAMAYO_TOP_P", "0.98")),
        temperature=float(os.environ.get("ALPAMAYO_TEMPERATURE", "0.6")),
        max_generation_length=int(
            os.environ.get("ALPAMAYO_MAX_GENERATION_LENGTH", "256")
        ),
        network=SimpleNamespace(
            hf_endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        ),
    )
    serve(settings, args.host, args.port)


if __name__ == "__main__":
    main()


__all__ = ["InferenceService", "WorkerHandler", "main", "make_server", "serve"]
