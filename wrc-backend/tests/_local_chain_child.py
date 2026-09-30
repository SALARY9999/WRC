"""Child process for ``tests/test_local_full_chain.py``.  Not a test module.

This file does not match ``test_*.py``, so discovery ignores it: it is launched
by the test as a subprocess, because ``app._settings`` is built at import time
from the environment and ``load_backend_config`` is ``lru_cache``d - the ports
cannot be redirected in-process, and a real datagram has to be sent to a socket
this process is listening on.

Two seams are stubbed, and nothing else
--------------------------------------
* ``pipeline.CameraWindowProvider`` - the camera source.  This machine has no
  cv2, so the real CIMG decoder cannot run.  The stub produces a genuine
  ``CameraWindow`` whose timestamps are consistent with the pose history, so
  ``InputBuilder``'s freshness gates all evaluate the real way and the real
  rejection reasons are reachable.
* ``AlpamayoRunner.ensure_loaded`` / ``AlpamayoRunner.infer`` - the checkpoint
  and the GPU.  The stub marks the model loaded (so the production ``loaded``
  property reports the truth for a simulated load) and returns a real
  ``ModelResult`` holding a 64-point straight trajectory.

Everything between them is the production path: the real loop, the real gates,
the real decision layer, the real crop, the real serializer, the real datagram
on a real socket, and for mode ``app`` the real FastAPI app and its real
``on_event`` startup/shutdown.

Modes
-----
``chain``  one real cycle, no HTTP.  Prints the parsed datagram and the snapshot.
``app``    the real app through ``fastapi.testclient.TestClient``, entered and
           exited as a context manager so the real ``_shutdown`` runs.

The last line of stdout is ``RESULT <json>``; the parent parses that and does
the asserting, so a failure is reported as a normal unittest failure with the
observed values in it.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import types

# ``app`` imports cv2 at module level; this machine has none.  A bare stub is
# safe in a throwaway child process, and the stub is never used: nothing on this
# path reaches a cv2 code path (only the camera sources and
# ``_save_infer_package`` do, and both are stubbed or disabled).
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import app  # noqa: E402
import pipeline  # noqa: E402
from camera.models import CameraWindow  # noqa: E402
from inference.alpamayo import AlpamayoRunner, ModelResult  # noqa: E402
from vehicle.models import GlobalPose, vehicle_azimuth_for_compass  # noqa: E402

RESULT_PREFIX = "RESULT "


class _LogCapture(logging.Handler):
    """Collects the lines the operator's runbook greps for."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - diagnostics must not break the run
            return
        with self._lock:
            self.messages.append(message)

    def matching(self, needle: str) -> list[str]:
        with self._lock:
            return [m for m in self.messages if needle in m]


class _StubCameras:
    """A ``CameraWindowProvider`` stand-in that builds a real ``CameraWindow``.

    The window it returns describes frames captured *now* (arrival stamps on the
    monotonic microsecond clock) at the vehicle's *newest* pose instant, which is
    exactly what a healthy provider produces - so every downstream gate, gate
    rejection and log line is reached the real way.
    """

    def __init__(self, cimg, **kwargs) -> None:
        self.camera_indices = [int(index) for index in cimg.camera_indices]
        self.sources = [
            f"cimg#cam{camera_id}"
            for camera_id in range(1, len(self.camera_indices) + 1)
        ]
        self.frames_per_camera = int(kwargs.get("frames_per_camera", 4))
        width, height = cimg.target_size
        self.frame_width = int(width)
        self.frame_height = int(height)
        # Assigned by the child after the runner exists: the stub needs the
        # runner's own history, and the runner builds its cameras in __init__.
        self.history = None
        self.window_builds = 0
        self.refusals: dict[str, int] = {}
        self.last_window_built_at: float | None = None
        self._stop = threading.Event()

    def reported_camera_count(self) -> int:
        """Same contract as ``CameraWindowProvider``: one stats row per camera."""
        return len(self.camera_indices)

    def start(self) -> None:
        self._stop.clear()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        return None

    def alive_readers(self) -> tuple[str, ...]:
        """Shutdown diagnostics: the stub's readers are always gone."""
        return ()

    def latest_window(self, sync_timeout_s: float = 2.0, catch_up_timeout_s: float = 0.05):
        latest = self.history.latest() if self.history is not None else None
        if latest is None:
            return None
        now_us = time.monotonic_ns() // 1000
        t0_us = int(latest.timestamp_us)
        n = len(self.camera_indices)
        t = self.frames_per_camera
        self.window_builds += 1
        self.last_window_built_at = time.monotonic()
        return CameraWindow(
            t0_us=t0_us,
            frames=torch.zeros(
                (n, t, 3, self.frame_height, self.frame_width), dtype=torch.uint8
            ),
            camera_indices=list(self.camera_indices),
            timestamps_us=[t0_us] * n,
            # Per-frame identities, the same shape the CIMG reader publishes.
            frame_ids=[
                [f"cam{index}:{t0_us - k * 100_000}" for k in range(t)]
                for index in self.camera_indices
            ],
            frame_timestamps_us=[[t0_us] * t for _ in range(n)],
            frame_arrivals_us=[[now_us] * t for _ in range(n)],
            frame_age_us=0,
        )

    def _reader_stats(self) -> list[dict]:
        """One healthy reader per camera, in the real key set.

        The decision layer refuses to run with a reader list of the wrong
        length (``TakeoverDecisionLayer.decide`` -> ``camera_status_unavailable``)
        or with an unknown frame age (-> ``camera_stale``), so a stub that
        reported nothing here would silently keep the whole cycle in the
        decision layer's fallback branch and the run would prove much less.
        Reporting four fresh readers puts the cycle on the real deviation path.
        """
        age_ms = 0.0 if self.last_window_built_at else None
        return [
            {
                "source": source,
                "health": "HEALTHY",
                "frames_total": self.window_builds * self.frames_per_camera,
                "frame_age_ms": age_ms,
                "restarts": 0,
                "consecutive_failures": 0,
                "last_error": None,
                "timestamp_mode": "capture_time_ns",
                "last_frame_ts_us": None,
            }
            for source in self.sources
        ]

    def stats(self) -> dict:
        return {
            "readers": self._reader_stats(),
            "window_builds": self.window_builds,
            "refusals": dict(self.refusals),
            "last_window_age_ms": (
                round((time.monotonic() - self.last_window_built_at) * 1000.0, 1)
                if self.last_window_built_at
                else None
            ),
        }

    def camera_health(self) -> dict[str, str]:
        return {source: "HEALTHY" for source in self.sources}

    def unhealthy_readers(self) -> tuple[str, ...]:
        return ()

    def cimg_health_view(self) -> dict:
        """Idle CIMG link: this child never opens the camera socket.

        ``/health`` and ``/status`` always carry ``camera_cimg``.  The stub
        reports a listener that has not accepted a vehicle, which is what the
        real provider publishes before the first connection.
        """
        return {
            "connection_state": "idle",
            "connected": False,
            "peer": None,
            "sessions": 0,
        }


def _install_seams() -> None:
    """Replace the camera provider and the model call, and record both."""

    def _ensure_loaded(self) -> None:
        # Stand in for the checkpoint load.  Setting ``_model`` is what makes the
        # production ``loaded`` property report True, so the snapshot and
        # /ready see a loaded model rather than a stub-shaped lie.
        if self._model is None:
            self._model = object()

    def _infer(self, window, ego_xyz, ego_rot, ref_pose) -> ModelResult:
        # The real ``infer`` reaches ``ensure_loaded`` on its first call, which
        # is what flips ``loaded`` to True.  Stubbing ``infer`` without this
        # would leave the snapshot reporting model_loaded=False forever - a lie
        # the child would then assert on.
        self.ensure_loaded()
        count = 64
        # 1 m apart along model +x, which is north for heading 0 - the same
        # fixture shape ``tests/test_replay_tool.py`` replays.
        positions = np.column_stack(
            (
                np.arange(count, dtype=np.float64),
                np.zeros(count, dtype=np.float64),
                np.zeros(count, dtype=np.float64),
            )
        )
        rotations = np.tile(np.eye(3, dtype=np.float64), (count, 1, 1))
        return ModelResult(
            cot="stub chain of thought",
            positions=positions,
            rotations=rotations,
            t0_us=int(window.t0_us),
            ref_pose=ref_pose,
        )

    AlpamayoRunner.ensure_loaded = _ensure_loaded
    AlpamayoRunner.infer = _infer
    pipeline.CameraWindowProvider = _StubCameras


class _PoseFeeder(threading.Thread):
    """Plays the vehicle: keeps the pose history fed on the monotonic clock.

    This is an input, not a stubbed collaborator.  The real ``UdpPoseListener``
    is bound and running in both modes; this process simply has no vehicle to
    send to it, so the samples a real receiver would append after decoding are
    appended directly - on the same clock, at the same rate, in the same order.
    ``ALPAMAYO_POSE_MODE=arrival`` is set by the parent, which is the mode whose
    ``timestamp_us`` *is* server monotonic microseconds, so the fed history and
    the stub window's ``t0_us`` share one time domain and no gate is loosened.

    A one-shot seed is not enough: the freshness gate only allows 800 ms, so a
    child that seeded once and then polled would watch its own pose go stale and
    could never observe the ready state it was waiting for.  Continuous feeding
    is both truer to the field and free of that race.

    The vehicle is stationary (``v_*`` zero, fixed lat/lon) deliberately: it
    keeps the trajectory trim's ``matched_segment_index`` at 0 and the crop a
    function of the model output alone, so the wire assertions have one less
    moving part.
    """

    def __init__(self, history, *, rate_hz: float = 10.0, backfill_s: float = 2.0) -> None:
        super().__init__(daemon=True, name="stub-vehicle")
        self.history = history
        self.period_s = 1.0 / rate_hz
        self.backfill_s = backfill_s
        self.samples = 0
        self.first_ts_us: int | None = None
        # Not ``self._stop``: ``threading.Thread`` owns that name as a private
        # method, and shadowing it breaks any later ``join()``.  Same trap as
        # the one PipelineRunner hit - see the note in pipeline.PipelineRunner.
        self._stop_event = threading.Event()

    @staticmethod
    def _sample(ts_us: int) -> GlobalPose:
        return GlobalPose(
            timestamp_us=ts_us,
            latitude=28.0,
            longitude=113.0,
            altitude=0.0,
            v_north=0.0,
            v_east=0.0,
            v_up=0.0,
            roll_deg=0.0,
            pitch_deg=0.0,
            # Compass east is vehicle azimuth 180.  The chain test expects
            # heading 90 so the stub +x track moves in longitude.
            azimuth_deg=vehicle_azimuth_for_compass(90.0),
            arrival_us=ts_us,
            payload_timestamp_us=ts_us,
        )

    def prime(self) -> int:
        """Backfill, so ``aligned_samples``' 1.5 s of coverage exists at once.

        Without this the first cycles are rejected for
        ``pose_history_unavailable`` while the feeder fills the buffer up.
        """
        now_us = time.monotonic_ns() // 1000
        step_us = int(round(self.period_s * 1_000_000))
        count = int(self.backfill_s / self.period_s)
        for k in range(count, -1, -1):
            self.history.add(self._sample(now_us - k * step_us))
        self.samples += count + 1
        self.first_ts_us = now_us - count * step_us
        return now_us

    def run(self) -> None:
        while not self._stop_event.wait(self.period_s):
            self.history.add(self._sample(time.monotonic_ns() // 1000))
            self.samples += 1

    def stop(self) -> None:
        self._stop_event.set()


def _snapshot_view(snapshot) -> dict:
    return {
        "sequence": snapshot.sequence,
        "phase": snapshot.phase,
        "cycle_id": snapshot.cycle_id,
        "inference_count": snapshot.inference_count,
        "model_loaded": snapshot.model_loaded,
        "last_error": snapshot.last_error,
        "last_num_waypoints": snapshot.last_num_waypoints,
        "last_window_ok_at": snapshot.last_window_ok_at,
        "last_control_flag": snapshot.last_control_flag,
        "last_sent_control_flag": snapshot.last_sent_control_flag,
        "last_pose_arrival_us": snapshot.last_pose_arrival_us,
        "unhealthy_readers": list(snapshot.unhealthy_readers),
    }


def _settings_view(settings) -> dict:
    return {
        "waypoint_host": settings.waypoint_host,
        "waypoint_port": settings.waypoint_port,
        "magic": settings.magic,
        "version": settings.version,
        "vehicle_id": settings.vehicle_id,
        "byte_order": settings.byte_order,
        "camera_indices": list(settings.cimg_camera_indices),
        "pose_mode": settings.pose_mode,
    }


def run_chain(capture: _LogCapture) -> dict:
    """Mode ``chain``: real pipeline cycles, no HTTP.

    The child sends and does not listen - the parent owns the capture socket, so
    the port is reserved before this process starts and the byte-level parsing
    happens on the asserting side rather than being reported from here.
    """
    _install_seams()
    settings = app._settings
    runner = app.PipelineRunner(settings)
    runner.cameras.history = runner.history
    feeder = _PoseFeeder(runner.history)
    feeder.prime()
    try:
        feeder.start()
        runner.start()
        deadline = time.monotonic() + 20.0
        # Two sends, not one: the second is what makes the sequence number and
        # the loop's continuation observable at all.
        while time.monotonic() < deadline and runner.sender.last_seq < 2:
            time.sleep(0.02)
        # Let one more cycle publish, so the snapshot is not the state of the
        # very first iteration.
        target = runner.inference_count + 1
        while time.monotonic() < deadline and runner.inference_count < target:
            time.sleep(0.02)
    finally:
        runner.stop()
        feeder.stop()

    decision = runner.last_takeover_decision
    return {
        "feeder_samples": feeder.samples,
        "inference_count": runner.inference_count,
        "sender_last_seq": runner.sender.last_seq,
        "last_control_flag": runner.last_control_flag,
        "last_sent_control_flag": runner.last_sent_control_flag,
        "last_takeover_decision": (
            None
            if decision is None
            else {
                "control_flag": decision.control_flag,
                "decision_reason": decision.decision_reason,
                "reference_source": decision.reference_source,
                "reference_fallback_reason": decision.reference_fallback_reason,
                "remaining_points": decision.remaining_points,
            }
        ),
        "snapshot": _snapshot_view(runner.snapshot),
        "settings": _settings_view(settings),
        "logs": {
            "inference_started": capture.matching("inference started:"),
            "inference_sent": capture.matching("inference sent:"),
            "waypoint_out": capture.matching("waypoint-out:"),
            "no_window": capture.matching("no camera window for"),
            "cycle_failed": capture.matching("pipeline cycle failed"),
            "rejected": capture.matching("gate blocked inference"),
        },
    }


def run_app(capture: _LogCapture) -> dict:
    """Mode ``app``: the real endpoints, then the real graceful shutdown."""
    from fastapi.testclient import TestClient

    _install_seams()
    observed: dict = {"queries": [], "shutdown_seconds": None, "poll_seconds": None}
    poll_deadline = time.monotonic() + 20.0
    with TestClient(app.app) as client:
        # Before anything is seeded, /ready must refuse.  This is the "starting"
        # state a real box sits in while the checkpoint loads, and it is
        # deterministic here: no camera window means no inference has run, so
        # the model cannot have marked itself loaded.
        first = client.get("/ready")
        observed["first_ready_status"] = first.status_code
        observed["first_ready_body"] = first.json()
        observed["first_ready_reason"] = first.json()["reason"]

        health = client.get("/health")
        observed["health_status"] = health.status_code
        observed["health_keys"] = list(health.json())
        observed["health_body"] = health.json()

        status = client.get("/status")
        observed["status_status"] = status.status_code
        observed["status_keys"] = list(status.json())

        # ``_startup`` builds the runner, so both handles only become reachable
        # here - after the real startup has already run.  Feeding the history now
        # is not a cheat: the loop reads ``self.history`` on every iteration, so
        # the next window is built from it exactly as it would be from a live
        # receiver.  Nothing else writes to this history: the pose listener is
        # bound but nobody sends, and mock mode is off.
        app._runner.cameras.history = app._runner.history
        feeder = _PoseFeeder(app._runner.history)
        observed["seeded_t0_us"] = feeder.prime()
        feeder.start()

        poll_started = time.monotonic()
        while time.monotonic() < poll_deadline:
            response = client.get("/ready")
            body = response.json()
            observed["last_ready_status"] = response.status_code
            observed["last_ready_body"] = body
            observed["queries"].append(
                {"status": response.status_code, "ready": body["ready"], "reason": body["reason"]}
            )
            if response.status_code == 200:
                break
            time.sleep(0.1)
        observed["poll_seconds"] = time.monotonic() - poll_started
        observed["inference_count_at_exit"] = app._runner.inference_count
        observed["sender_last_seq_at_exit"] = app._runner.sender.last_seq
        observed["snapshot_at_exit"] = _snapshot_view(app._runner.snapshot)
        observed["settings"] = _settings_view(app._settings)
        # Stopped before the exit so the shutdown measurement below is not
        # polluted by a feeder thread that production would never have.
        feeder.stop()
        # Measured around the context exit ALONE, so "bounded shutdown" is a
        # claim about the shutdown and not about the polling above it.
        exit_started = time.monotonic()
    observed["shutdown_seconds"] = time.monotonic() - exit_started
    observed["feeder_samples"] = feeder.samples
    # The runner object is still reachable through the closure above; whether
    # ``app._runner`` itself was cleared is what step 5 fixes, so it is recorded
    # rather than asserted until then.
    observed["runner_cleared_after_shutdown"] = app._runner is None
    observed["logs"] = {
        "pipeline_started": capture.matching("pipeline started"),
        "pipeline_stopping": capture.matching("pipeline stopping"),
        "pipeline_stopped": capture.matching("pipeline stopped"),
        "stop_incomplete": capture.matching("pipeline stop incomplete"),
        "inference_sent": capture.matching("inference sent:"),
    }
    return observed


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "chain"
    capture = _LogCapture()
    logging.getLogger("app").addHandler(capture)
    logging.getLogger("app").setLevel(logging.DEBUG)
    try:
        if mode == "chain":
            observed = run_chain(capture)
        elif mode == "app":
            observed = run_app(capture)
        else:
            raise SystemExit(f"unknown mode {mode!r}")
    except Exception as exc:  # noqa: BLE001 - reported to the parent verbatim
        import traceback

        print(f"CHILD-FAILED {mode}", file=sys.stderr)
        traceback.print_exc()
        print(
            RESULT_PREFIX
            + json.dumps({"mode": mode, "crashed": f"{type(exc).__name__}: {exc}"})
        )
        return 1
    print(RESULT_PREFIX + json.dumps(observed, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
