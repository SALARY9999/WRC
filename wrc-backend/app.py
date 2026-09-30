"""Alpamayo 1.5 real-time inference pipeline for the unmanned vehicle.

Data flow:

    CIMG camera input + UDP GlobalPose/ChassisInfo (vehicle -> us)
        -> build 4camera x 4frame window + 16-step ego history
        -> Alpamayo inference (64 future points)
        -> lat/lon waypoints
        -> UDP control packet back to vehicle (control_flag + waypoints)

Unknown vehicle binary payloads are deliberately isolated behind the
``VehicleUdpListener`` decoder hooks so that once a capture sample is available
only the decoder needs to be replaced.

Do NOT modify files under alpamayo1.5/src or alpasim upstream.  The wire
``control`` byte (0 = 不接管, 1 = 接管) is selected independently after
all input/output safety checks pass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from typing import Any

# ``torch`` is not dead: ``_log_startup_snapshot`` prints the CUDA build and
# whether a device is visible, which is the first thing to check when a model
# load fails in the field.  ``cv2``, ``numpy``, ``threading``, ``deque``,
# ``dataclass``, ``math`` and ``struct`` went with the legacy camera provider.
import torch

logger = logging.getLogger(__name__)

_log_level = os.environ.get("ALPAMAYO_LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# Tested camera synchronization/window assembly ported from zhoudong backend.
from camera.sync import CameraWindow, CameraWindowProvider
from config.loader import (
    AppConfig,
    CameraFreshnessConfig,
    load_app_config,
    load_backend_config,
    load_camera_freshness_config,
    load_navigation_text,
    resolve_config_path,
)
from vehicle.planned_route import PlannedRouteTcpReceiver
from trajectory.coordinates import (
    enu_from_ll,
    enu_to_ego,
    local_xy_to_latlon,
    yaw_to_matrix,
)
from trajectory.takeover import (
    ModelFrameOrigin,
    PlannedRoute,
    PlannedRouteCache,
    PlannedRouteReferenceProvider,
    ReferenceTrajectoryRequest,
    ReferenceTrajectorySelector,
    TakeoverDecision,
    TakeoverDecisionEngine,
    TrajectoryCompensator,
    TrajectoryDeviationCalculator,
)
from vehicle.history import (
    PoseHistory,
    build_ego_arrays,
    poses_to_model_frame_xy,
)
from vehicle.models import ChassisInfo, GlobalPose, PoseSample
from vehicle.receiver import (
    MockPoseProvider,
    UdpPoseListener,
    decode_chassis_json,
    decode_global_pose_json,
    decode_vehicle_timestamp_json,
)

# The outbound wire serializer now lives in ``waypoint.sender``.  Re-exported
# under its historical name: ``tools/`` and the golden protocol tests build a
# WaypointSender via ``app.WaypointSender``.
from waypoint.sender import WaypointSender  # noqa: E402

# The latency-driven trajectory crop now lives in ``trajectory.postprocess``.
# ``trim_waypoints_to_latest_pose`` is the crop that controls the send.
from trajectory.postprocess import (  # noqa: E402
    TrajectoryPostprocessResult,
    WaypointTrimResult,
    trim_waypoints_to_latest_pose,
)

# The per-cycle decision procedure.  It shares ``trajectory.takeover`` with the
# primitives imported above, which is the point of the directory closure: the
# root ``trajectory_takeover.py`` that used to hold them is gone, so there is no
# longer a second module whose name differs by one character.
from trajectory.takeover import TakeoverDecisionLayer  # noqa: E402

# The model-facing layer: input assembly, and the checkpoint / prompt /
# sampling call.  ``AlpamayoRunner``, ``ModelResult`` and ``_messages_text``
# keep their historical ``app.`` import path - ``tools/infer_images_pose_only.py``
# and ``test_backend.py`` build a runner through it.
from inference.alpamayo import (  # noqa: E402
    AlpamayoRunner,
    ModelResult,
    _messages_text,
)

# Camera naming, shared by the prompt text and the inference package.  Kept
# importable from ``app`` for the same reason as everything else here.
from inference.camera_labels import (  # noqa: E402
    _CAMERA_DISPLAY_NAMES,
    _CAMERA_PACKAGE_LABELS,
    _camera_package_label,
)

# The input gates, the ego tensors and the model-input diagnostics.  The
# rejection reason constants are re-exported so a caller can name a rejection
# instead of string-matching it.
from inference.input_builder import (  # noqa: E402
    FRESHNESS_REASONS,
    InputBuildResult,
    InputBuilder,
    REASON_CAMERA_POSE_DELTA,
    REASON_CAMERA_STALE,
    REASON_CAMERA_WINDOW_STALE,
    REASON_POSE_HISTORY_UNAVAILABLE,
    REASON_POSE_STALE,
)

# The real-time loop itself.  This must stay the *same class object* the
# pipeline module defines: the AST/structural tests, ``tools/package_harness.py``
# and ``vehicle/receiver.py`` all reach into its members by that path.
from pipeline import PipelineRunner  # noqa: E402


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

# The effective configuration now lives in ``config/loader.py``.  ``Settings``
# and ``Settings.from_env()`` remain the historical names used by the offline
# tooling and tests, so they are re-exported rather than renamed.
Settings = AppConfig


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def _log_startup_snapshot(settings: Settings, runner: "PipelineRunner") -> None:
    """Log the effective runtime values once without duplicating infer records."""
    config_path = resolve_config_path()
    config_version: Any = None
    config_sha256 = "unavailable"
    if config_path is not None:
        try:
            config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
            config_version = load_backend_config().get("version")
        except Exception as exc:  # configuration diagnostics must not stop startup
            config_sha256 = f"unavailable ({exc})"

    cuda_version = torch.version.cuda or "unavailable"
    lines = [
        "=" * 50,
        "Alpamayo backend startup",
        f"start_time={time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "[environment]",
        f"hostname={socket.gethostname()}",
        f"python={platform.python_version()} executable={sys.executable}",
        f"torch={torch.__version__}",
        f"cuda={cuda_version} available={torch.cuda.is_available()}",
        f"git_commit={_git_commit()}",
        "",
        "[config]",
        f"ALPAMAYO_CONFIG={config_path if config_path is not None else '(none)'}",
        f"config_version={config_version}",
        f"config_sha256={config_sha256}",
        f"navigation_text={json.dumps(settings.navigation_text, ensure_ascii=False)}",
        f"control_mode={settings.control_mode}",
        f"reference_source={settings.reference_source}",
        f"takeover_threshold_m={settings.takeover_threshold_m}",
        f"release_threshold_m={settings.release_threshold_m}",
        f"takeover_confirmations={settings.takeover_confirmations}",
        f"release_confirmations={settings.release_confirmations}",
        f"takeover_min_remaining_points={settings.takeover_min_remaining_points}",
        f"camera_clock=capture_time_ns",
        f"pose_timestamp_mode=unix",
        "",
        "[model]",
        f"checkpoint={settings.checkpoint}",
        f"temperature={settings.temperature}",
        f"top_p={settings.top_p}",
        f"max_generation_length={settings.max_generation_length}",
        f"mock_mode={settings.mock_mode}",
        # The one value here whose meaning depends on the mode, so it says so on
        # the line rather than in a comment nobody reading backend.log will see.
        # ``multi_process`` runs the pool continuously and paces it by the poll
        # interval; a sleep of min_cycle_s between cycles is exactly the
        # throughput the extra GPUs were bought for, so the setting is ignored
        # there and this is where an operator finds that out.
        (
            f"min_cycle_s={settings.min_cycle_s} INERT(mode=multi_process "
            "ignores it)"
            if settings.inference_pool.multi_process and settings.min_cycle_s > 0
            else f"min_cycle_s={settings.min_cycle_s}"
        ),
        "",
        "[inference_pool]",
        # Configuration only.  The device inventory - names, free VRAM - is
        # logged by ``GpuScheduler.start()`` a moment later, because it is probed
        # there and probing it twice would be two different answers; and
        # ``min_cycle_s`` above is here rather than there because it is static.
        f"mode={settings.inference_pool.mode}",
        "gpu_devices="
        + (
            ",".join(str(d) for d in settings.inference_pool.gpu_devices)
            or "(none)"
        ),
        f"worker_count={settings.inference_pool.worker_count}",
        f"minimum_ready_workers={settings.inference_pool.minimum_ready_workers}",
        f"latest_only={settings.inference_pool.latest_only}",
        f"pending_capacity={settings.inference_pool.pending_capacity}",
        "scheduler_poll_interval_ms="
        f"{settings.inference_pool.scheduler_poll_interval_ms}",
        f"worker_start_timeout_s={settings.inference_pool.worker_start_timeout_s}",
        f"worker_warmup={settings.inference_pool.worker_warmup}",
        f"result_max_age_ms={settings.inference_pool.result_max_age_ms}",
        f"max_capture_lag_ms={settings.inference_pool.max_capture_lag_ms}",
        f"metrics_window_s={settings.inference_pool.metrics_window_s}",
        "",
        "[network]",
        # The historical four lines, unchanged: ``pose_udp_host``/``port`` are
        # the endpoint ``UdpPoseListener`` actually binds now, rather than the
        # setting that used to be logged while the listener ran on its own
        # default.  Everything below them is the rest of the inventory, so one
        # startup log answers "what is this process listening on" completely.
        f"pose_udp_host={settings.pose_udp_host}",
        f"pose_udp_port={settings.pose_udp_port}",
        f"waypoint_host={settings.waypoint_host}",
        f"waypoint_udp_port={settings.waypoint_port}",
        f"cimg_host={settings.cimg_host}",
        f"cimg_port={settings.cimg_port}",
        f"route_tcp_enabled={settings.route_tcp_enabled}",
        f"route_tcp_host={settings.route_tcp_host}",
        f"route_tcp_port={settings.route_tcp_port}",
        f"http_host={settings.network.http.host}",
        f"http_port={settings.network.http.port}",
        f"hf_endpoint={settings.network.hf_endpoint}",
        "",
        "[camera]",
        f"cimg_host={settings.cimg_host}",
        f"cimg_port={settings.cimg_port}",
        f"cimg_camera_indices={','.join(str(index) for index in settings.cimg_camera_indices)}",
        f"cimg_read_timeout_s={settings.cimg_read_timeout_s}",
        f"cimg_max_payload_bytes={settings.cimg_max_payload_bytes}",
        f"cimg_max_jpeg_bytes={settings.cimg_max_jpeg_bytes}",
        f"cimg_queue_frames={settings.cimg_queue_frames}",
        f"cimg_max_backlog_groups={settings.cimg_max_backlog_groups}",
        f"cimg_group_fps={settings.cimg_group_fps}",
        f"frames_per_camera={settings.frames_per_camera}",
        f"target_fps={settings.target_fps}",
        f"frame_size={settings.frame_width}x{settings.frame_height}",
        f"max_camera_skew_us={settings.max_camera_skew_us}",
        f"camera_stale_ms={settings.camera_stale_ms}",
        f"camera_pose_delta_ms={settings.camera_pose_delta_ms}",
        f"window_max_age_ms={settings.window_max_age_ms}",
        f"camera_frame_timeout_ms={settings.camera_frame_timeout_ms}",
        "",
        "[pose]",
        "heading_source=vehicle_azimuth",
        f"azimuth_offset_deg={settings.azimuth_offset_deg}",
        f"history_steps={settings.history_steps}",
        f"history_dt_s={settings.history_dt_s}",
        f"pose_stale_ms={settings.pose_stale_ms}",
        "=" * 50,
    ]
    for line in lines:
        logger.info(line)


# --------------------------------------------------------------------------- #
# Geo helpers, vehicle models, pose history
# --------------------------------------------------------------------------- #
#
# These moved to ``trajectory/coordinates.py``, ``vehicle/models.py`` and
# ``vehicle/history.py`` and are re-exported above so that the historical
# ``from app import CameraWindow, PoseSample, Settings, build_ego_arrays,
# yaw_to_matrix`` import path (test_backend.py, tools/package_harness.py)
# keeps working unchanged.



# --------------------------------------------------------------------------- #
# Legacy manual control flag
# --------------------------------------------------------------------------- #
#
# ``_CAMERA_DISPLAY_NAMES``, ``_CAMERA_PACKAGE_LABELS``, ``_camera_package_label``
# and ``_messages_text`` used to live around here.  They now belong to
# ``inference/camera_labels.py`` and ``inference/alpamayo.py`` and are
# re-exported at the top of this file, so ``from app import _messages_text``
# keeps working.

_CONTROL_FLAG_OVERRIDE: int | None = None


def set_control_flag(value: int | None) -> None:
    """运行时写入控制位（0 = 不接管, 1 = 接管；None 回到配置默认值）。

    旧入口仅保留接口兼容；实际下发已由航线偏差引擎控制，不读取此值。
    """
    global _CONTROL_FLAG_OVERRIDE
    _CONTROL_FLAG_OVERRIDE = (
        None if value is None else (1 if int(value) == 1 else 0)
    )


def resolve_control_flag(
    settings: "Settings",
    window: "CameraWindow | None" = None,
    result: "ModelResult | None" = None,
) -> int:
    """旧手工控制查询入口；保留兼容，不再用于实际航点下发。

    取值优先级：set_control_flag() 的运行时值 > 配置项
    ``ALPAMAYO_CONTROL_FLAG``。window / result 作为可选入参备好，
    方便后续接入自己的判据（航点与原航线偏移、雷达触发、外部指令等）。
    """
    if _CONTROL_FLAG_OVERRIDE is not None:
        return _CONTROL_FLAG_OVERRIDE
    return 1 if int(settings.control_flag) == 1 else 0


# --------------------------------------------------------------------------- #
# Model wrapper and pipeline loop
# --------------------------------------------------------------------------- #
#
# ``ModelResult`` and ``AlpamayoRunner`` now live in ``inference/alpamayo.py``
# (together with the legacy prompt internals), and ``PipelineRunner`` lives in
# ``pipeline.py``.  Both are re-exported at the top of this file, so ``from app
# import PipelineRunner`` and ``from app import AlpamayoRunner`` - the
# historical paths used by ``tools/``, ``test_backend.py`` and the structural
# tests - resolve to the same class objects the new modules define.
#
# What remains here is assembly, lifecycle, HTTP, and compatibility
# re-exports.

# --------------------------------------------------------------------------- #
# FastAPI / pipeline
# --------------------------------------------------------------------------- #

from fastapi import FastAPI, Response  # noqa: E402

# The published-view module the three handlers render from.  A leaf: it imports
# nothing from here, from ``pipeline`` or from the camera/model layers.
#
# Imported under an alias, NOT as ``health``: this module defines a route
# handler called ``health`` below, and that ``def`` rebinds the global name - so
# a bare ``import health`` would be shadowed by the function and every
# ``health.something`` reference inside the handlers would raise AttributeError
# at request time.  The handler name must stay ``health`` because ``app.health()``
# is called directly by the golden regression harness and by tests.
import health as health_view  # noqa: E402


_settings = Settings.from_env()
_runner: PipelineRunner | None = None

app = FastAPI(title="Alpamayo real-time avoidance backend", version="0.2.0")


def _snapshot_and_alive() -> tuple[health_view.RuntimeSnapshot, bool]:
    """The runner's published view, and whether its thread is alive.

    Read as a pair so one response describes one instant.  ``is_alive()`` stays
    a live call rather than a snapshot field on purpose: a thread that has died
    must not be reported as alive until the next publish, which may be a whole
    inference away - and a dead loop is the one thing this endpoint exists to
    make visible.

    With no runner the empty snapshot is returned, whose defaults are what the
    handlers reported before Stage 3 for that case: ``inference_count`` 0,
    ``last_error`` null, and both control flags null.
    """
    runner = _runner
    if runner is None:
        return health_view.EMPTY_SNAPSHOT, False
    return runner.snapshot, runner.is_alive()


@app.on_event("startup")
def _startup() -> None:
    global _runner
    if _runner is None:
        _runner = PipelineRunner(_settings)
        _log_startup_snapshot(_settings, _runner)
        _runner.start()
        logger.info("pipeline started")


@app.on_event("shutdown")
def _shutdown() -> None:
    """Stop the pipeline, bounded, and say what could not be stopped.

    The two ``logger.info`` lines are unchanged byte-for-byte - operators grep
    ``backend.log`` for them, and ``run_backend_real.sh`` starts the replacement
    one second after signalling uvicorn.  What is new is the ERROR line: a
    component that did not stop (a decoder still holding UDP 31101-31104, say)
    used to be invisible here and only showed up as the *next* instance failing
    to bind.  Reporting it at shutdown puts the cause next to the symptom.

    ``_runner`` is cleared afterwards, but only for a shutdown that actually
    finished.  It is kept when ``report.failed`` is non-empty, and that is the
    whole point of keeping it: the runner is what makes the failure observable.
    While it is held, ``/health`` and ``/status`` keep answering from the last
    published snapshot and ``/ready`` keeps returning 503 - ``stop()`` left the
    phase at ``stopping`` precisely because a component did not stop.  Clearing
    it here, as this used to do unconditionally, would leave an operator with a
    process that looks cleanly stopped while a decoder still holds UDP
    31101-31104.  It never was cleared before Stage 3 either, so a second
    shutdown - or an ASGI server that runs the handler twice - would stop an
    already-stopped runner and log a second ``pipeline stopping``.
    """
    global _runner
    if _runner is None:
        return
    logger.info("pipeline stopping")
    report = _runner.stop()
    logger.info("pipeline stopped")
    if report.failed:
        logger.error(
            "pipeline stop incomplete: components_not_stopped=%s "
            "elapsed_ms=%.0f drain_s=%.1f reap_s=%.1f budget_exceeded=%s",
            ",".join(report.failed),
            report.elapsed_ms,
            report.drain_s,
            report.reap_s,
            report.budget_exceeded,
        )
        # Kept so the failure stays visible on /health, /status and /ready.  A
        # repeat shutdown call is safe and self-healing: stop() is idempotent,
        # and the second one clears the runner if it now succeeds.
        return
    _runner = None


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus the pipeline's own view.  Always 200 - see health.py.

    The body is assembled by ``health_view.build_health_body`` so the field order and
    the split between the live half and the published half live in one place
    that can be tested without FastAPI or a runner.
    """
    snapshot, pipeline_alive = _snapshot_and_alive()
    runner = _runner
    return health_view.build_health_body(
        snapshot,
        health_view.HealthSources(
            settings=_settings,
            # Live reads on purpose: ``tools/check_chain.sh`` samples this body
            # twice, two seconds apart, and derives each decoder's fps from the
            # delta of ``frames_total``.  Publishing those once per inference
            # cycle would freeze them exactly when a slow pipeline needs
            # diagnosing - see the module docstring in health.py.
            camera_health=runner.cameras.camera_health() if runner else {},
            camera_stats=runner.cameras.stats() if runner else {},
            # Live read, like the two above: connection state, peer, receive
            # fps and frame age are exactly the numbers an operator needs
            # while the pipeline is stuck.
            camera_cimg=(
                runner.cameras.cimg_health_view() if runner else {}
            ),
        ),
        now_monotonic_ns=time.monotonic_ns(),
        pipeline_alive=pipeline_alive,
    )


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    """200 when this process is healthy and receiving its inputs, else 503.

    Answers "is this process working", NOT "is it taking over": a missing
    planned route is a legitimate vehicle-side state that the decision layer
    already fails safe on, so it is reported as a warning rather than a 503 that
    a restart could not fix - see ``health.evaluate_readiness``.

    A JSON body is returned either way, so a caller reading only the status code
    and one reading only the body cannot draw different conclusions.
    """
    snapshot, pipeline_alive = _snapshot_and_alive()
    body = health_view.build_ready_body(
        snapshot,
        now_monotonic_ns=time.monotonic_ns(),
        pipeline_alive=pipeline_alive,
    )
    if not body["ready"]:
        response.status_code = 503
    return body


@app.get("/status")
def status() -> dict[str, Any]:
    """The vehicle stream and the per-cycle record.

    The field set is unchanged from before Stage 3 - new fields are not added
    here (that is ``/health``'s and ``/ready``'s job), because
    ``tools/check_chain.sh`` derives the pose rate from the deltas of
    ``chassis_packets`` and ``pose_timestamp`` and reads them by name.

    What changed is the *source* of the cycle-derived half: the pose, chassis
    and planned-route blocks are still live reads, while ``last_t0_us``,
    both control flags, the decision and the trajectory
    summary now come from one published snapshot - so they describe a single
    cycle instead of whichever cycle each attribute read happened to see.
    """
    snapshot, _ = _snapshot_and_alive()
    runner = _runner
    return health_view.build_status_body(
        snapshot,
        health_view.StatusSources(
            settings=_settings,
            runner_present=runner is not None,
            timestamp_manager=runner.timestamp_manager if runner else None,
            pose_listener=runner.pose_listener if runner else None,
            # The newest pose comes from the shared history, not the listener:
            # the listener is one of two writers and mock mode has none at all.
            history=runner.history if runner else None,
            planned_route_cache=runner.planned_route_cache if runner else None,
            route_tcp_receiver=runner.route_tcp_receiver if runner else None,
            camera_cimg=(
                runner.cameras.cimg_health_view() if runner else {}
            ),
        ),
    )


if __name__ == "__main__":
    import argparse  # noqa: E402

    import uvicorn  # noqa: E402

    # The HTTP listen address is the one endpoint with a command-line layer, so
    # the documented order - argument, then environment, then
    # ``config/config.json``, then the built-in default - holds end to end.
    # ``_settings.network.http`` already carries the lower three: the loader
    # applies ``ALPAMAYO_HOST``/``ALPAMAYO_PORT`` over the JSON section.  The
    # production launcher does not come through here at all; it invokes uvicorn
    # directly, and reads the same values out of the config file.
    _parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _parser.add_argument(
        "--host",
        default=None,
        help="listen address (default: network.http.host, currently "
        f"{_settings.network.http.host})",
    )
    _parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="listen port (default: network.http.port, currently "
        f"{_settings.network.http.port})",
    )
    _args = _parser.parse_args()
    _host = _args.host if _args.host else _settings.network.http.host
    _port = _args.port if _args.port is not None else _settings.network.http.port
    if not 1 <= _port <= 65535:
        _parser.error(f"--port {_port} must be in 1..65535")

    uvicorn.run(
        "app:app",
        host=_host,
        port=_port,
        reload=False,
    )
