"""Single loading entry point for the effective backend configuration.

Everything that used to be scattered across ``app.py::Settings.from_env()``,
``camera_freshness_config.py`` and ``navigation_config.py`` now resolves here,
so the effective configuration can be read, validated and logged in one place.
Those two shim modules and the root ``backend_config.py`` are gone: this is the
only import path for configuration now, for the app, the tools and the tests
alike.

Precedence is unchanged and is, per setting:

1. the environment variable, when set and non-empty;
2. the value in ``config/config.json`` (when the setting has a JSON field);
3. the built-in default.

The ``network`` section is the one place that rule did not hold.  Every
endpoint - TCP/UDP listen address and port, the waypoint target, and the HTTP
listen address the backend serves on - used to be read from the environment
with a hardcoded fallback, so a host or port written into the document was
silently ignored.  ``NetworkConfig`` gives them the JSON layer the rest of the
file already had.  The vehicle sends camera, pose and planned_route directly
to this process, and this process sends waypoints directly to the vehicle.
A ``network.relay`` section or a ``PC_IP`` / ``RELAY_HOST`` / ``RELAY_PORT``
variable is a retired PC-relay setting and is rejected at load time.
``network.camera_sources`` and ``ALPAMAYO_CAMERA_INPUT`` /
``ALPAMAYO_CAMERA_SOURCES`` / ``ALPAMAYO_CAMERA_INDICES`` are the retired
MPEG-TS input and are rejected the same way.  Cameras arrive as CIMG on
``network.cimg``.

Two settings are deliberately *not* read from the environment:

* ``control_mode`` is always ``"trajectory_deviation"``;
* ``reference_source`` is always ``"planned_route"``.

``run_backend_real.sh`` used to export ``ALPAMAYO_CONTROL_MODE``.  That export
did nothing and has been removed.  Do not put it back, and do not teach this
loader to read it.

Vehicle heading calibration is also JSON-only: the single source is
``vehicle_heading.azimuth_offset_deg`` in ``config/config.json``.  A shell
override would make the recorded configuration disagree with the actual
vehicle-frame transform.

``ALPAMAYO_CONTROL_MODE`` and ``ALPAMAYO_REFERENCE_SOURCE`` must not be
re-activated here; older snapshots under ``.remote_compare/`` referenced an
``ALPAMAYO_CONTROL_MODE=fixed`` / ``ALPAMAYO_REFERENCE_SOURCE=history_pose``
mode that the live code has since removed.

This module reads the environment and the JSON document only.  It performs no
I/O beyond that, opens no sockets, and imports nothing from the pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# JSON document
# --------------------------------------------------------------------------- #

def resolve_config_path() -> Path | None:
    """Resolve the unified path, retaining legacy path variables as fallback.

    ``ALPAMAYO_CONFIG`` wins when set; otherwise ``config/config.json`` beside
    this package, or ``None`` when that file does not exist.  Moved here from
    the root ``backend_config.py``, which was the third module implementing
    configuration loading.  The default is anchored on this file's directory
    (``<root>/config/``), not on the process working directory, so the answer is
    the same however the process was started.
    """
    configured = os.environ.get("ALPAMAYO_CONFIG", "").strip()
    if configured:
        return Path(configured)

    default_path = Path(__file__).resolve().parent / "config.json"
    return default_path if default_path.is_file() else None


@lru_cache(maxsize=8)
def _read_document(path: str) -> Mapping[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        document: Any = json.load(file)
    if not isinstance(document, Mapping):
        raise ValueError("backend config root must be an object")
    return document


def load_backend_config() -> Mapping[str, Any]:
    """The parsed ``config/config.json`` document (empty when absent)."""
    path = resolve_config_path()
    return _read_document(str(path.resolve())) if path is not None else {}


# --------------------------------------------------------------------------- #
# Small typed readers
# --------------------------------------------------------------------------- #

def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: list[str]) -> list[str]:
    val = os.environ.get(name)
    if val is None or not val.strip():
        return list(default)
    return [x.strip() for x in val.split(",") if x.strip()]


def _non_negative_number(value: Any, default: float) -> float:
    """Like ``_positive_number``, but ``0`` is a real setting.

    ``dispatch_interval_ms=0`` means "do not rate-limit", which has to survive
    the loader.  A negative or non-numeric value still falls back to the
    default rather than starting the pool with a reversed interval.
    """
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number >= 0 else default


def _positive_number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number > 0 else default


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if number > 0 else default


# --------------------------------------------------------------------------- #
# vehicle_heading
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VehicleHeadingConfig:
    """Vehicle body-heading calibration loaded only from config.json."""

    azimuth_offset_deg: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "VehicleHeadingConfig":
        raw = values.get("azimuth_offset_deg", 0.0)
        if isinstance(raw, bool):
            raise ValueError("vehicle_heading.azimuth_offset_deg must be a number")
        try:
            offset = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "vehicle_heading.azimuth_offset_deg must be a finite number"
            ) from exc
        if not math.isfinite(offset):
            raise ValueError(
                "vehicle_heading.azimuth_offset_deg must be a finite number"
            )
        return cls(azimuth_offset_deg=offset)


def load_vehicle_heading_config() -> VehicleHeadingConfig:
    """Load the vehicle heading calibration; environment overrides are forbidden."""
    try:
        values = load_backend_config().get("vehicle_heading", {})
    except OSError as exc:
        logger.warning("cannot load vehicle_heading; using defaults: %s", exc)
        return VehicleHeadingConfig()
    if not isinstance(values, Mapping):
        raise ValueError("vehicle_heading must be an object")
    return VehicleHeadingConfig.from_mapping(values)


# --------------------------------------------------------------------------- #
# camera_freshness
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CameraFreshnessConfig:
    camera_stale_ms: float = 1000.0
    camera_pose_delta_ms: float = 1000.0
    window_max_age_ms: int = 2000
    frame_timeout_ms: int = 3000

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CameraFreshnessConfig":
        defaults = cls()
        return cls(
            camera_stale_ms=_positive_number(
                values.get("camera_stale_ms"), defaults.camera_stale_ms
            ),
            camera_pose_delta_ms=_positive_number(
                values.get("camera_pose_delta_ms"), defaults.camera_pose_delta_ms
            ),
            window_max_age_ms=max(
                200,
                _positive_int(
                    values.get("window_max_age_ms"), defaults.window_max_age_ms
                ),
            ),
            frame_timeout_ms=_positive_int(
                values.get("frame_timeout_ms"), defaults.frame_timeout_ms
            ),
        )


def _non_negative_int(value: Any, default: int) -> int:
    """A zero-or-more integer.  Zero is meaningful (it disables retention)."""
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


@dataclass(frozen=True)
class RetentionConfig:
    """How many snapshot and inference-package directories to keep.

    ``0`` disables pruning for that tree.  The defaults are ten minutes at the
    production 1 Hz cadence: one snapshot directory per second, one inference
    package per cycle.
    """

    snapshot_keep: int = 600
    infer_package_keep: int = 600

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RetentionConfig":
        defaults = cls()
        return cls(
            snapshot_keep=_non_negative_int(
                values.get("snapshot_keep"), defaults.snapshot_keep
            ),
            infer_package_keep=_non_negative_int(
                values.get("infer_package_keep"), defaults.infer_package_keep
            ),
        )


def load_retention_config() -> RetentionConfig:
    """Load ``retention``.  A bad value falls back; the environment wins.

    Precedence matches the rest of this module: a non-empty environment
    variable, then the JSON field, then the built-in default.
    """
    defaults = RetentionConfig()
    try:
        values = load_backend_config().get("retention", {})
    except OSError as exc:
        logger.warning("cannot load retention; using defaults: %s", exc)
        values = {}
    if not isinstance(values, Mapping):
        logger.warning("retention must be an object; using defaults")
        values = {}
    config = RetentionConfig.from_mapping(values)
    snapshot_raw = os.environ.get("ALPAMAYO_SNAPSHOT_KEEP", "").strip()
    package_raw = os.environ.get("ALPAMAYO_INFER_PACKAGE_KEEP", "").strip()
    return RetentionConfig(
        snapshot_keep=(
            _non_negative_int(snapshot_raw, config.snapshot_keep)
            if snapshot_raw
            else config.snapshot_keep
        ),
        infer_package_keep=(
            _non_negative_int(package_raw, config.infer_package_keep)
            if package_raw
            else config.infer_package_keep
        ),
    )


def load_camera_freshness_config() -> CameraFreshnessConfig:
    """Load camera_freshness; malformed optional values fall back safely."""
    try:
        values = load_backend_config().get("camera_freshness", {})
        if not isinstance(values, Mapping):
            logger.warning("camera_freshness must be an object; using defaults")
            return CameraFreshnessConfig()
        return CameraFreshnessConfig.from_mapping(values)
    except Exception as exc:  # noqa: BLE001 - freshness defaults keep startup safe
        logger.warning("cannot load camera_freshness; using defaults: %s", exc)
        return CameraFreshnessConfig()


# --------------------------------------------------------------------------- #
# navigation_text
# --------------------------------------------------------------------------- #

def load_navigation_text() -> str:
    """Load navigation text without making the optional setting startup-critical."""
    # Preserve the pre-existing environment-variable path as an override.
    if "ALPAMAYO_NAV_TEXT" in os.environ:
        return os.environ["ALPAMAYO_NAV_TEXT"].strip()

    try:
        document = load_backend_config()
        value = document.get("navigation_text", "")
        if not isinstance(value, str):
            logger.warning("navigation_text must be a string; ignoring configured value")
            return ""
        return value.strip()
    except Exception as exc:  # noqa: BLE001 - optional input must not stop inference
        logger.warning("cannot read navigation_text from backend config: %s", exc)
        return ""


# --------------------------------------------------------------------------- #
# inference_pool
# --------------------------------------------------------------------------- #

#: One GPU, one complete model instance, everything on the pipeline thread -
#: the original behaviour, and the default.
INFERENCE_MODE_INLINE = "inline"
#: One worker process per GPU, each with its own complete model instance, with
#: a scheduler in the pipeline thread deciding who runs what.
INFERENCE_MODE_MULTI_PROCESS = "multi_process"
INFERENCE_MODES = (INFERENCE_MODE_INLINE, INFERENCE_MODE_MULTI_PROCESS)


@dataclass(frozen=True)
class InferencePoolConfig:
    """How many GPUs run models, and how tasks are handed to them.

    Every value here has a defined meaning; none is a knob that only looks
    tunable.  The defaults reproduce exactly the single-GPU inline behaviour, so
    an unmodified ``config/config.json`` starts the pipeline the vehicle has
    always run.

    ``mode``
        ``"inline"``        the pipeline thread loads one model and blocks on
                            it, then sleeps ``min_cycle_s``.  Unchanged.
        ``"multi_process"`` one spawned process per entry in ``gpu_devices``,
                            each loading a complete checkpoint.  The pipeline
                            thread never blocks on a model.
    ``gpu_devices``
        *Physical* device indices, the numbers ``nvidia-smi`` shows.  The list
        length is the worker count - nothing in the code assumes four, and one,
        four, five or six all work.  Each worker pins itself with
        ``CUDA_VISIBLE_DEVICES``, so the parent's value is overridden.
    ``latest_only`` / ``pending_capacity``
        While every worker is busy, the scheduler keeps only the newest task.
        ``latest_only`` must have ``pending_capacity == 1``; the two are the
        same policy written twice, and a config that disagrees is rejected
        rather than silently resolved.
    ``result_max_age_ms``
        A task whose result arrives later than this (measured from the moment
        the input was frozen) is dropped rather than sent.  This is the budget
        the whole pool is sized against: with four workers each taking ~1.5 s,
        a task waits at most one inference behind the newest one.
    ``max_capture_lag_ms``
        Two checks in one number, both about the *camera* clock.  A task more
        than this far behind the newest capture window is stale; a capture
        timestamp that moves more than this in either direction between
        consecutive tasks is a clock fault (a step, not a slow frame) and the
        task is dropped.  Camera and pose share one server-monotonic arrival
        clock in every timestamp mode, so this bound is domain-independent.
    ``scheduler_poll_interval_ms``
        How often the pipeline thread drains results and fills idle workers.
        This - never ``min_cycle_s`` - is what paces the multi-process loop.
    ``worker_start_timeout_s``
        How long a worker may spend loading its checkpoint before it is
        declared failed.  Loading a 10B checkpoint takes a minute or more.
    ``minimum_ready_workers``
        Readiness floor.  ``/ready`` reports not-ready until this many workers
        are loaded and able to accept a task, so a pool that came up degraded
        says so instead of quietly running at half rate.  The bound is the
        number of slots: local GPUs plus ``network_workers``.
    ``dispatch_interval_ms``
        Minimum gap between dispatches.  ``submit`` and the pending fill share
        one clock.  ``0`` means do not rate-limit; a task that arrives inside
        the gap waits in the existing pending slot.
    ``network_workers``
        Base URLs for the other PGX machines.  Each is one slot that POSTs a
        JPEG task to ``/infer``.  Those slots have no local process.  The GPU
        on this machine stays in ``gpu_devices``.
    ``network_connect_timeout_s`` / ``network_infer_timeout_s``
        How long connecting may take, and how long a dispatched request may
        run before that slot is marked unreachable.
    ``network_health_interval_ms``
        How often a loading or unreachable network slot calls ``GET /health``.
        It accepts work again only when the answer is explicitly ready.
    """

    mode: str = INFERENCE_MODE_INLINE
    gpu_devices: tuple[int, ...] = ()
    latest_only: bool = True
    pending_capacity: int = 1
    result_max_age_ms: float = 2500.0
    max_capture_lag_ms: float = 2500.0
    scheduler_poll_interval_ms: float = 20.0
    worker_start_timeout_s: float = 180.0
    minimum_ready_workers: int = 1
    metrics_window_s: float = 60.0
    worker_warmup: bool = True
    dispatch_interval_ms: float = 0.0
    network_workers: tuple[str, ...] = ()
    network_connect_timeout_s: float = 2.0
    network_infer_timeout_s: float = 4.0
    network_health_interval_ms: float = 500.0

    @property
    def multi_process(self) -> bool:
        return self.mode == INFERENCE_MODE_MULTI_PROCESS

    @property
    def poll_interval_s(self) -> float:
        return self.scheduler_poll_interval_ms / 1000.0

    @property
    def worker_count(self) -> int:
        return len(self.gpu_devices)

    def validate(self) -> str | None:
        """Fast structural check; ``None`` when the config is usable.

        Returns a message rather than raising so ``/health`` can report a
        config problem that the startup path turned into a refusal, and so the
        same function can serve both the loader (which raises) and the pool
        (which reports).
        """
        if self.mode not in INFERENCE_MODES:
            return (
                f"inference_pool.mode={self.mode!r} is not one of "
                f"{list(INFERENCE_MODES)}"
            )
        devices = [int(d) for d in self.gpu_devices]
        if any(d < 0 for d in devices):
            return f"inference_pool.gpu_devices has a negative index: {devices}"
        duplicates = sorted({d for d in devices if devices.count(d) > 1})
        if duplicates:
            return (
                "inference_pool.gpu_devices lists the same device twice: "
                f"{duplicates}.  One GPU runs one complete model instance, so "
                "two workers cannot share a device."
            )
        if self.latest_only and self.pending_capacity != 1:
            return (
                "inference_pool.latest_only=true requires pending_capacity=1 "
                f"(got {self.pending_capacity}); latest-only keeps exactly one "
                "pending task"
            )
        if self.pending_capacity < 1:
            return (
                f"inference_pool.pending_capacity={self.pending_capacity} must "
                "be at least 1"
            )
        for name, value in (
            ("result_max_age_ms", self.result_max_age_ms),
            ("max_capture_lag_ms", self.max_capture_lag_ms),
            ("scheduler_poll_interval_ms", self.scheduler_poll_interval_ms),
            ("worker_start_timeout_s", self.worker_start_timeout_s),
            ("metrics_window_s", self.metrics_window_s),
        ):
            if not (isinstance(value, (int, float)) and value > 0):
                return f"inference_pool.{name}={value!r} must be greater than 0"
        if self.dispatch_interval_ms < 0 or not math.isfinite(self.dispatch_interval_ms):
            return (
                "inference_pool.dispatch_interval_ms="
                f"{self.dispatch_interval_ms!r} must be zero or greater"
            )
        for name, value in (
            ("network_connect_timeout_s", self.network_connect_timeout_s),
            ("network_infer_timeout_s", self.network_infer_timeout_s),
            ("network_health_interval_ms", self.network_health_interval_ms),
        ):
            if not (isinstance(value, (int, float)) and value > 0):
                return f"inference_pool.{name}={value!r} must be greater than 0"
        urls = [str(url) for url in self.network_workers]
        if any(not url.startswith(("http://", "https://")) for url in urls):
            return (
                "inference_pool.network_workers entries must be http(s) URLs, "
                f"got {urls}"
            )
        duplicate_urls = sorted({url for url in urls if urls.count(url) > 1})
        if duplicate_urls:
            return (
                "inference_pool.network_workers lists the same URL twice: "
                f"{duplicate_urls}.  One machine runs one model, so two slots "
                "cannot share it."
            )
        if self.multi_process:
            if not devices:
                return (
                    "inference_pool.mode='multi_process' needs a non-empty "
                    "inference_pool.gpu_devices"
                )
            slot_count = len(devices) + len(urls)
            if not 1 <= self.minimum_ready_workers <= slot_count:
                return (
                    "inference_pool.minimum_ready_workers="
                    f"{self.minimum_ready_workers} must be between 1 and the "
                    f"number of workers ({slot_count})"
                )
        return None

    def require_valid(self) -> None:
        problem = self.validate()
        if problem is not None:
            raise ValueError(problem)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "gpu_devices": [int(d) for d in self.gpu_devices],
            "worker_count": self.worker_count,
            "latest_only": self.latest_only,
            "pending_capacity": self.pending_capacity,
            "result_max_age_ms": self.result_max_age_ms,
            "max_capture_lag_ms": self.max_capture_lag_ms,
            "scheduler_poll_interval_ms": self.scheduler_poll_interval_ms,
            "worker_start_timeout_s": self.worker_start_timeout_s,
            "minimum_ready_workers": self.minimum_ready_workers,
            "metrics_window_s": self.metrics_window_s,
            "worker_warmup": self.worker_warmup,
            "dispatch_interval_ms": self.dispatch_interval_ms,
            "network_workers": list(self.network_workers),
            "network_connect_timeout_s": self.network_connect_timeout_s,
            "network_infer_timeout_s": self.network_infer_timeout_s,
            "network_health_interval_ms": self.network_health_interval_ms,
        }


def _read_device_list(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    """Parse ``[1, 3]`` from JSON or ``"1,3"`` from the environment.

    A malformed entry is kept as its raw string so the validator can report
    ``gpu_devices=['1', 'x']`` verbatim - silently dropping an unparseable
    device index would start a pool with fewer workers than configured, which
    is the failure this whole module is trying to make impossible.
    """
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        value = [part.strip() for part in text.split(",") if part.strip()]
    if not isinstance(value, (list, tuple)):
        return default
    out: list[int] = []
    for entry in value:
        try:
            out.append(int(entry))
        except (TypeError, ValueError):
            # Sentinel that fails validation with the offending value visible.
            out.append(-1)
    return tuple(out)


def _read_url_list(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse network-worker URLs from a JSON list or a comma-separated env value."""
    if value is None:
        return default
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, (list, tuple)):
        return default
    return tuple(str(item).strip() for item in value if str(item).strip())


def load_inference_pool_config() -> InferencePoolConfig:
    """Resolve the pool config: environment, then JSON, then defaults.

    Rejects an unusable configuration at load time.  This runs during
    ``AppConfig.from_env()``, i.e. before any GPU is touched, so a typo in
    ``gpu_devices`` is a startup error with a readable message rather than four
    workers where one dies on a CUDA call an hour later.
    """
    defaults = InferencePoolConfig()
    values: Mapping[str, Any] = {}
    try:
        raw = load_backend_config().get("inference_pool", {})
    except Exception as exc:  # noqa: BLE001 - same contract as every other section
        # An absent or unreadable document falls back to defaults with a
        # warning, exactly as camera_freshness and navigation_text do.  It
        # cannot hide a misconfigured pool: ``multi_process`` is only reachable
        # through an explicit device list, and the default is ``inline``, which
        # is the behaviour the vehicle has always run.
        logger.warning(
            "cannot read inference_pool from backend config; using defaults: %s", exc
        )
        raw = {}
    if isinstance(raw, Mapping):
        values = raw
    elif raw is not None:
        raise ValueError("inference_pool must be an object")

    mode = os.environ.get("ALPAMAYO_INFERENCE_MODE", "").strip().lower() or str(
        values.get("mode", defaults.mode)
    ).strip().lower()
    devices = _read_device_list(
        os.environ.get("ALPAMAYO_GPU_DEVICES"),
        _read_device_list(values.get("gpu_devices"), defaults.gpu_devices),
    )
    latest_only = _env_bool(
        "ALPAMAYO_LATEST_ONLY",
        bool(values.get("latest_only", defaults.latest_only)),
    )
    env_workers = os.environ.get("ALPAMAYO_NETWORK_WORKERS")
    network_workers = _read_url_list(
        env_workers if env_workers is not None else values.get("network_workers"),
        defaults.network_workers,
    )
    slot_count = len(devices) + len(network_workers)
    config = InferencePoolConfig(
        mode=mode,
        gpu_devices=devices,
        latest_only=latest_only,
        pending_capacity=int(
            os.environ.get(
                "ALPAMAYO_PENDING_CAPACITY",
                str(values.get("pending_capacity", defaults.pending_capacity)),
            )
        ),
        result_max_age_ms=_positive_number(
            os.environ.get("ALPAMAYO_RESULT_MAX_AGE_MS"),
            _positive_number(
                values.get("result_max_age_ms"), defaults.result_max_age_ms
            ),
        ),
        max_capture_lag_ms=_positive_number(
            os.environ.get("ALPAMAYO_MAX_CAPTURE_LAG_MS"),
            _positive_number(
                values.get("max_capture_lag_ms"), defaults.max_capture_lag_ms
            ),
        ),
        scheduler_poll_interval_ms=_positive_number(
            os.environ.get("ALPAMAYO_SCHEDULER_POLL_INTERVAL_MS"),
            _positive_number(
                values.get("scheduler_poll_interval_ms"),
                defaults.scheduler_poll_interval_ms,
            ),
        ),
        worker_start_timeout_s=_positive_number(
            os.environ.get("ALPAMAYO_WORKER_START_TIMEOUT_S"),
            _positive_number(
                values.get("worker_start_timeout_s"),
                defaults.worker_start_timeout_s,
            ),
        ),
        minimum_ready_workers=int(
            os.environ.get(
                "ALPAMAYO_MINIMUM_READY_WORKERS",
                str(
                    values.get(
                        "minimum_ready_workers",
                        # In multi-process the useful default is "every worker
                        # asked for", local GPUs and network slots together.
                        # Inline has no workers, so the floor is irrelevant.
                        # An explicit config or environment value still wins,
                        # which is how a cluster keeps running on one worker.
                        max(1, slot_count)
                        if mode == INFERENCE_MODE_MULTI_PROCESS
                        else defaults.minimum_ready_workers,
                    )
                ),
            )
        ),
        metrics_window_s=_positive_number(
            os.environ.get("ALPAMAYO_METRICS_WINDOW_S"),
            _positive_number(values.get("metrics_window_s"), defaults.metrics_window_s),
        ),
        worker_warmup=_env_bool(
            "ALPAMAYO_WORKER_WARMUP",
            bool(values.get("worker_warmup", defaults.worker_warmup)),
        ),
        dispatch_interval_ms=_non_negative_number(
            os.environ.get("ALPAMAYO_DISPATCH_INTERVAL_MS"),
            _non_negative_number(
                values.get("dispatch_interval_ms"), defaults.dispatch_interval_ms
            ),
        ),
        network_workers=network_workers,
        network_connect_timeout_s=_positive_number(
            os.environ.get("ALPAMAYO_NETWORK_CONNECT_TIMEOUT_S"),
            _positive_number(
                values.get("network_connect_timeout_s"),
                defaults.network_connect_timeout_s,
            ),
        ),
        network_infer_timeout_s=_positive_number(
            os.environ.get("ALPAMAYO_NETWORK_INFER_TIMEOUT_S"),
            _positive_number(
                values.get("network_infer_timeout_s"),
                defaults.network_infer_timeout_s,
            ),
        ),
        network_health_interval_ms=_positive_number(
            os.environ.get("ALPAMAYO_NETWORK_HEALTH_INTERVAL_MS"),
            _positive_number(
                values.get("network_health_interval_ms"),
                defaults.network_health_interval_ms,
            ),
        ),
    )
    config.require_valid()
    return config


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #

LISTEN = "listen"
REMOTE = "remote"

TCP = "tcp"
UDP = "udp"

# Every endpoint this backend owns, with the protocol and direction the code
# uses it in.  Direction is a property of the component - this port is bound
# *here*, that address is sendto'd *to* - so a ``protocol``/``direction`` in
# config/config.json that contradicts this table is rejected rather than obeyed.
#
# Listen addresses may be a specific LAN address or ``0.0.0.0``.  The vehicle
# configures the PGX LAN address it sends to; it must not be told ``0.0.0.0``.
# ``waypoint`` is the opposite direction: this process sends to the vehicle.
_ENDPOINT_ROLES: Mapping[str, tuple[str, str]] = {
    "cimg": (TCP, LISTEN),
    "pose_udp": (UDP, LISTEN),
    "waypoint": (UDP, REMOTE),
    "route_tcp": (TCP, LISTEN),
    "http": (TCP, LISTEN),
}

# Production listen ports, and the vehicle address the PC relay used to forward
# waypoints to (UDP 7010).  ``127.0.0.1:7003`` was the relay's local downlink,
# not a vehicle port, and is rejected below.
_ENDPOINT_DEFAULTS: Mapping[str, tuple[str, int]] = {
    "cimg": ("0.0.0.0", 30106),
    "pose_udp": ("0.0.0.0", 7001),
    "waypoint": ("168.130.182.10", 7010),
    "route_tcp": ("0.0.0.0", 30006),
    "http": ("0.0.0.0", 8090),
}

# The environment variables each endpoint still honours.  Relay variables are
# not in this table: setting one is an error, handled by ``_reject_retired_relay``.
_ENDPOINT_ENV: Mapping[str, tuple[str, str]] = {
    "cimg": ("ALPAMAYO_CIMG_HOST", "ALPAMAYO_CIMG_PORT"),
    "pose_udp": ("ALPAMAYO_POSE_UDP_HOST", "ALPAMAYO_POSE_UDP_PORT"),
    "waypoint": ("ALPAMAYO_WAYPOINT_HOST", "ALPAMAYO_WAYPOINT_PORT"),
    "route_tcp": ("ALPAMAYO_ROUTE_TCP_HOST", "ALPAMAYO_ROUTE_TCP_PORT"),
    "http": ("ALPAMAYO_HOST", "ALPAMAYO_PORT"),
}

# Listeners this one process binds.  ``waypoint`` is a send target and is not
# in the list.  A TCP listener and a UDP listener on one port number never
# conflict.
_APP_PROCESS = ("cimg", "pose_udp", "route_tcp", "http")

# The PC relay is gone.  A shell or a document that still names it must fail
# here, rather than being ignored while the process starts on the direct ports.
_RETIRED_RELAY_ENV = (
    "PC_IP",
    "RELAY_HOST",
    "RELAY_PORT",
    "ALPAMAYO_RELAY_TCP_HOST",
    "ALPAMAYO_RELAY_TCP_PORT",
    "ALPAMAYO_RELAY_INJECT_HOST",
    "ALPAMAYO_RELAY_INJECT_PORT",
    "ALPAMAYO_RELAY_DOWNLINK_HOST",
    "ALPAMAYO_RELAY_DOWNLINK_PORT",
    "ALPAMAYO_PC_RELAY_SERVER_HOST",
    "ALPAMAYO_PC_RELAY_SERVER_PORT",
    "ALPAMAYO_PC_RELAY_SERVER_DIRECT_HOST",
    "ALPAMAYO_PC_RELAY_SERVER_DIRECT_PORT",
    "ALPAMAYO_PC_TUNNEL_RELAY_REMOTE_HOST",
    "ALPAMAYO_PC_TUNNEL_RELAY_REMOTE_PORT",
)
# MPEG-TS / RTSP input is gone.  A shell or a document that still names it
# must fail here, rather than being ignored while the process listens for CIMG.
_RETIRED_CAMERA_ENV = (
    "ALPAMAYO_CAMERA_INPUT",
    "ALPAMAYO_CAMERA_SOURCES",
    "ALPAMAYO_CAMERA_INDICES",
    "ALPAMAYO_FFMPEG",
    "ALPAMAYO_MPEGTS_LOG_DIR",
)
# The relay listened here and the backend used to send waypoints at it.
_RETIRED_WAYPOINT = ("127.0.0.1", 7003)

# ``0.0.0.0`` (and its IPv6 counterpart) covers every address on the host, so it
# overlaps anything else on the same port.  An empty or ``*`` host means the
# same to every ``bind()`` that accepts it.
_WILDCARD_HOSTS = frozenset({"", "*", "0.0.0.0", "::", "[::]"})

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

# The offline model-test API ``tools/request_model_test.py`` posts to, and the
# URL the PC's ``run_package_test.py`` passes it.  Both used to be the literal
# ``http://127.0.0.1:8091/infer``, written down twice in two repositories.
DEFAULT_PACKAGE_TEST_URL = "http://127.0.0.1:8091/infer"


def _endpoint_attr(role: str) -> str:
    """``"pose_udp"`` stays ``"pose_udp"``; a dotted role would become a field."""
    return role.replace(".", "_")


@dataclass(frozen=True)
class NetworkEndpoint:
    """One ``host:port`` together with how the code uses it.

    ``protocol`` and ``direction`` are carried rather than implied so the
    validator and the startup log can both say which of the two kinds of
    endpoint they are looking at, which is the distinction that decides whether
    an address may be a wildcard and whether two endpoints can coexist.
    """

    protocol: str
    direction: str
    host: str
    port: int

    @property
    def is_listen(self) -> bool:
        return self.direction == LISTEN

    def __str__(self) -> str:
        return f"{self.host}:{self.port} ({self.protocol} {self.direction})"


def _default_endpoint(role: str) -> NetworkEndpoint:
    protocol, direction = _ENDPOINT_ROLES[role]
    host, port = _ENDPOINT_DEFAULTS[role]
    return NetworkEndpoint(protocol, direction, host, port)


def default_endpoint(role: str) -> NetworkEndpoint:
    """The built-in default for ``role``.

    For the one caller that needs a fallback without a config object in hand:
    The CIMG receiver's own default host and port come from here, so a tool
    that builds a ``CimgConfig`` without a loaded document still binds the
    same ``network.cimg`` endpoint the backend listens on.
    """
    if role not in _ENDPOINT_ROLES:
        raise KeyError(f"unknown network endpoint {role!r}")
    return _default_endpoint(role)


def _as_port(role: str, raw: Any) -> int:
    """The port as an ``int``, or a ``ValueError`` naming the endpoint.

    Raised rather than collected in ``validate()`` because the value is needed
    to build the endpoint at all; the message carries the endpoint name so the
    fault still points at one line of config.json.  ``True`` is rejected even
    though it is an ``int``, because a boolean where a port belongs is always a
    typo.
    """
    if not isinstance(raw, bool):
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str):
            try:
                return int(raw.strip(), 10)
            except ValueError:
                pass
    raise ValueError(f"network.{role}.port must be an integer, got {raw!r}")


def _read_endpoint(role: str, values: Mapping[str, Any]) -> NetworkEndpoint:
    """Resolve one endpoint: environment, then JSON, then the default.

    An empty environment variable counts as unset, which is the contract the
    rest of this module documents and what ``_env_list`` already does; before
    this, ``ALPAMAYO_CIMG_HOST=""`` produced an empty bind address instead.
    """
    default = _default_endpoint(role)
    values = values or {}
    env_host, env_port = _ENDPOINT_ENV[role]

    declared_protocol = str(values.get("protocol", default.protocol)).strip().lower()
    declared_direction = (
        str(values.get("direction", default.direction)).strip().lower()
    )
    if (declared_protocol, declared_direction) != (default.protocol, default.direction):
        raise ValueError(
            f"network.{role} declares {declared_protocol}/{declared_direction}, "
            f"but this endpoint is {default.protocol}/{default.direction} by "
            "construction: the direction is fixed by the component that uses "
            "it, so protocol and direction are not configurable"
        )

    host = os.environ.get(env_host, "").strip() or str(
        values.get("host", default.host)
    ).strip()

    env_port_raw = os.environ.get(env_port, "").strip()
    port = _as_port(
        role, env_port_raw if env_port_raw else values.get("port", default.port)
    )

    return NetworkEndpoint(default.protocol, default.direction, host, port)


def _validate_http_url(name: str, value: str) -> str | None:
    """``None`` when ``value`` is an absolute http(s) URL, else the problem."""
    parsed = urlparse(str(value).strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return f"{name}={value!r} is not an absolute http(s) URL"
    return None


@dataclass(frozen=True)
class NetworkConfig:
    """Every communication endpoint this backend binds or reaches.

    One source: ``app.py`` and ``pipeline.py`` take the listeners and the
    waypoint target from ``AppConfig.network``.  The vehicle opens camera,
    pose and planned_route toward the listen addresses.  Waypoints are sent
    to ``waypoint``, which is the vehicle, not a local forwarder.
    """

    cimg: NetworkEndpoint = field(default_factory=lambda: _default_endpoint("cimg"))
    pose_udp: NetworkEndpoint = field(
        default_factory=lambda: _default_endpoint("pose_udp")
    )
    waypoint: NetworkEndpoint = field(
        default_factory=lambda: _default_endpoint("waypoint")
    )
    route_tcp: NetworkEndpoint = field(
        default_factory=lambda: _default_endpoint("route_tcp")
    )
    route_tcp_enabled: bool = True
    http: NetworkEndpoint = field(default_factory=lambda: _default_endpoint("http"))
    # The HuggingFace client base URL the model loader downloads through.
    # ``HF_ENDPOINT`` in the environment still wins, because the loader sets it
    # with ``setdefault`` and the two mirror each other.
    hf_endpoint: str = DEFAULT_HF_ENDPOINT
    # The offline model-test API this checkout's ``tools/request_model_test.py``
    # posts to by default.  It is a client target, not a listener, so it is a
    # URL rather than an endpoint pair: the scheme and path matter as much as
    # host:port.
    package_test_url: str = DEFAULT_PACKAGE_TEST_URL

    def endpoints(self) -> list[tuple[str, NetworkEndpoint]]:
        """``(role, endpoint)`` for every endpoint, in table order."""
        return [
            (role, getattr(self, _endpoint_attr(role))) for role in _ENDPOINT_ROLES
        ]

    def listen_conflicts(self) -> list[str]:
        """Endpoints one process cannot bind at the same time.

        Only listeners are compared, and only when the protocol matches.
        ``waypoint`` is a send and is not compared.  Two addresses conflict
        when either is a wildcard or when they are equal, so ``0.0.0.0:7001``
        and ``127.0.0.1:7001`` are reported rather than discovered at bind time.
        """
        problems: list[str] = []
        for process, roles in (("backend", _APP_PROCESS),):
            for index, first in enumerate(roles):
                for second in roles[index + 1 :]:
                    a = getattr(self, _endpoint_attr(first))
                    b = getattr(self, _endpoint_attr(second))
                    if not (a.is_listen and b.is_listen):
                        continue
                    if a.protocol != b.protocol or a.port != b.port:
                        continue
                    if not _addresses_overlap(a.host, b.host):
                        continue
                    problems.append(
                        f"network.{first} ({a}) and network.{second} ({b}) cannot "
                        f"both be bound by the {process} process: one {a.protocol} "
                        "port on overlapping addresses"
                    )
        return problems

    def validate(self) -> str | None:
        """Fast structural check; ``None`` when the config is usable.

        Returns a message rather than raising so ``/health`` can report a
        config problem and the loader can raise the same text, exactly as
        ``InferencePoolConfig.validate`` does.
        """
        for role, endpoint in self.endpoints():
            if not endpoint.host:
                return (
                    f"network.{role}.host must not be empty "
                    f"({endpoint.protocol} {endpoint.direction})"
                )
            if not 1 <= endpoint.port <= 65535:
                return (
                    f"network.{role}.port={endpoint.port} must be in 1..65535"
                )
            if not endpoint.is_listen and endpoint.host in _WILDCARD_HOSTS:
                return (
                    f"network.{role}.host={endpoint.host!r} is a wildcard, which "
                    f"cannot be a {endpoint.protocol} target; name the host that "
                    "actually receives this traffic"
                )
            if (
                role == "waypoint"
                and (endpoint.host, endpoint.port) == _RETIRED_WAYPOINT
            ):
                return (
                    "network.waypoint 127.0.0.1:7003 is the retired PC relay "
                    "downlink; set the vehicle address (UDP port 7010)"
                )
        conflicts = self.listen_conflicts()
        if conflicts:
            return conflicts[0]
        problem = _validate_http_url("network.http_client.hf_endpoint", self.hf_endpoint)
        if problem is not None:
            return problem
        problem = _validate_http_url(
            "network.http_client.package_test_url", self.package_test_url
        )
        if problem is not None:
            return problem
        return None

    def require_valid(self) -> None:
        problem = self.validate()
        if problem is not None:
            raise ValueError(problem)

    def with_endpoint(
        self, role: str, *, host: str | None = None, port: int | None = None
    ) -> NetworkConfig:
        """The same configuration with one endpoint replaced by command-line values.

        ``load_network_config`` checks the document, the environment and the
        defaults.  A rebuilt endpoint, validated, puts an out-of-range port on
        the same footing as a bad port in ``config/config.json``: a usage error
        before a listening socket exists, rather than a traceback from ``bind()``.

        ``role`` names the entry the way ``_ENDPOINT_ROLES`` and the document do.
        Protocol and direction come from the table rather than from the caller,
        so a flag cannot turn a listener into a target.  ``None`` means "this
        flag was not given" and keeps the current value.
        """
        if role not in _ENDPOINT_ROLES:
            raise KeyError(f"unknown network endpoint {role!r}")
        current = getattr(self, _endpoint_attr(role))
        return replace(
            self,
            **{
                _endpoint_attr(role): NetworkEndpoint(
                    current.protocol,
                    current.direction,
                    current.host if host is None else str(host).strip(),
                    current.port if port is None else port,
                )
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            role: {
                "protocol": endpoint.protocol,
                "direction": endpoint.direction,
                "host": endpoint.host,
                "port": endpoint.port,
            }
            for role, endpoint in self.endpoints()
        } | {
            "route_tcp_enabled": self.route_tcp_enabled,
            "hf_endpoint": self.hf_endpoint,
            "package_test_url": self.package_test_url,
        }


def _addresses_overlap(first: str, second: str) -> bool:
    """Whether two bind addresses can cover the same socket."""
    a, b = first.strip().lower(), second.strip().lower()
    if a in _WILDCARD_HOSTS or b in _WILDCARD_HOSTS:
        return True
    return a == b


def _named_document(resolved: Path, selector: str) -> Mapping[str, Any]:
    """Read a document the operator asked for by name, or refuse to guess.

    ``selector`` is the flag or variable that named it, and it is part of the
    message: "cannot read /tmp/x.json" leaves out the part the operator needs -
    which of the ways in picked a file that turned out not to be there.  Empty
    for the shipped path, which nobody named.
    """
    try:
        return _read_document(str(resolved))
    except Exception as exc:  # noqa: BLE001 - one message for every read failure
        source = f" (named by {selector})" if selector else ""
        raise ValueError(f"cannot read {resolved}{source}: {exc}") from exc


def _network_document(path: str | Path | None) -> Mapping[str, Any]:
    """The document the network section comes from, or ``{}`` for a legacy tree.

    ``{}`` is only for the one case where nothing was named and the shipped file
    is not there.  Everything else raises, including a document that exists but
    does not parse - falling back to the defaults there would move every socket
    without saying so.
    """
    named = os.environ.get("ALPAMAYO_CONFIG", "").strip()
    if path is not None:
        return _named_document(Path(path).resolve(), "--config")
    if named:
        return _named_document(Path(named).resolve(), "ALPAMAYO_CONFIG")

    default_path = resolve_config_path()
    if default_path is None:
        logger.warning(
            "no config/config.json; using the built-in network defaults"
        )
        return {}
    return _named_document(Path(default_path).resolve(), "")


def _reject_retired_relay(raw_network: Mapping[str, Any]) -> None:
    """Fail when a PC-relay setting is still present.

    Ignoring the key and starting on the direct ports would look like a
    successful migration while an operator's old file was never read.  An
    empty value counts as unset, matching the rest of this loader.
    """
    present = [
        name for name in _RETIRED_RELAY_ENV if os.environ.get(name, "").strip()
    ]
    if present:
        raise ValueError(
            "retired PC relay setting "
            + ", ".join(present)
            + " is set; the vehicle connects directly to this backend and "
            "these variables are not applied"
        )
    if "relay" in raw_network:
        raise ValueError(
            "network.relay is the retired PC relay; remove it. "
            "Pose is received on network.pose_udp and waypoints are sent "
            "to network.waypoint"
        )


def _reject_retired_camera(raw_network: Mapping[str, Any]) -> None:
    """Fail when an MPEG-TS / RTSP camera setting is still present.

    Ignoring the key and listening for CIMG would look like the old input
    list was still in effect.  An empty value counts as unset.
    """
    present = [
        name for name in _RETIRED_CAMERA_ENV if os.environ.get(name, "").strip()
    ]
    if present:
        raise ValueError(
            "retired MPEG-TS camera setting "
            + ", ".join(present)
            + " is set; cameras arrive as CIMG on network.cimg and "
            "these variables are not applied"
        )
    if "camera_sources" in raw_network:
        raise ValueError(
            "network.camera_sources is the retired MPEG-TS input list; "
            "remove it. Cameras arrive as CIMG on network.cimg"
        )


def load_network_config(path: str | Path | None = None) -> NetworkConfig:
    """Resolve every communication endpoint: environment, JSON, then default.

    The JSON section is new.  Before this, all eight values were read from the
    environment with hardcoded fallbacks and ``config/config.json`` had no
    ``network`` key at all, so a host or port written into the document was
    silently ignored.  Rejects an unusable configuration at load time, which
    runs inside ``AppConfig.from_env()`` - before a socket is opened - so a bad
    port is a startup error with a readable message rather than a bind failure
    an hour later.

    ``path`` names a document instead of the process-wide
    ``ALPAMAYO_CONFIG`` / ``config/config.json`` resolution.  Tests use it so
    a temporary file is not confused with the shipped configuration.

    A document that was named - by ``path``, by ``ALPAMAYO_CONFIG``, or because
    it is the shipped ``config/config.json`` - is never substituted with the
    defaults when it cannot be read.  Only the absence of the shipped file,
    which is what a deployment predating the ``network`` section looks like,
    means "use the built-in defaults"; a file that is there but unparseable is a
    mistake in something someone just edited, and starting the vehicle on a
    different set of addresses than the one they wrote down is worse than not
    starting.  See ``_network_document``.
    """
    document = _network_document(path)

    def section(container: Mapping[str, Any], name: str, path: str) -> Mapping[str, Any]:
        value = container.get(name, {})
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be an object")
        return value

    raw = section(document, "network", "network")
    _reject_retired_relay(raw)
    _reject_retired_camera(raw)

    config = NetworkConfig(
        cimg=_read_endpoint("cimg", section(raw, "cimg", "network.cimg")),
        pose_udp=_read_endpoint("pose_udp", section(raw, "pose_udp", "network.pose_udp")),
        waypoint=_read_endpoint("waypoint", section(raw, "waypoint", "network.waypoint")),
        route_tcp=_read_endpoint("route_tcp", section(raw, "route_tcp", "network.route_tcp")),
        route_tcp_enabled=_env_bool(
            "ALPAMAYO_ROUTE_TCP_ENABLED",
            bool(section(raw, "route_tcp", "network.route_tcp").get("enabled", True)),
        ),
        http=_read_endpoint("http", section(raw, "http", "network.http")),
        hf_endpoint=(
            os.environ.get("HF_ENDPOINT", "").strip()
            or str(
                section(raw, "http_client", "network.http_client").get(
                    "hf_endpoint", DEFAULT_HF_ENDPOINT
                )
            ).strip()
        ),
        package_test_url=(
            os.environ.get("ALPAMAYO_PACKAGE_TEST_URL", "").strip()
            or str(
                section(raw, "http_client", "network.http_client").get(
                    "package_test_url", DEFAULT_PACKAGE_TEST_URL
                )
            ).strip()
        ),
    )
    config.require_valid()
    return config


# --------------------------------------------------------------------------- #
# Effective configuration
# --------------------------------------------------------------------------- #

@dataclass
class AppConfig:
    """Effective runtime configuration.

    Field names, units, defaults and the JSON/env override order are identical
    to the former ``app.py::Settings``; these are a compatibility surface, not
    an internal detail.  The dataclass stays mutable because the offline model
    test overrides ``checkpoint`` / ``navigation_text`` on a loaded instance.
    """

    frames_per_camera: int = 4
    target_fps: float = 10.0
    frame_width: int = 576
    frame_height: int = 320
    max_camera_skew_us: int = 300_000
    reconnect_s: float = 2.0

    # CIMG direct JPEG push (the vehicle sends four JPEGs on one TCP stream).
    cimg_host: str = "0.0.0.0"
    cimg_port: int = 30106
    # Alpamayo index for CIMG camera_id 1..4, in that order.  The vehicle
    # numbers its cameras front/right/left/back on 30106, and Alpamayo indexes
    # front=1, right=2, left=0, back=4.
    cimg_camera_indices: list[int] = field(
        default_factory=lambda: [1, 2, 0, 4]
    )
    # Seconds without a complete packet before the connection is dropped so the
    # vehicle re-establishes it.
    cimg_read_timeout_s: float = 15.0
    cimg_max_payload_bytes: int = 64 * 1024 * 1024
    cimg_max_jpeg_bytes: int = 16 * 1024 * 1024
    # Frames buffered per camera.  60 is six seconds at 10 groups/s.
    cimg_queue_frames: int = 60
    # Complete packets allowed to wait in one read; beyond this the older ones
    # are dropped and only the newest is ingested.
    cimg_max_backlog_groups: int = 2
    cimg_group_fps: float = 10.0

    # Pose source: udp | mock
    pose_mode: str = "udp"
    pose_udp_host: str = "0.0.0.0"
    pose_udp_port: int = 7001
    azimuth_offset_deg: float = 0.0

    # Output UDP waypoint protocol.  This is the vehicle, not a local relay.
    waypoint_host: str = "168.130.182.10"
    waypoint_port: int = 7010
    magic: int = 0
    version: int = 1
    vehicle_id: int = 0
    byte_order: str = "<"
    allow_unconfigured_udp: bool = False

    # Model
    checkpoint: str = (
        "/data/users_data/wanghaohao/701/model_weights/Alpamayo-1.5-10B"
    )
    num_traj_samples: int = 1
    top_p: float = 0.98
    temperature: float = 0.6
    max_generation_length: int = 256
    # Optional manually configured route instruction for Alpamayo's official
    # helper.create_message(nav_text=...) prompt path.
    navigation_text: str = ""
    # Control is independent of model text; default is no takeover.
    control_flag: int = 0
    # Runtime control is always derived from planned-route deviation.
    control_mode: str = "trajectory_deviation"
    trajectory_dt_s: float = 0.1
    takeover_threshold_m: float = 0.8  # Initial field-tuning value.
    release_threshold_m: float = 0.4  # Initial field-tuning value.
    takeover_confirmations: int = 2
    release_confirmations: int = 2
    takeover_min_remaining_points: int = 10
    # Model trajectory span below this, while the route still reaches it, is a stop.
    stop_progress_m: float = 1.0
    reference_source: str = "planned_route"
    route_max_age_s: float = 30.0
    route_max_match_distance_m: float = 10.0
    route_tcp_enabled: bool = True
    route_tcp_host: str = "0.0.0.0"
    route_tcp_port: int = 30006

    # Ego history
    history_steps: int = 16
    history_dt_s: float = 0.1

    # General
    mock_mode: bool = True
    min_cycle_s: float = 1.0
    pose_stale_ms: float = 800.0
    # Camera window freshness: skip inference when t0 is older than the
    # latest pose by more than this (avoid stale-image control).
    camera_stale_ms: float = 1000.0
    # Maximum receive-time difference between camera and pose.  Both values
    # use server monotonic time, so this remains valid in every timestamp mode.
    camera_pose_delta_ms: float = 1000.0
    window_max_age_ms: int = 2000
    camera_frame_timeout_ms: int = 3000
    # Per-inference package directory. Empty disables package saving.
    infer_package_dir: str = ""
    # Pose-in and waypoint-out JSONL. Empty disables the trace. Files are
    # append-only and are never rotated or deleted by the process.
    vehicle_trace_dir: str = ""
    # Newest child directories kept under the snapshot and package roots.
    # Zero disables pruning.  See RetentionConfig.
    snapshot_keep: int = 600
    infer_package_keep: int = 600
    # How many GPUs run models and how tasks reach them.  Defaults to the
    # original single-GPU inline behaviour; see InferencePoolConfig.
    inference_pool: InferencePoolConfig = field(
        default_factory=InferencePoolConfig
    )
    # Every communication endpoint in one object, so a consumer can take the
    # endpoint it needs rather than the flat field it happens to be named after.
    # The flat ``*_host``/``*_port`` fields below keep their names - they are the
    # compatibility surface the tools and tests read - and are filled from here.
    network: NetworkConfig = field(default_factory=NetworkConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build the effective configuration (one call per process)."""
        mock_mode = _env_bool("ALPAMAYO_MOCK_MODE", True)
        cimg_idx_raw = os.environ.get(
            "ALPAMAYO_CIMG_CAMERA_INDICES", "1,2,0,4"
        )
        cimg_indices = [
            int(x.strip()) for x in cimg_idx_raw.split(",") if x.strip()
        ]
        pose_mode_default = "mock" if mock_mode else "udp"
        freshness = load_camera_freshness_config()
        heading = load_vehicle_heading_config()
        retention = load_retention_config()
        network = load_network_config()
        return cls(
            frames_per_camera=int(os.environ.get("ALPAMAYO_FRAMES_PER_CAMERA", "4")),
            target_fps=float(os.environ.get("ALPAMAYO_TARGET_FPS", "10")),
            frame_width=int(os.environ.get("ALPAMAYO_FRAME_WIDTH", "576")),
            frame_height=int(os.environ.get("ALPAMAYO_FRAME_HEIGHT", "320")),
            max_camera_skew_us=int(
                os.environ.get("ALPAMAYO_MAX_CAMERA_SKEW_US", "300000")
            ),
            cimg_host=network.cimg.host,
            cimg_port=network.cimg.port,
            cimg_camera_indices=cimg_indices,
            cimg_read_timeout_s=float(
                os.environ.get("ALPAMAYO_CIMG_READ_TIMEOUT_S", "15")
            ),
            cimg_max_payload_bytes=int(
                os.environ.get("ALPAMAYO_CIMG_MAX_PAYLOAD_BYTES", str(64 << 20))
            ),
            cimg_max_jpeg_bytes=int(
                os.environ.get("ALPAMAYO_CIMG_MAX_JPEG_BYTES", str(16 << 20))
            ),
            cimg_queue_frames=int(
                os.environ.get("ALPAMAYO_CIMG_QUEUE_FRAMES", "60")
            ),
            cimg_max_backlog_groups=int(
                os.environ.get("ALPAMAYO_CIMG_MAX_BACKLOG_GROUPS", "2")
            ),
            cimg_group_fps=float(
                os.environ.get("ALPAMAYO_CIMG_GROUP_FPS", "10")
            ),
            pose_mode=os.environ.get("ALPAMAYO_POSE_MODE", pose_mode_default)
            .strip()
            .lower(),
            pose_udp_host=network.pose_udp.host,
            pose_udp_port=network.pose_udp.port,
            azimuth_offset_deg=heading.azimuth_offset_deg,
            waypoint_host=network.waypoint.host,
            waypoint_port=network.waypoint.port,
            magic=int(os.environ.get("ALPAMAYO_WAYPOINT_MAGIC", "0"), 0),
            version=int(os.environ.get("ALPAMAYO_WAYPOINT_VERSION", "1")),
            vehicle_id=int(os.environ.get("ALPAMAYO_VEHICLE_ID", "0")),
            byte_order=os.environ.get("ALPAMAYO_BYTE_ORDER", "<"),
            allow_unconfigured_udp=_env_bool(
                "ALPAMAYO_ALLOW_UNCONFIGURED_UDP",
                mock_mode,
            ),
            checkpoint=os.environ.get(
                "ALPAMAYO_CHECKPOINT",
                "/data/users_data/wanghaohao/701/model_weights/Alpamayo-1.5-10B",
            ),
            num_traj_samples=int(
                os.environ.get("ALPAMAYO_NUM_TRAJ_SAMPLES", "1")
            ),
            top_p=float(os.environ.get("ALPAMAYO_TOP_P", "0.98")),
            temperature=float(os.environ.get("ALPAMAYO_TEMPERATURE", "0.6")),
            max_generation_length=int(
                os.environ.get("ALPAMAYO_MAX_GENERATION_LENGTH", "256")
            ),
            # Keep the existing environment variable as a compatibility
            # override while making the JSON field the normal manual source.
            navigation_text=load_navigation_text(),
            control_flag=(
                1
                if os.environ.get("ALPAMAYO_CONTROL_FLAG", "0").strip() == "1"
                else 0
            ),
            control_mode="trajectory_deviation",
            trajectory_dt_s=float(
                os.environ.get("ALPAMAYO_TRAJECTORY_DT_S", "0.1")
            ),
            takeover_threshold_m=float(
                os.environ.get("ALPAMAYO_TAKEOVER_THRESHOLD_M", "0.8")
            ),
            release_threshold_m=float(
                os.environ.get("ALPAMAYO_RELEASE_THRESHOLD_M", "0.4")
            ),
            takeover_confirmations=int(
                os.environ.get("ALPAMAYO_TAKEOVER_CONFIRMATIONS", "2")
            ),
            release_confirmations=int(
                os.environ.get("ALPAMAYO_RELEASE_CONFIRMATIONS", "2")
            ),
            takeover_min_remaining_points=int(
                os.environ.get("ALPAMAYO_TAKEOVER_MIN_REMAINING_POINTS", "10")
            ),
            stop_progress_m=float(
                os.environ.get("ALPAMAYO_STOP_PROGRESS_M", "1.0")
            ),
            reference_source="planned_route",
            route_max_age_s=float(os.environ.get("ALPAMAYO_ROUTE_MAX_AGE_S", "30")),
            route_max_match_distance_m=float(os.environ.get("ALPAMAYO_ROUTE_MAX_MATCH_DISTANCE_M", "10")),
            route_tcp_enabled=network.route_tcp_enabled,
            route_tcp_host=network.route_tcp.host,
            route_tcp_port=network.route_tcp.port,
            history_steps=int(os.environ.get("ALPAMAYO_HISTORY_STEPS", "16")),
            history_dt_s=float(os.environ.get("ALPAMAYO_HISTORY_DT_S", "0.1")),
            mock_mode=mock_mode,
            min_cycle_s=float(os.environ.get("ALPAMAYO_MIN_CYCLE_S", "1.0")),
            pose_stale_ms=float(
                os.environ.get("ALPAMAYO_POSE_STALE_MS", "800")
            ),
            camera_stale_ms=freshness.camera_stale_ms,
            camera_pose_delta_ms=freshness.camera_pose_delta_ms,
            window_max_age_ms=freshness.window_max_age_ms,
            camera_frame_timeout_ms=freshness.frame_timeout_ms,
            infer_package_dir=os.environ.get(
                "ALPAMAYO_INFER_PACKAGE_DIR", ""
            ).strip(),
            vehicle_trace_dir=os.environ.get(
                "ALPAMAYO_VEHICLE_TRACE_DIR", ""
            ).strip(),
            snapshot_keep=retention.snapshot_keep,
            infer_package_keep=retention.infer_package_keep,
            inference_pool=load_inference_pool_config(),
            network=network,
        )


def load_app_config() -> AppConfig:
    """Load the effective configuration exactly once per process."""
    return AppConfig.from_env()
