"""Convert camera and vehicle timestamps onto one Unix microsecond axis.

Camera time is the CIMG packet's ``capture_time_ns``. Pose time is the
vehicle's Unix timestamp in seconds. Both are always present; this module
only changes the unit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from config.loader import resolve_config_path

# Writing any of these looks like the clock can still be selected or bounded.
_RETIRED_KEYS = (
    "enabled",
    "pose_timestamp_mode",
    "arrival_fallback",
    "pose_unix_unit",
    "max_future_offset_ms",
    "camera_timestamp_mode",
    "camera_unix_unit",
)


class TimestampManagerConfig:
    """No clock settings remain. Retired keys fail at load time."""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TimestampManagerConfig":
        _reject_retired(values)
        return cls()

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "TimestampManagerConfig":
        """Load time fields from a JSON file.

        The file may contain the time fields directly or below ``time_sync``.
        """

        config_path = Path(path)
        with config_path.open(encoding="utf-8") as fh:
            document = json.load(fh)
        if not isinstance(document, Mapping):
            raise ValueError("time sync config root must be an object")
        values = document.get("time_sync", document)
        if not isinstance(values, Mapping):
            raise ValueError("time_sync config must be an object")
        return cls.from_mapping(values)


class TimestampManager:
    """Convert the timestamps the runtime pipeline aligns on."""

    def __init__(
        self,
        config: TimestampManagerConfig | None = None,
        *,
        config_error: str | None = None,
    ) -> None:
        self.config = config if config is not None else TimestampManagerConfig()
        self.config_error = config_error

    @classmethod
    def from_config(cls) -> "TimestampManager":
        """Load time-sync configuration from the unified JSON file."""
        config_path = resolve_config_path()
        if config_path is None:
            return cls()
        return cls(TimestampManagerConfig.from_file(config_path))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "TimestampManager":
        return cls(TimestampManagerConfig.from_file(path))

    def resolve_camera(self, capture_ns: int) -> int:
        """CIMG ``capture_time_ns`` as Unix microseconds."""
        return int(capture_ns) // 1000

    def resolve_pose(self, unix_timestamp: int | float) -> int:
        """Vehicle Unix seconds as Unix microseconds."""
        return int(float(unix_timestamp) * 1_000_000)


def _reject_retired(values: Mapping[str, Any]) -> None:
    retired = [name for name in _RETIRED_KEYS if name in values]
    if not retired:
        return
    names = ", ".join(retired)
    raise ValueError(
        f"retired time sync setting {names}; "
        "camera time is capture_time_ns and pose time is Unix seconds"
    )


__all__ = [
    "TimestampManager",
    "TimestampManagerConfig",
]
