from __future__ import annotations

import time
import unittest
from collections import Counter, deque

try:
    import numpy as np
    from camera.sync import CameraWindowProvider, FrameEntry, SynchronizedGroup
except ModuleNotFoundError as exc:  # Local lightweight environment.
    raise unittest.SkipTest(f"camera runtime dependency unavailable: {exc}")


def _group(timestamp_us: int, arrival_us: int) -> SynchronizedGroup:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    return SynchronizedGroup(
        target_timestamp_us=timestamp_us,
        packets={
            0: FrameEntry(timestamp_us, frame, pts=10, arrival_us=arrival_us),
            1: FrameEntry(timestamp_us, frame, pts=20, arrival_us=arrival_us),
        },
        max_skew_us=0,
    )


def _provider(group: SynchronizedGroup) -> CameraWindowProvider:
    provider = object.__new__(CameraWindowProvider)
    provider.frames_per_camera = 1
    provider.camera_indices = [0, 1]
    provider._groups = deque([group], maxlen=1)
    provider._last_returned_key = None
    provider.window_max_age_us = 1_000_000
    provider.refusals = Counter()
    provider._last_stale_log_at = 0.0
    provider._next_target_us = group.target_timestamp_us + 100_000
    provider._has_unhealthy_reader = lambda: False
    provider._fill_window = lambda _timeout: True
    provider._read_group = lambda **_kwargs: False
    provider._window_too_old = lambda: False
    provider._build = lambda: object()
    return provider


class CameraWindowFreshnessTests(unittest.TestCase):
    def test_latest_window_is_returned_only_once_without_new_frames(self) -> None:
        now_us = time.monotonic_ns() // 1000
        provider = _provider(_group(now_us, now_us))

        self.assertIsNotNone(provider.latest_window())
        self.assertIsNone(provider.latest_window())
        self.assertEqual(
            provider.refusals["refusing cached window: already returned"], 1
        )

    def test_stale_arrival_invalidates_cached_window(self) -> None:
        now_us = time.monotonic_ns() // 1000
        provider = _provider(_group(now_us, now_us - 2_000_000))

        self.assertIsNone(provider.latest_window())
        self.assertFalse(provider._groups)
        self.assertIsNone(provider._next_target_us)


if __name__ == "__main__":
    unittest.main()
