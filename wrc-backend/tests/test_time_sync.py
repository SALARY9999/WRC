from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from time_sync import TimestampManager, TimestampManagerConfig


class TimestampManagerTests(unittest.TestCase):
    def test_capture_time_ns_becomes_microseconds(self) -> None:
        capture_ns = 1_800_000_000_123_456_000
        timestamp_us = TimestampManager().resolve_camera(capture_ns)
        self.assertEqual(timestamp_us, capture_ns // 1000)

    def test_pose_seconds_become_microseconds(self) -> None:
        timestamp_us = TimestampManager().resolve_pose(1_800_000_000.25)
        self.assertEqual(timestamp_us, 1_800_000_000_250_000)

    def test_a_timestamp_ahead_of_the_server_is_kept(self) -> None:
        capture_ns = 9_000_000_000_000_000_000
        self.assertEqual(
            TimestampManager().resolve_camera(capture_ns),
            capture_ns // 1000,
        )
        self.assertEqual(
            TimestampManager().resolve_pose(9_000_000_000),
            9_000_000_000_000_000,
        )

    def test_a_timestamp_before_2000_is_kept(self) -> None:
        self.assertEqual(TimestampManager().resolve_camera(1_000), 1)
        self.assertEqual(TimestampManager().resolve_pose(1), 1_000_000)

    def test_retired_clock_settings_are_rejected(self) -> None:
        retired = {
            "camera_timestamp_mode": "sei_unix",
            "camera_unix_unit": "milliseconds",
            "enabled": False,
            "pose_timestamp_mode": "arrival",
            "arrival_fallback": True,
            "pose_unix_unit": "milliseconds",
            "max_future_offset_ms": 200,
        }
        for name, value in retired.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    TimestampManagerConfig.from_mapping({name: value})
                self.assertIn(name, str(caught.exception))

    def test_an_empty_time_sync_object_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(
                json.dumps({"time_sync": {}}),
                encoding="utf-8",
            )
            TimestampManagerConfig.from_file(path)
            manager = TimestampManager.from_file(path)
        self.assertIsNone(manager.config_error)

    def test_a_file_without_time_sync_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(json.dumps({"version": 1}), encoding="utf-8")
            manager = TimestampManager.from_file(path)
        self.assertIsInstance(manager.config, TimestampManagerConfig)


if __name__ == "__main__":
    unittest.main()
