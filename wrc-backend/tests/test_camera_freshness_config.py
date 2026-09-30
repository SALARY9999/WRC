from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.loader import (
    CameraFreshnessConfig,
    load_camera_freshness_config,
)


class CameraFreshnessConfigTests(unittest.TestCase):
    def _load(self, document: dict) -> CameraFreshnessConfig:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "config.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with patch.dict(os.environ, {"ALPAMAYO_CONFIG": str(path)}, clear=True):
            return load_camera_freshness_config()

    def test_defaults_when_section_is_missing(self) -> None:
        self.assertEqual(self._load({}), CameraFreshnessConfig())

    def test_custom_values_are_loaded(self) -> None:
        config = self._load(
            {
                "camera_freshness": {
                    "camera_stale_ms": 1800,
                    "camera_pose_delta_ms": 1700.5,
                    "window_max_age_ms": 2800,
                    "frame_timeout_ms": 4500,
                }
            }
        )
        self.assertEqual(config.camera_stale_ms, 1800)
        self.assertEqual(config.camera_pose_delta_ms, 1700.5)
        self.assertEqual(config.window_max_age_ms, 2800)
        self.assertEqual(config.frame_timeout_ms, 4500)

    def test_missing_fields_fall_back_individually(self) -> None:
        config = self._load({"camera_freshness": {"camera_stale_ms": 1500}})
        self.assertEqual(config.camera_stale_ms, 1500)
        self.assertEqual(config.camera_pose_delta_ms, 1000)
        self.assertEqual(config.window_max_age_ms, 2000)
        self.assertEqual(config.frame_timeout_ms, 3000)

    def test_invalid_values_use_safe_defaults(self) -> None:
        config = self._load(
            {
                "camera_freshness": {
                    "camera_stale_ms": -1,
                    "camera_pose_delta_ms": "invalid",
                    "window_max_age_ms": 0,
                    "frame_timeout_ms": None,
                }
            }
        )
        self.assertEqual(config, CameraFreshnessConfig())

    def test_window_minimum_is_preserved(self) -> None:
        config = self._load({"camera_freshness": {"window_max_age_ms": 100}})
        self.assertEqual(config.window_max_age_ms, 200)

    def test_legacy_environment_variables_do_not_override_json(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "config.json"
        path.write_text(
            json.dumps({"camera_freshness": {"camera_stale_ms": 1600}}),
            encoding="utf-8",
        )
        with patch.dict(
            os.environ,
            {
                "ALPAMAYO_CONFIG": str(path),
                "ALPAMAYO_CAMERA_STALE_MS": "9999",
                "ALPAMAYO_CAMERA_POSE_DELTA_MS": "9999",
                "ALPAMAYO_WINDOW_MAX_AGE_MS": "9999",
                "ALPAMAYO_CAMERA_FRAME_TIMEOUT_MS": "9999",
            },
            clear=True,
        ):
            config = load_camera_freshness_config()
        self.assertEqual(config.camera_stale_ms, 1600)
        self.assertEqual(config.camera_pose_delta_ms, 1000)
        self.assertEqual(config.window_max_age_ms, 2000)
        self.assertEqual(config.frame_timeout_ms, 3000)


if __name__ == "__main__":
    unittest.main()
