"""Config contract: defaults, JSON fields and environment overrides.

These expectations are written out independently of ``config/loader.py`` so a
drift in a default value or an override order fails here rather than on the
vehicle.  The matrix mirrors what ``run_backend_real.sh`` actually injects.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.loader import (
    AppConfig,
    CameraFreshnessConfig,
    load_app_config,
    load_camera_freshness_config,
    load_navigation_text,
)

# Every ALPAMAYO_* variable the loader may consult; cleared before each case so
# the developer's own shell cannot influence the result.
_MANAGED = (
    "ALPAMAYO_MOCK_MODE", "ALPAMAYO_POSE_MODE",
    "ALPAMAYO_CAMERA_INPUT", "ALPAMAYO_CAMERA_SOURCES", "ALPAMAYO_CAMERA_INDICES",
    "ALPAMAYO_FFMPEG", "ALPAMAYO_MPEGTS_LOG_DIR",
    "ALPAMAYO_FRAMES_PER_CAMERA", "ALPAMAYO_TARGET_FPS",
    "ALPAMAYO_FRAME_WIDTH", "ALPAMAYO_FRAME_HEIGHT", "ALPAMAYO_MAX_CAMERA_SKEW_US",
    "ALPAMAYO_CIMG_CAMERA_INDICES", "ALPAMAYO_CIMG_GROUP_FPS",
    "ALPAMAYO_POSE_UDP_PORT", "ALPAMAYO_WAYPOINT_HOST", "ALPAMAYO_WAYPOINT_PORT",
    "ALPAMAYO_WAYPOINT_MAGIC", "ALPAMAYO_WAYPOINT_VERSION", "ALPAMAYO_VEHICLE_ID",
    "ALPAMAYO_BYTE_ORDER", "ALPAMAYO_ALLOW_UNCONFIGURED_UDP", "ALPAMAYO_CHECKPOINT",
    "ALPAMAYO_NUM_TRAJ_SAMPLES", "ALPAMAYO_TOP_P", "ALPAMAYO_TEMPERATURE",
    "ALPAMAYO_MAX_GENERATION_LENGTH", "ALPAMAYO_NAV_TEXT", "ALPAMAYO_CONTROL_FLAG",
    "ALPAMAYO_CONTROL_MODE", "ALPAMAYO_TRAJECTORY_DT_S",
    "ALPAMAYO_TAKEOVER_THRESHOLD_M", "ALPAMAYO_RELEASE_THRESHOLD_M",
    "ALPAMAYO_TAKEOVER_CONFIRMATIONS", "ALPAMAYO_RELEASE_CONFIRMATIONS",
    "ALPAMAYO_TAKEOVER_MIN_REMAINING_POINTS", "ALPAMAYO_REFERENCE_SOURCE",
    "ALPAMAYO_ROUTE_MAX_AGE_S", "ALPAMAYO_ROUTE_MAX_MATCH_DISTANCE_M",
    "ALPAMAYO_ROUTE_TCP_ENABLED", "ALPAMAYO_ROUTE_TCP_HOST", "ALPAMAYO_ROUTE_TCP_PORT",
    "ALPAMAYO_HISTORY_STEPS", "ALPAMAYO_HISTORY_DT_S", "ALPAMAYO_MIN_CYCLE_S",
    "ALPAMAYO_POSE_STALE_MS", "ALPAMAYO_INFER_PACKAGE_DIR",
    "ALPAMAYO_VEHICLE_TRACE_DIR", "ALPAMAYO_CONFIG",
    "ALPAMAYO_DISPATCH_INTERVAL_MS", "ALPAMAYO_NETWORK_WORKERS",
    "ALPAMAYO_NETWORK_CONNECT_TIMEOUT_S", "ALPAMAYO_NETWORK_INFER_TIMEOUT_S",
    "ALPAMAYO_NETWORK_HEALTH_INTERVAL_MS",
    "ALPAMAYO_SNAPSHOT_KEEP", "ALPAMAYO_INFER_PACKAGE_KEEP",
    # The network section's variables (config/loader.py, load_network_config):
    # the endpoints every listener and target is built from.
    "ALPAMAYO_CIMG_HOST", "ALPAMAYO_CIMG_PORT",
    "ALPAMAYO_POSE_UDP_HOST",
    "ALPAMAYO_HOST", "ALPAMAYO_PORT",
    "ALPAMAYO_RELAY_TCP_HOST", "ALPAMAYO_RELAY_TCP_PORT",
    "ALPAMAYO_RELAY_INJECT_HOST", "ALPAMAYO_RELAY_INJECT_PORT",
    "ALPAMAYO_RELAY_DOWNLINK_HOST", "ALPAMAYO_RELAY_DOWNLINK_PORT",
    "HF_ENDPOINT", "ALPAMAYO_PACKAGE_TEST_URL",
)


class _CleanEnv(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in _MANAGED:
            self.addCleanup(os.environ.pop, name, None)
            os.environ.pop(name, None)


class DefaultValueTests(_CleanEnv):
    def test_vehicle_facing_defaults_are_unchanged(self):
        cfg = AppConfig.from_env()
        # Ports and protocol identity are a wire contract.
        self.assertEqual(cfg.pose_udp_port, 7001)
        self.assertEqual(cfg.vehicle_trace_dir, "")
        self.assertEqual(cfg.waypoint_port, 7010)
        self.assertEqual(cfg.route_tcp_port, 30006)
        self.assertEqual(cfg.waypoint_host, "168.130.182.10")
        self.assertEqual(cfg.byte_order, "<")
        self.assertEqual(cfg.version, 1)
        self.assertEqual(cfg.magic, 0)
        self.assertEqual(cfg.vehicle_id, 0)

    def test_model_and_trajectory_defaults_are_unchanged(self):
        cfg = AppConfig.from_env()
        self.assertEqual(cfg.frames_per_camera, 4)
        self.assertEqual(cfg.history_steps, 16)
        self.assertAlmostEqual(cfg.history_dt_s, 0.1)
        self.assertAlmostEqual(cfg.trajectory_dt_s, 0.1)
        self.assertEqual(cfg.num_traj_samples, 1)
        self.assertAlmostEqual(cfg.top_p, 0.98)
        self.assertAlmostEqual(cfg.temperature, 0.6)
        self.assertEqual(cfg.max_generation_length, 256)

    def test_cimg_input_cadence_defaults_to_ten_hz(self):
        cfg = AppConfig.from_env()
        self.assertAlmostEqual(cfg.cimg_group_fps, 10.0)

    def test_takeover_defaults_are_unchanged(self):
        cfg = AppConfig.from_env()
        self.assertAlmostEqual(cfg.takeover_threshold_m, 0.8)
        self.assertAlmostEqual(cfg.release_threshold_m, 0.4)
        self.assertEqual(cfg.takeover_confirmations, 2)
        self.assertEqual(cfg.release_confirmations, 2)
        self.assertEqual(cfg.takeover_min_remaining_points, 10)
        self.assertAlmostEqual(cfg.stop_progress_m, 1.0)

    def test_mock_mode_defaults_are_unchanged(self):
        cfg = AppConfig.from_env()
        self.assertTrue(cfg.mock_mode)
        self.assertEqual(cfg.pose_mode, "mock")
        self.assertTrue(cfg.allow_unconfigured_udp)
        self.assertEqual(cfg.cimg_host, "0.0.0.0")
        self.assertEqual(cfg.cimg_port, 30106)
        self.assertEqual(cfg.cimg_camera_indices, [1, 2, 0, 4])
        self.assertAlmostEqual(cfg.cimg_read_timeout_s, 15.0)
        self.assertEqual(cfg.cimg_max_payload_bytes, 64 * 1024 * 1024)
        self.assertEqual(cfg.cimg_max_jpeg_bytes, 16 * 1024 * 1024)
        self.assertEqual(cfg.cimg_queue_frames, 60)
        self.assertEqual(cfg.cimg_max_backlog_groups, 2)

    def test_control_mode_and_reference_source_are_not_env_driven(self):
        # The launcher no longer exports these.  The loader must still ignore
        # them: re-activating either one is a vehicle-behaviour change.
        with patch.dict(os.environ, {
            "ALPAMAYO_CONTROL_MODE": "fixed",
            "ALPAMAYO_REFERENCE_SOURCE": "history_pose",
            "ALPAMAYO_MOCK_MODE": "0",
        }, clear=False):
            cfg = AppConfig.from_env()
        self.assertEqual(cfg.control_mode, "trajectory_deviation")
        self.assertEqual(cfg.reference_source, "planned_route")

    def test_freshness_defaults_come_from_json_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({}), encoding="utf-8")
            with patch.dict(os.environ, {"ALPAMAYO_CONFIG": str(config)}, clear=False):
                cfg = AppConfig.from_env()
        self.assertAlmostEqual(cfg.camera_stale_ms, 1000.0)
        self.assertAlmostEqual(cfg.camera_pose_delta_ms, 1000.0)
        self.assertEqual(cfg.window_max_age_ms, 2000)
        self.assertEqual(cfg.camera_frame_timeout_ms, 3000)

    def test_retention_defaults_to_ten_minutes_at_one_hertz(self):
        cfg = AppConfig.from_env()
        self.assertEqual(cfg.snapshot_keep, 600)
        self.assertEqual(cfg.infer_package_keep, 600)


class EnvironmentOverrideTests(_CleanEnv):
    def test_retired_mpegts_settings_are_rejected(self):
        with patch.dict(os.environ, {"ALPAMAYO_CAMERA_INPUT": "cimg"}, clear=False):
            with self.assertRaises(ValueError) as caught:
                AppConfig.from_env()
        self.assertIn("ALPAMAYO_CAMERA_INPUT", str(caught.exception))

    def test_production_script_environment_is_honoured(self):
        env = {
            "ALPAMAYO_MOCK_MODE": "0",
            "ALPAMAYO_POSE_MODE": "udp",
            "ALPAMAYO_CIMG_CAMERA_INDICES": "1,2,0,4",
            "ALPAMAYO_WAYPOINT_MAGIC": "0x514D3031",
            "ALPAMAYO_VEHICLE_ID": "1",
            "ALPAMAYO_CONTROL_FLAG": "0",
            "ALPAMAYO_ROUTE_TCP_ENABLED": "true",
            "ALPAMAYO_ROUTE_TCP_PORT": "30006",
            "ALPAMAYO_INFER_PACKAGE_DIR": "/data/users_data/wanghaohao/infer_packages",
            "ALPAMAYO_VEHICLE_TRACE_DIR": "/data/users_data/wanghaohao/vehicle_trace",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = AppConfig.from_env()
        self.assertFalse(cfg.mock_mode)
        self.assertEqual(cfg.pose_mode, "udp")
        self.assertFalse(cfg.allow_unconfigured_udp)
        self.assertEqual(cfg.cimg_camera_indices, [1, 2, 0, 4])
        self.assertEqual(cfg.magic, 0x514D3031)
        self.assertEqual(cfg.vehicle_id, 1)
        self.assertEqual(cfg.control_flag, 0)
        self.assertTrue(cfg.route_tcp_enabled)
        self.assertEqual(cfg.route_tcp_port, 30006)
        self.assertEqual(
            cfg.vehicle_trace_dir, "/data/users_data/wanghaohao/vehicle_trace"
        )
        self.assertFalse(hasattr(cfg, "camera_sources"))

    def test_retention_environment_overrides_json_and_zero_disables(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text(
                json.dumps({"retention": {"snapshot_keep": 12, "infer_package_keep": 8}}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"ALPAMAYO_CONFIG": str(config)}, clear=False):
                from_json = AppConfig.from_env()
            self.assertEqual(from_json.snapshot_keep, 12)
            self.assertEqual(from_json.infer_package_keep, 8)
            with patch.dict(
                os.environ,
                {
                    "ALPAMAYO_CONFIG": str(config),
                    "ALPAMAYO_SNAPSHOT_KEEP": "3",
                    "ALPAMAYO_INFER_PACKAGE_KEEP": "0",
                },
                clear=False,
            ):
                overridden = AppConfig.from_env()
        self.assertEqual(overridden.snapshot_keep, 3)
        self.assertEqual(overridden.infer_package_keep, 0)

    def test_scalar_overrides_are_read_verbatim(self):
        env = {
            "ALPAMAYO_MOCK_MODE": "0",
            "ALPAMAYO_FRAMES_PER_CAMERA": "2",
            "ALPAMAYO_TARGET_FPS": "5",
            "ALPAMAYO_CIMG_GROUP_FPS": "12.5",
            "ALPAMAYO_TAKEOVER_THRESHOLD_M": "1.5",
            "ALPAMAYO_RELEASE_THRESHOLD_M": "0.5",
            "ALPAMAYO_TAKEOVER_CONFIRMATIONS": "4",
            "ALPAMAYO_RELEASE_CONFIRMATIONS": "3",
            "ALPAMAYO_TAKEOVER_MIN_REMAINING_POINTS": "7",
            "ALPAMAYO_STOP_PROGRESS_M": "1.5",
            "ALPAMAYO_HISTORY_STEPS": "8",
            "ALPAMAYO_HISTORY_DT_S": "0.2",
            "ALPAMAYO_BYTE_ORDER": ">",
            "ALPAMAYO_WAYPOINT_VERSION": "2",
            "ALPAMAYO_MIN_CYCLE_S": "0.5",
            "ALPAMAYO_POSE_STALE_MS": "500",
            "ALPAMAYO_ARBITRARY_FUTURE_KEY": "ignored",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = AppConfig.from_env()
        self.assertEqual(cfg.frames_per_camera, 2)
        self.assertAlmostEqual(cfg.target_fps, 5.0)
        self.assertAlmostEqual(cfg.cimg_group_fps, 12.5)
        self.assertAlmostEqual(cfg.takeover_threshold_m, 1.5)
        self.assertAlmostEqual(cfg.release_threshold_m, 0.5)
        self.assertEqual(cfg.takeover_confirmations, 4)
        self.assertEqual(cfg.release_confirmations, 3)
        self.assertEqual(cfg.takeover_min_remaining_points, 7)
        self.assertAlmostEqual(cfg.stop_progress_m, 1.5)
        self.assertEqual(cfg.history_steps, 8)
        self.assertAlmostEqual(cfg.history_dt_s, 0.2)
        self.assertEqual(cfg.byte_order, ">")
        self.assertEqual(cfg.version, 2)
        self.assertAlmostEqual(cfg.min_cycle_s, 0.5)
        self.assertAlmostEqual(cfg.pose_stale_ms, 500.0)

    def test_control_flag_only_accepts_exactly_one(self):
        for raw, expected in [("1", 1), ("0", 0), ("yes", 0), ("", 0), ("2", 0)]:
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {"ALPAMAYO_CONTROL_FLAG": raw}, clear=False):
                    self.assertEqual(AppConfig.from_env().control_flag, expected)

    def test_mock_mode_accepts_the_usual_truthy_spellings(self):
        for raw, expected in [("1", True), ("true", True), ("yes", True),
                              ("on", True), ("0", False), ("no", False), ("", False)]:
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {"ALPAMAYO_MOCK_MODE": raw}, clear=False):
                    self.assertIs(AppConfig.from_env().mock_mode, expected)


class JsonDocumentTests(_CleanEnv):
    def _with_config(self, document: dict, env: dict | None = None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        config = Path(tmp.name) / "config.json"
        config.write_text(json.dumps(document), encoding="utf-8")
        merged = {"ALPAMAYO_CONFIG": str(config)}
        merged.update(env or {})
        with patch.dict(os.environ, merged, clear=False):
            return {
                "app": AppConfig.from_env(),
                "freshness": load_camera_freshness_config(),
                "nav": load_navigation_text(),
            }

    def test_shipped_config_json_values_are_loaded(self):
        result = self._with_config({
            "version": 1,
            "navigation_text": "follow the planned route",
            "vehicle_heading": {"azimuth_offset_deg": 180},
            "camera_freshness": {
                "camera_stale_ms": 1500,
                "camera_pose_delta_ms": 900,
                "window_max_age_ms": 2500,
                "frame_timeout_ms": 4000,
            },
        })
        self.assertAlmostEqual(result["app"].camera_stale_ms, 1500.0)
        self.assertAlmostEqual(result["app"].camera_pose_delta_ms, 900.0)
        self.assertEqual(result["app"].window_max_age_ms, 2500)
        self.assertEqual(result["app"].camera_frame_timeout_ms, 4000)
        self.assertEqual(result["app"].navigation_text, "follow the planned route")
        self.assertEqual(result["app"].azimuth_offset_deg, 180.0)

    def test_vehicle_heading_rejects_non_finite_offset(self):
        with self.assertRaisesRegex(ValueError, "azimuth_offset_deg"):
            self._with_config({
                "vehicle_heading": {"azimuth_offset_deg": "nan"},
            })

    def test_env_nav_text_overrides_the_json_field(self):
        result = self._with_config(
            {"navigation_text": "from json"}, {"ALPAMAYO_NAV_TEXT": "from env"}
        )
        self.assertEqual(result["nav"], "from env")

    def test_malformed_freshness_falls_back_field_by_field(self):
        result = self._with_config({
            "camera_freshness": {
                "camera_stale_ms": "nonsense",
                "camera_pose_delta_ms": -5,
                "window_max_age_ms": 10,       # below the 200 ms floor
                "frame_timeout_ms": True,      # bool is not a number
            }
        })
        self.assertAlmostEqual(result["freshness"].camera_stale_ms, 1000.0)
        self.assertAlmostEqual(result["freshness"].camera_pose_delta_ms, 1000.0)
        self.assertEqual(result["freshness"].window_max_age_ms, 200)
        self.assertEqual(result["freshness"].frame_timeout_ms, 3000)

    def test_non_string_navigation_text_is_ignored(self):
        result = self._with_config({"navigation_text": ["not", "a", "string"]})
        self.assertEqual(result["nav"], "")

    def test_a_named_config_file_that_is_missing_is_not_defaulted(self):
        """Setting ``ALPAMAYO_CONFIG`` is a request; an unmeetable one aborts.

        It used to start on the built-in defaults instead, which is the one
        outcome the operator cannot see: the document they pinned the run to is
        not the one that ran.
        """
        with patch.dict(os.environ,
                        {"ALPAMAYO_CONFIG": str(Path(tempfile.gettempdir()) / "nope.json")},
                        clear=False):
            with self.assertRaises(ValueError) as caught:
                AppConfig.from_env()
        self.assertIn("ALPAMAYO_CONFIG", str(caught.exception))

    def test_no_document_at_all_still_uses_the_defaults(self):
        """The legacy tree: nothing named, no shipped file, built-in values."""
        with patch("config.loader.resolve_config_path", return_value=None):
            cfg = AppConfig.from_env()
        self.assertAlmostEqual(cfg.camera_stale_ms, 1000.0)

    def test_freshness_config_object_is_frozen(self):
        cfg = CameraFreshnessConfig()
        with self.assertRaises(Exception):
            cfg.camera_stale_ms = 1.0  # type: ignore[misc]


class SingleLoadTests(_CleanEnv):
    def test_load_app_config_returns_an_app_config(self):
        self.assertIsInstance(load_app_config(), AppConfig)

    def test_config_is_not_frozen_but_is_loaded_once_per_call(self):
        # Mutable: the offline model test overrides checkpoint/navigation_text.
        cfg = load_app_config()
        cfg.checkpoint = "/tmp/other"
        self.assertEqual(cfg.checkpoint, "/tmp/other")
        # A fresh load is unaffected by that mutation.
        self.assertNotEqual(load_app_config().checkpoint, "/tmp/other")


if __name__ == "__main__":
    unittest.main()
