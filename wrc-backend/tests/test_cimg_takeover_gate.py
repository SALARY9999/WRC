"""The production CIMG deployment must clear the camera-status gate.

CIMG publishes one stats row per camera.  A count taken from anywhere else
used to be 0, so every cycle stopped at ``camera_status_unavailable`` and
``control`` stayed 0.

This test builds the runner the launcher builds: indices ``1,2,0,4``.  It does
not call ``start()``, so it does not listen on 30106 or 30006 and does not
send a waypoint.
"""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from _cv2_isolation import cv2_stub

with cv2_stub():
    import pipeline

from config.loader import AppConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = ROOT / "config" / "config.json"


class ProductionCimgTakeoverGateTests(unittest.TestCase):
    def test_four_healthy_cimg_rows_pass_the_camera_status_gate(self) -> None:
        env = {
            "ALPAMAYO_CONFIG": str(SHIPPED_CONFIG),
            "ALPAMAYO_MOCK_MODE": "0",
            "ALPAMAYO_POSE_MODE": "udp",
            "ALPAMAYO_CIMG_CAMERA_INDICES": "1,2,0,4",
            "ALPAMAYO_WAYPOINT_MAGIC": "0x514D3031",
            "ALPAMAYO_VEHICLE_ID": "1",
            "ALPAMAYO_CONTROL_FLAG": "0",
            "ALPAMAYO_INFER_PACKAGE_DIR": "",
            "ALPAMAYO_SNAPSHOT_DIR": "",
        }
        with patch.dict(os.environ, env, clear=False):
            for retired in (
                "ALPAMAYO_CAMERA_INPUT",
                "ALPAMAYO_CAMERA_SOURCES",
                "ALPAMAYO_CAMERA_INDICES",
                "ALPAMAYO_FFMPEG",
                "ALPAMAYO_MPEGTS_LOG_DIR",
            ):
                os.environ.pop(retired, None)
            settings = AppConfig.from_env()
            runner = pipeline.PipelineRunner(settings)
        self.addCleanup(runner.sender.close)

        self.assertEqual(settings.cimg_camera_indices, [1, 2, 0, 4])
        self.assertEqual((settings.cimg_host, settings.cimg_port), ("0.0.0.0", 30106))
        self.assertEqual(runner.decision_layer.expected_camera_count, 4)
        self.assertEqual(
            runner.cameras.reported_camera_count(),
            runner.decision_layer.expected_camera_count,
        )

        source = runner.cameras.readers[0]
        now_us = time.monotonic_ns() // 1000
        for state in source._cameras.values():
            state.last_arrival_us = now_us
        rows = runner.cameras.stats()["readers"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["health"] == "HEALTHY" for row in rows))
        self.assertTrue(
            all(
                row["frame_age_ms"] is not None
                and float(row["frame_age_ms"]) <= settings.camera_stale_ms
                for row in rows
            )
        )

        engine = runner.takeover_engine
        engine.takeover_count = 1
        engine.release_count = 1
        decision = runner.decision_layer.decide(
            positions=np.zeros((8, 2)),
            t0_us=0,
            ref_pose=None,
            camera_stats_fn=lambda: runner.cameras.stats()["readers"],
            latest_pose_fn=lambda: None,
        )
        self.assertNotEqual(decision.decision_reason, "camera_status_unavailable")
        self.assertEqual(decision.control_flag, 0)
        self.assertEqual(decision.takeover_count, 1)
        self.assertEqual(decision.release_count, 1)
        self.assertEqual(engine.takeover_count, 1)
        self.assertEqual(engine.release_count, 1)
