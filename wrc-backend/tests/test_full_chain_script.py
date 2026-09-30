"""tools/run_full_chain_test.sh must not be able to stop production.

The checks are textual.  Running the script would start a backend.
"""

from __future__ import annotations

import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "run_full_chain_test.sh"


class FullChainScriptGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = SCRIPT.read_text(encoding="utf-8")

    def test_it_does_not_kill_the_production_backend_or_bind_its_ports(self) -> None:
        self.assertNotIn("pkill -f '[u]vicorn", self.text)
        self.assertNotIn("pkill -f '[p]ort_sink.py 7003'", self.text)
        self.assertNotIn("--port 8090", self.text)
        self.assertNotIn("port_sink.py 7003", self.text)
        line = self.text.split("PRODUCTION_PORTS=", 1)[1].split("\n", 1)[0]
        for port in ("8090", "7001", "30106", "30006"):
            self.assertIn(port, line)
        for retired in ("17100", "7003"):
            self.assertNotIn(retired, line)

    def test_it_uses_a_dynamic_http_port_and_waypoint_port(self) -> None:
        self.assertIn('HTTP_PORT="$(free_port STREAM)"', self.text)
        self.assertIn('WAYPOINT_PORT="$(free_port DGRAM)"', self.text)
        self.assertIn('ALPAMAYO_WAYPOINT_PORT="$WAYPOINT_PORT"', self.text)
        self.assertIn('--port "$HTTP_PORT"', self.text)
        self.assertIn("ALPAMAYO_ROUTE_TCP_ENABLED=false", self.text)
        self.assertIn('CIMG_PORT="$(free_port STREAM)"', self.text)
        self.assertIn('ALPAMAYO_CIMG_PORT="$CIMG_PORT"', self.text)
        self.assertIn("ALPAMAYO_CIMG_CAMERA_INDICES=1,2,0,4", self.text)

    def test_it_does_not_start_mpegts_or_select_a_camera_mode(self) -> None:
        for banned in (
            "ffmpeg",
            "mpegts",
            "ALPAMAYO_CAMERA_INPUT",
            "ALPAMAYO_CAMERA_SOURCES",
            "ALPAMAYO_CAMERA_INDICES",
            "camera_sources",
        ):
            self.assertNotIn(banned, self.text)

    def test_it_does_not_export_unread_freshness_or_control_variables(self) -> None:
        for name in (
            "ALPAMAYO_WINDOW_MAX_AGE_MS",
            "ALPAMAYO_CAMERA_STALE_MS",
            "ALPAMAYO_CAMERA_FRAME_TIMEOUT_MS",
            "ALPAMAYO_CONTROL_MODE",
        ):
            self.assertNotIn(name, self.text)
