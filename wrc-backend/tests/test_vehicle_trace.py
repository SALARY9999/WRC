"""Pose-in and waypoint-out traces written on PGX-1.

The files are append-only.  A write that fails must not raise into the pose
listener or the waypoint send.  ``pipeline.py`` is checked as source here,
the same way the trim tests do, so this file does not import torch.
"""

from __future__ import annotations

import ast
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vehicle.receiver import UdpPoseListener
from vehicle.trace import (
    VEHICLE_TRACE_NAME,
    JsonlTrace,
    open_jsonl_trace,
    waypoint_record,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]


class _History:
    """Enough of PoseHistory for ingest.  The real buffer imports torch."""

    def __init__(self) -> None:
        self.poses: list = []

    def add(self, pose) -> None:
        self.poses.append(pose)

    def snapshot(self) -> list:
        return list(self.poses)


def _packet(**pose_overrides) -> bytes:
    pose = {
        "latitude": 28.0,
        "longitude": 113.0,
        "height": 12.5,
        "azimuth": 90.0,
        "vNorth": 1.0,
        "vEast": 0.0,
        "chassis": {"cur_spd": 2.5, "yaw_rate": 10.0},
    }
    pose.update(pose_overrides)
    body = {
        "type": "vehicle_state",
        "timestamp": 1_700_000_000.25,
        "pose": pose,
    }
    return json.dumps(body).encode()


def _method(name: str) -> ast.FunctionDef:
    tree = ast.parse((BACKEND_ROOT / "pipeline.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "PipelineRunner":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == name:
                return item
    raise AssertionError(f"PipelineRunner.{name} is gone")


def _call_names(fn: ast.FunctionDef) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            found.append((node.lineno, ast.unparse(func)))
        elif isinstance(func, ast.Name):
            found.append((node.lineno, func.id))
    return found


class JsonlTraceTests(unittest.TestCase):
    def test_appends_and_never_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / VEHICLE_TRACE_NAME
            trace = JsonlTrace(str(path))
            trace.append({"event": "pose", "latitude": 1.0})
            trace.append({"event": "pose", "latitude": 2.0})
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            [json.loads(line)["latitude"] for line in lines], [1.0, 2.0]
        )

    def test_empty_directory_disables_recording(self):
        self.assertIsNone(open_jsonl_trace("", VEHICLE_TRACE_NAME))
        self.assertIsNone(open_jsonl_trace("   ", VEHICLE_TRACE_NAME))

    def test_write_failure_is_logged_and_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = JsonlTrace(str(Path(tmp) / VEHICLE_TRACE_NAME))
            with patch("vehicle.trace.open", side_effect=OSError("disk full")):
                with self.assertLogs("vehicle.trace", level="WARNING") as logs:
                    trace.append({"event": "pose"})
        self.assertIn("disk full", logs.output[0])


class PoseTraceTests(unittest.TestCase):
    def test_each_decoded_pose_is_appended_with_chassis(self):
        with tempfile.TemporaryDirectory() as tmp:
            listener = UdpPoseListener(
                _History(),
                trace=JsonlTrace(str(Path(tmp) / VEHICLE_TRACE_NAME)),
            )
            listener.ingest(_packet())
            listener.ingest(_packet(latitude=28.1))
            lines = [
                json.loads(line)
                for line in (Path(tmp) / VEHICLE_TRACE_NAME).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
        self.assertEqual(len(listener.history.snapshot()), 2)
        self.assertEqual([row["latitude"] for row in lines], [28.0, 28.1])
        first = lines[0]
        self.assertEqual(first["event"], "pose")
        self.assertAlmostEqual(first["longitude"], 113.0)
        self.assertAlmostEqual(first["altitude"], 12.5)
        self.assertAlmostEqual(first["azimuth_deg"], 90.0)
        self.assertAlmostEqual(first["heading_deg"], 0.0)
        self.assertAlmostEqual(first["v_north"], 1.0)
        self.assertAlmostEqual(first["v_east"], 0.0)
        self.assertAlmostEqual(first["chassis_speed_mps"], 2.5)
        self.assertAlmostEqual(
            first["chassis_yaw_rate_rad_s"], math.radians(10.0)
        )
        self.assertEqual(first["payload_timestamp_us"], 1_700_000_000_250_000)
        self.assertIsNotNone(first["arrival_us"])

    def test_missing_chassis_is_recorded_empty_and_pose_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            listener = UdpPoseListener(
                _History(),
                trace=JsonlTrace(str(Path(tmp) / VEHICLE_TRACE_NAME)),
            )
            body = json.loads(_packet())
            body["pose"].pop("chassis")
            listener.ingest(json.dumps(body).encode())
            row = json.loads(
                (Path(tmp) / VEHICLE_TRACE_NAME).read_text(encoding="utf-8")
            )
        self.assertEqual(len(listener.history.snapshot()), 1)
        self.assertIsNone(row["chassis_speed_mps"])
        self.assertIsNone(row["chassis_yaw_rate_rad_s"])

    def test_trace_failure_does_not_drop_the_pose(self):
        class _Boom:
            def append(self, record):
                raise OSError("full")

        listener = UdpPoseListener(_History(), trace=_Boom())  # type: ignore[arg-type]
        with self.assertLogs("vehicle.receiver", level="WARNING"):
            listener.ingest(_packet())
        self.assertEqual(len(listener.history.snapshot()), 1)


class WaypointTraceTests(unittest.TestCase):
    def test_record_keeps_every_sent_point_and_the_source(self):
        # Stand-in addresses. The live path copies whatever the process was
        # configured with; these values are not a deployment default.
        host = "10.1.2.3"
        port = 18010
        endpoint = "http://10.4.5.6:18092"
        record = waypoint_record(
            send_timestamp=1_700_000_000.5,
            waypoint_seq=7,
            control_flag=1,
            host=host,
            port=port,
            latitude=[28.0, 28.1],
            longitude=[113.0, 113.1],
            count=2,
            inference_latency_ms=812.25,
            cycle_id=4,
            worker_id=2,
            device=-1,
            endpoint=endpoint,
        )
        self.assertEqual(record["event"], "waypoints")
        self.assertEqual(record["waypoint_seq"], 7)
        self.assertEqual(record["control_flag"], 1)
        self.assertEqual(record["host"], host)
        self.assertEqual(record["port"], port)
        self.assertEqual(record["cycle_id"], 4)
        self.assertEqual(record["worker_id"], 2)
        self.assertEqual(record["device"], -1)
        self.assertEqual(record["endpoint"], endpoint)
        self.assertEqual(record["waypoints"], [[28.0, 113.0], [28.1, 113.1]])
        self.assertNotIn("waypoints_before_trim", record)
        self.assertAlmostEqual(record["inference_latency_ms"], 812.25)

    def test_record_keeps_the_uncropped_trajectory_beside_the_sent_points(self):
        record = waypoint_record(
            send_timestamp=1_700_000_000.5,
            waypoint_seq=7,
            control_flag=0,
            host="10.1.2.3",
            port=18010,
            latitude=[28.2],
            longitude=[113.2],
            count=1,
            latitude_before_trim=[28.0, 28.1, 28.2],
            longitude_before_trim=[113.0, 113.1, 113.2],
            inference_latency_ms=812.25,
            cycle_id=4,
            worker_id=2,
            device=-1,
            endpoint="http://10.4.5.6:18092",
        )
        self.assertEqual(record["waypoints"], [[28.2, 113.2]])
        self.assertEqual(
            record["waypoints_before_trim"],
            [[28.0, 113.0], [28.1, 113.1], [28.2, 113.2]],
        )

    def test_pipeline_records_after_send_and_names_the_worker(self):
        handler = _method("_handle_result")
        calls = _call_names(handler)
        send_at = next(line for line, name in calls if name == "self.sender.send")
        record_at = next(
            line for line, name in calls if name == "self._record_sent_waypoints"
        )
        self.assertLess(send_at, record_at)
        recorder = _method("_record_sent_waypoints")
        recorder_calls = [name for _, name in _call_names(recorder)]
        self.assertIn("trace.append", recorder_calls)
        record_call = next(
            node
            for node in ast.walk(recorder)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "waypoint_record")
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "waypoint_record"
                )
            )
        )
        keywords = {kw.arg for kw in record_call.keywords}
        self.assertIn("latitude_before_trim", keywords)
        self.assertIn("longitude_before_trim", keywords)
        handler_record = next(
            node
            for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_record_sent_waypoints"
        )
        handler_keywords = {kw.arg for kw in handler_record.keywords}
        self.assertIn("latitude_before_trim", handler_keywords)
        self.assertIn("longitude_before_trim", handler_keywords)
        infer = next(
            node.value
            for node in ast.walk(handler)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "infer_record"
                for target in node.targets
            )
        )
        self.assertIsInstance(infer, ast.Dict)
        keys = {
            key.value
            for key in infer.keys
            if isinstance(key, ast.Constant)
        }
        self.assertIn("worker_id", keys)
        self.assertIn("endpoint", keys)

    def test_pose_and_waypoints_append_to_the_same_file_in_order(self):
        host = "10.1.2.3"
        port = 18010
        endpoint = "http://10.4.5.6:18092"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / VEHICLE_TRACE_NAME
            trace = JsonlTrace(str(path))
            listener = UdpPoseListener(_History(), trace=trace)
            listener.ingest(_packet())
            trace.append(
                waypoint_record(
                    send_timestamp=1_700_000_001.0,
                    waypoint_seq=1,
                    control_flag=1,
                    host=host,
                    port=port,
                    latitude=[28.0],
                    longitude=[113.0],
                    count=1,
                    inference_latency_ms=10.0,
                    cycle_id=1,
                    worker_id=1,
                    device=-1,
                    endpoint=endpoint,
                )
            )
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(list(path.parent.iterdir()), [path])
        self.assertEqual([row["event"] for row in rows], ["pose", "waypoints"])
        self.assertEqual(rows[1]["host"], host)
        self.assertEqual(rows[1]["port"], port)
        self.assertEqual(rows[1]["endpoint"], endpoint)

    def test_pipeline_opens_one_shared_trace(self):
        init = _method("__init__")
        opens = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open_jsonl_trace"
        ]
        self.assertEqual(len(opens), 1)
        self.assertEqual(ast.unparse(opens[0].args[1]), "VEHICLE_TRACE_NAME")
        start = _method("start")
        traces = [
            node
            for node in ast.walk(start)
            if isinstance(node, ast.keyword) and node.arg == "trace"
        ]
        self.assertEqual(
            [ast.unparse(node.value) for node in traces],
            ["self._vehicle_trace"],
        )


if __name__ == "__main__":
    unittest.main()
