"""The replay tool must work before it is pointed at real data.

``tools/replay_trajectory_postprocess.py`` re-runs ``trim_waypoints_to_latest_pose``
on a recorded package.  A check that silently passes is worse than no check, so
these tests build packages that are consistent by construction and then break
them one field at a time, asserting the tool notices.

The fixtures are written to a temporary directory and the tool is imported by
path (``tools/`` is not a package).  Nothing here reads or writes a package
outside that temporary directory.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _source_tree import pipeline_result_tree

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "replay_trajectory_postprocess",
        ROOT / "tools" / "replay_trajectory_postprocess.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()

REF_LAT, REF_LON, REF_ALT, REF_HEADING = 28.0, 113.0, 0.0, 0.0


def _reference_pose(la=None, lo=None, alt=None, heading=None, ts=1_000_000) -> dict:
    return {
        "timestamp_us": ts,
        "latitude": REF_LAT if la is None else la,
        "longitude": REF_LON if lo is None else lo,
        "altitude": REF_ALT if alt is None else alt,
        "heading_deg": REF_HEADING if heading is None else heading,
    }


def _pose_latest(x_offset_deg: float = 0.0, ts: int = 2_000_000) -> dict:
    return {
        "timestamp_us": ts,
        "payload_timestamp_us": ts,
        "arrival_us": ts,
        "latitude": REF_LAT + x_offset_deg,
        "longitude": REF_LON,
        "altitude": REF_ALT,
        "heading_deg": REF_HEADING,
    }


def _trajectory(count: int = 64) -> np.ndarray:
    """1 m apart along model +x, which is north for heading 0."""
    return np.column_stack(
        (np.arange(count, dtype=float), np.zeros(count), np.zeros(count))
    )


def _crop_section(ref: dict, pose: dict | None) -> dict:
    """The ``trajectory_postprocess_input`` block the backend now writes."""
    return {
        "trajectory_key": "model_waypoints_local_xyz",
        "latest_pose": pose,
        "reference_pose": ref,
        "inference_latency_ms": 41.5,
        "original_waypoint_count": len(_trajectory()),
    }


def _build_package(
    path: Path,
    *,
    sent: np.ndarray | None = None,
    pose_latest: dict | None = None,
    num_waypoints: int | None = None,
    with_crop_input: bool = True,
    crop_latest_pose: dict | None = None,
    crop_reference_pose: dict | None = None,
) -> dict:
    """A package shaped like ``_save_infer_package`` writes, consistent by default.

    ``with_crop_input`` off produces the pre-Stage-2 shape, which the tool must
    keep accepting.  On (the default, matching the current backend) the crop's
    own input is recorded; ``crop_latest_pose`` overrides just that block, which
    is how the tests show the block is what gets replayed.
    """
    ref = _reference_pose()
    raw = _trajectory()
    pose = _pose_latest() if pose_latest is None else pose_latest
    # The pose the crop consumed.  Identical to pose_latest unless a test says
    # otherwise: in a real package a vehicle moving between the crop and the
    # write makes them differ.
    crop_pose = pose if crop_latest_pose is None else crop_latest_pose

    if sent is None:
        # What the pipeline would actually have sent for this pose.
        from trajectory.coordinates import local_xy_to_latlon
        from trajectory.postprocess import trim_waypoints_to_latest_pose
        from vehicle.models import GlobalPose, PoseSample, vehicle_azimuth_for_compass

        latest = GlobalPose(
            timestamp_us=int(crop_pose["timestamp_us"]),
            latitude=crop_pose["latitude"], longitude=crop_pose["longitude"],
            altitude=crop_pose["altitude"], v_north=0.0, v_east=0.0, v_up=0.0,
            roll_deg=0.0, pitch_deg=0.0,
            azimuth_deg=vehicle_azimuth_for_compass(crop_pose["heading_deg"]),
        )
        pose_sample = PoseSample(
            int(ref["timestamp_us"]), ref["latitude"], ref["longitude"],
            ref["altitude"], ref["heading_deg"],
        )
        cropped = trim_waypoints_to_latest_pose(raw, latest, pose_sample).positions
        lat, lon = local_xy_to_latlon(
            cropped[:, 0], cropped[:, 1],
            pose_sample.latitude, pose_sample.longitude, pose_sample.heading_deg,
        )
        waypoints = [
            [round(float(a), 8), round(float(b), 8)] for a, b in zip(lat, lon)
        ]
        sent = cropped
    else:
        waypoints = [[0.0, 0.0]] * len(sent)

    record = {
        "reference_pose": ref,
        "pose_latest": pose,
        "output_waypoints_local_xyz": sent.tolist(),
        "output_waypoints": waypoints,
        "num_waypoints": len(sent) if num_waypoints is None else num_waypoints,
        "inference_latency_ms": 41.5,
    }
    record["model_waypoints_local_xyz"] = raw.tolist()
    if with_crop_input:
        record["trajectory_postprocess_input"] = _crop_section(
            ref if crop_reference_pose is None else crop_reference_pose,
            crop_pose,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return record


class _PackageDir(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _pkg(self, name: str = "20260917_120000_000000") -> Path:
        return self.root / name / "infer.json"


class ReplayAcceptsConsistentPackagesTests(_PackageDir):
    def test_a_consistent_package_passes_every_check(self):
        path = self._pkg()
        _build_package(path)
        failures, notes = tool.replay_package(path)
        self.assertEqual(failures, [])
        self.assertIn("send-checked", notes)

    def test_main_exits_zero_and_summarises(self):
        _build_package(self._pkg())
        self.assertEqual(tool.main([str(self.root), "--quiet"]), 0)

    def test_an_empty_directory_is_an_error_not_a_pass(self):
        # A gate that reports success on zero packages would be worthless.
        self.assertEqual(tool.main([str(self.root)]), 2)


class ReplayDetectsProblemsTests(_PackageDir):
    def _failures(self, path: Path):
        failures, _ = tool.replay_package(path)
        return failures

    def test_a_raw_trajectory_that_was_not_what_was_sent_is_caught(self):
        # A pre-Stage-2 package: the crop's input was pose_latest, so moving
        # the pose is what invalidates the recorded crop.
        path = self._pkg()
        record = _build_package(path, with_crop_input=False)
        record["pose_latest"]["latitude"] += 0.001  # ~111 m north
        path.write_text(json.dumps(record), encoding="utf-8")
        failures = self._failures(path)
        self.assertTrue(
            any("idempotent" in f for f in failures),
            f"a pose that invalidates the recorded crop was not detected: {failures}",
        )

    def test_a_num_waypoints_mismatch_is_caught(self):
        path = self._pkg()
        _build_package(path, num_waypoints=99)
        self.assertTrue(any("num_waypoints=99" in f for f in self._failures(path)))

    def test_a_corrupt_package_is_a_failure_not_a_crash(self):
        path = self._pkg()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        failures = self._failures(path)
        self.assertTrue(any("unreadable" in f for f in failures))

    def test_a_package_without_a_reference_pose_is_a_failure(self):
        path = self._pkg()
        record = _build_package(path)
        del record["reference_pose"]
        del record["trajectory_postprocess_input"]["reference_pose"]
        path.write_text(json.dumps(record), encoding="utf-8")
        self.assertTrue(any("reference_pose unusable" in f for f in self._failures(path)))

    def test_the_sections_reference_pose_rescues_a_missing_top_level_one(self):
        # Ordering, not leniency: the crop's own record is the better source,
        # so a package that lost the top-level copy is still replayable.
        path = self._pkg()
        record = _build_package(path)
        del record["reference_pose"]
        path.write_text(json.dumps(record), encoding="utf-8")
        failures, notes = tool.replay_package(path)
        self.assertEqual(failures, [])
        self.assertIn("crop-input", notes)

    def test_main_exits_non_zero_when_anything_fails(self):
        _build_package(self._pkg(), num_waypoints=99)
        self.assertEqual(tool.main([str(self.root), "--quiet"]), 1)


class ReplayUsesTheRecordedCropInputTests(_PackageDir):
    """The crop's own input is authoritative, and its absence is not fatal.

    ``pose_latest`` is written *after* the send, so a moving vehicle makes it
    newer than the pose the crop saw.  Replaying against it would replay a crop
    that never ran - which is why the recorded section wins, including when it
    says the crop had no pose at all.
    """

    def test_a_package_without_the_section_falls_back_and_says_so(self):
        path = self._pkg()
        _build_package(path, with_crop_input=False)
        failures, notes = tool.replay_package(path)
        self.assertEqual(failures, [])
        self.assertIn("legacy-pose_latest", notes)
        self.assertNotIn("crop-input", notes)

    def test_a_package_with_the_section_replays_against_it(self):
        path = self._pkg()
        _build_package(path)
        failures, notes = tool.replay_package(path)
        self.assertEqual(failures, [])
        self.assertIn("crop-input", notes)
        self.assertNotIn("legacy-pose_latest", notes)

    def test_moving_the_recorded_crop_pose_invalidates_the_replay(self):
        # The mirror of the legacy test: what breaks the replay now is moving
        # the pose inside the section, not pose_latest.
        path = self._pkg()
        record = _build_package(path)
        record["trajectory_postprocess_input"]["latest_pose"]["latitude"] += 0.001
        path.write_text(json.dumps(record), encoding="utf-8")
        failures, notes = tool.replay_package(path)
        self.assertIn("crop-input", notes)
        self.assertTrue(
            any("idempotent" in f for f in failures),
            f"the recorded crop pose was ignored: {failures}",
        )

    def test_a_newer_pose_latest_does_not_produce_a_phantom_failure(self):
        # Exactly the field condition: the vehicle moved between the crop and
        # the write, so pose_latest is newer than the pose the crop used.
        # Nothing is wrong, so nothing may fail.
        path = self._pkg()
        _build_package(path, crop_latest_pose=_pose_latest(),
                       pose_latest=_pose_latest(x_offset_deg=0.0005))
        failures, notes = tool.replay_package(path)
        self.assertEqual(failures, [])
        self.assertIn("crop-input", notes)
        self.assertIn("pose-moved", notes)

    def test_a_null_crop_pose_is_authoritative_not_a_fallback(self):
        # If the crop ran without a pose, replaying with pose_latest would
        # invent a crop.  The helper is asserted directly because that is the
        # rule; the checks above then exercise it end to end.
        record = {"pose_latest": _pose_latest()}
        record["trajectory_postprocess_input"] = _crop_section(_reference_pose(), None)
        notes: list[str] = []
        latest, ref_payload, latency = tool._crop_inputs(record, notes)
        self.assertIsNone(latest)
        self.assertEqual(ref_payload, _reference_pose())
        self.assertEqual(latency, 41.5)
        self.assertIn("crop-input", notes)

    def test_the_reference_pose_comes_from_the_section_when_present(self):
        # The two are written from the same sample today, so they can only be
        # told apart by making them differ.
        path = self._pkg()
        moved = _reference_pose(la=REF_LAT + 0.01)
        record = _build_package(path, sent=_trajectory()[:8],
                                crop_reference_pose=moved)
        path.write_text(json.dumps(record), encoding="utf-8")
        notes: list[str] = []
        _, ref_payload, _ = tool._crop_inputs(record, notes)
        self.assertEqual(ref_payload["latitude"], REF_LAT + 0.01)

    def test_the_summary_counts_which_input_was_used(self):
        _build_package(self._pkg("20260917_120000_000000"))
        _build_package(self._pkg("20260917_120001_000000"),
                       with_crop_input=False)
        out = self.root / "summary.json"
        self.assertEqual(tool.main([str(self.root), "--quiet", "--json", str(out)]), 0)
        summary = json.loads(out.read_text(encoding="utf-8"))["summary"]
        self.assertEqual(summary["crop_input"], 1)
        self.assertEqual(summary["legacy_pose_latest"], 1)
        self.assertEqual(summary["send_checked"], 2)


class PackageRecordsTheRawTrajectoryTests(unittest.TestCase):
    """The replay is only a replay if the package holds the crop's real input.

    ``model_waypoints_local_xyz`` must be the model output *before* the crop,
    at full precision.  Pointing it at ``trim.positions`` or rounding it would
    produce a package that still parses and still replays - and silently
    replays the wrong thing.  Asserted structurally because the code lives
    inside ``PipelineRunner._handle_result`` (``pipeline_result_tree``), which
    is where the post-model sequence sits for both the inline and pooled paths.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = pipeline_result_tree()

    def _assignment(self, target: str):
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == target
            ):
                return node
        self.fail(f"{target} is no longer assigned in PipelineRunner._handle_result")

    def test_the_raw_field_is_built_from_the_model_output(self):
        node = self._assignment("model_waypoint_local")
        iterated = sorted(
            ast.unparse(gen.iter)
            for gen in ast.walk(node)
            if isinstance(gen, ast.comprehension)
        )
        self.assertEqual(len(iterated), 1, f"unexpected comprehension shape: {iterated}")
        self.assertIn(
            "result.positions", iterated[0],
            "the raw trajectory must come from result.positions (model output); "
            f"found {iterated[0]!r}",
        )
        self.assertNotIn(
            "trim.positions", iterated[0],
            "the raw field must NOT be the cropped array",
        )

    def test_the_raw_field_is_not_rounded(self):
        source = ast.unparse(self._assignment("model_waypoint_local"))
        self.assertNotIn(
            "round(", source,
            "rounding the raw trajectory changes what the crop sees, so the "
            "replay could crop differently from the live run",
        )

    def test_the_record_carries_both_arrays(self):
        keys = {
            node.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for key in ("model_waypoints_local_xyz", "output_waypoints_local_xyz"):
            with self.subTest(key=key):
                self.assertIn(key, keys)

    def test_a_full_precision_float64_array_survives_the_json_round_trip(self):
        # The whole replay rests on this: if JSON lost precision, the replayed
        # crop could remove a different number of points and every package
        # would report a phantom mismatch.
        rng = np.random.default_rng(7)
        original = np.column_stack(
            (rng.normal(0, 50, 64), rng.normal(0, 50, 64), np.zeros(64))
        )
        restored = np.asarray(
            json.loads(json.dumps(original.tolist())), dtype=np.float64
        )
        self.assertTrue(np.array_equal(original, restored, equal_nan=True))


class ReplayDiscoversRealLayoutTests(_PackageDir):
    def test_only_infer_json_under_a_subdirectory_is_picked_up(self):
        _build_package(self._pkg("20260917_120000_000000"))
        _build_package(self._pkg("20260917_120001_000000"))
        # A stray JSON beside the packages must not be treated as one.
        (self.root / "notes.json").write_text("{}", encoding="utf-8")
        found = tool.collect_packages(self.root)
        self.assertEqual(len(found), 2)
        self.assertTrue(all(p.name == "infer.json" for p in found))


if __name__ == "__main__":
    unittest.main()
