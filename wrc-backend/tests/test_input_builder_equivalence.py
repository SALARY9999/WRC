"""``InputBuilder`` must reproduce the inline input block, field by field.

Stage 2 gate, same shape as ``test_postprocess_equivalence``.  The block that
used to sit inside ``PipelineRunner.run`` - the freshness gates, the camera-lag
gate, the pose-history alignment, ``build_ego_arrays`` and the model-input
diagnostics - is transcribed below as ``_reference_build`` and compared against
the real builder on a fixed corpus.

The reference is a transcription, not a copy: it keeps the original inline
shape (a dict of metrics, an if/elif chain for freshness, literal reason
strings) so that a transcription error in the builder shows up as a mismatch
instead of being shared by both sides.  Do not "tidy" it to match the new code
- that is exactly what it exists to detect.

Every reject reason is a string that reaches ``TakeoverDecisionEngine.invalid``
and the operator log, so the spellings are pinned here as literals.
"""

from __future__ import annotations

import types
import unittest
from pathlib import Path

import numpy as np
import torch

from camera.models import CameraWindow
from inference.input_builder import (
    REASON_CAMERA_POSE_DELTA,
    REASON_CAMERA_STALE,
    REASON_CAMERA_WINDOW_STALE,
    REASON_POSE_HISTORY_UNAVAILABLE,
    REASON_POSE_STALE,
    InputBuilder,
)
from vehicle.history import PoseHistory, build_ego_arrays
from vehicle.models import GlobalPose, vehicle_azimuth_for_compass

ROOT = Path(__file__).resolve().parents[1]

# Calibrations used by both sides.  Deliberately not the production defaults so
# that a hard-coded number in the builder would not go unnoticed.
CAMERA_STALE_MS = 1000.0
POSE_STALE_MS = 800.0
CAMERA_POSE_DELTA_MS = 1000.0
HISTORY_STEPS = 16
HISTORY_DT_S = 0.1

CAMERA_INDICES = [0, 1, 2, 4]
LAT0, LON0, ALT0, HEADING0 = 28.0, 113.0, 0.0, 0.0


def _window(
    *,
    t0_us: int,
    frame_arrivals_us=None,
    frame_ids=None,
) -> CameraWindow:
    return CameraWindow(
        t0_us=t0_us,
        # The builder never reads `frames`; the model does.  A small tensor is
        # enough to make the window a real CameraWindow.
        frames=torch.zeros((len(CAMERA_INDICES), 4, 3, 2, 2), dtype=torch.uint8),
        camera_indices=list(CAMERA_INDICES),
        timestamps_us=[t0_us - 3, t0_us - 2, t0_us - 1, t0_us],
        frame_ids=frame_ids,
        frame_arrivals_us=frame_arrivals_us,
    )


def _arrivals(t0_us: int, *, age_us: int = 0, missing_camera: bool = False):
    """Per-camera newest-frame arrivals, `age_us` behind a fixed decode time."""
    base = t0_us + 5_000_000 - age_us  # the decode clock, unrelated to t0
    rows = [[base - 3, base - 2, base - 1, base] for _ in CAMERA_INDICES]
    if missing_camera:
        rows[-1] = [base - 3, base - 2, base - 1, None]
    return rows


def _frame_ids(prefix: str = "cam") -> list[list[str]]:
    return [
        [f"{prefix}{index}:{1000 + index * 10 + t}" for t in range(4)]
        for index in CAMERA_INDICES
    ]


def _history(
    *,
    t0_us: int,
    count: int = 20,
    arrival_us: int | None = None,
    start_offset_us: int | None = None,
    heading: float = 0.0,
    latitude: float = LAT0,
) -> PoseHistory:
    """A history whose newest sample sits at `t0_us + start_offset_us`."""
    history = PoseHistory()
    newest = t0_us + (start_offset_us or 0)
    for i in range(count):
        stamp = newest - (count - 1 - i) * 100_000
        history.add(
            GlobalPose(
                timestamp_us=stamp,
                latitude=latitude + i * 1e-6,
                longitude=LON0 + i * 1e-6,
                altitude=ALT0,
                v_north=2.0,
                v_east=0.0,
                v_up=0.0,
                roll_deg=0.0,
                pitch_deg=0.0,
                azimuth_deg=vehicle_azimuth_for_compass(heading),
                payload_timestamp_us=stamp + 7,
                arrival_us=stamp if arrival_us is None else arrival_us,
            )
        )
    return history


# --------------------------------------------------------------------------- #
# The reference: the pre-Stage-2 inline block, transcribed verbatim.
# --------------------------------------------------------------------------- #

def _reference_build(window, history, now_us):
    """Reference transcription of the input block from app.py (pre-Stage-2).

    Returns a dict: ``reason`` is None on success, and the metrics are always
    present because the inline code always computed them.
    """
    latest = history.latest()
    camera_arrivals = [
        int(row[-1])
        for row in (window.frame_arrivals_us or [])
        if row and row[-1] is not None
    ]
    camera_arrival_us = (
        min(camera_arrivals)
        if len(camera_arrivals) == len(window.camera_indices)
        else None
    )
    camera_age_ms = (
        (now_us - camera_arrival_us) / 1000.0
        if camera_arrival_us is not None
        else float("inf")
    )
    pose_arrival_us = (
        int(latest.arrival_us)
        if latest is not None and latest.arrival_us is not None
        else None
    )
    pose_age_ms = (
        (now_us - pose_arrival_us) / 1000.0
        if pose_arrival_us is not None
        else float("inf")
    )
    camera_pose_delta_ms = (
        abs(camera_arrival_us - pose_arrival_us) / 1000.0
        if camera_arrival_us is not None and pose_arrival_us is not None
        else float("inf")
    )
    latest_frame_ids = [
        row[-1] if row else None for row in (window.frame_ids or [])
    ]

    metrics = {
        "latest_pose": latest,
        "camera_arrival_us": camera_arrival_us,
        "pose_arrival_us": pose_arrival_us,
        "camera_age_ms": camera_age_ms,
        "pose_age_ms": pose_age_ms,
        "camera_pose_delta_ms": camera_pose_delta_ms,
        "latest_frame_ids": latest_frame_ids,
    }

    freshness_error = None
    if camera_age_ms > CAMERA_STALE_MS:
        freshness_error = "camera stale"
    elif pose_age_ms > POSE_STALE_MS:
        freshness_error = "pose stale"
    elif camera_pose_delta_ms > CAMERA_POSE_DELTA_MS:
        freshness_error = "camera-pose delta"
    if freshness_error is not None:
        return dict(metrics, reason=freshness_error)

    camera_lag_ms = None
    if latest is not None:
        camera_lag_ms = (latest.timestamp_us - window.t0_us) / 1000.0
        if camera_lag_ms > CAMERA_STALE_MS:
            return dict(metrics, reason="camera_window_stale",
                        camera_lag_ms=camera_lag_ms)

    poses = history.aligned_samples(window.t0_us, HISTORY_STEPS, HISTORY_DT_S)
    if poses is None:
        return dict(metrics, reason="model_pose_history_unavailable",
                    camera_lag_ms=camera_lag_ms)

    ego_xyz, ego_rot, ref = build_ego_arrays(poses)
    ref_ll = (ref.latitude, ref.longitude, ref.altitude)
    ego_history_rows = []
    for sample in poses:
        import math

        from trajectory.coordinates import enu_from_ll, enu_to_ego

        east_m, north_m, up_m = enu_from_ll(
            sample.latitude, sample.longitude, sample.altitude, ref_ll
        )
        x_m, y_m, _z_m = enu_to_ego(
            east_m, north_m, up_m, math.radians(ref.heading_deg)
        )
        ego_history_rows.append(
            {
                "timestamp_us": int(sample.timestamp_us),
                "latitude": round(float(sample.latitude), 8),
                "longitude": round(float(sample.longitude), 8),
                "altitude": round(float(sample.altitude), 3),
                "heading_deg": round(float(sample.heading_deg), 3),
                "x_m": round(float(x_m), 4),
                "y_m": round(float(y_m), 4),
            }
        )

    from inference.camera_labels import _CAMERA_DISPLAY_NAMES, _camera_package_label

    frame_id_rows = []
    for cam_pos, cam_idx in enumerate(window.camera_indices):
        ids = (
            list(window.frame_ids[cam_pos])
            if getattr(window, "frame_ids", None)
            and cam_pos < len(window.frame_ids)
            else []
        )
        frame_id_rows.append(
            {
                "camera_index": int(cam_idx),
                "camera_label": _camera_package_label(int(cam_idx)),
                "display": _CAMERA_DISPLAY_NAMES.get(int(cam_idx), str(cam_idx)),
                "frame_ids": ids,
                "latest_frame_id": ids[-1] if ids else None,
            }
        )
    latest_frame_ids_by_label = {
        row["camera_label"]: row["latest_frame_id"] for row in frame_id_rows
    }
    return dict(
        metrics,
        reason=None,
        camera_lag_ms=camera_lag_ms,
        ego_xyz=ego_xyz,
        ego_rot=ego_rot,
        ref=ref,
        ego_history=ego_history_rows,
        camera_frame_ids=frame_id_rows,
        camera_frame_id=latest_frame_ids_by_label,
        reference_payload_timestamp_us=history.nearest_payload_timestamp_us(
            int(ref.timestamp_us)
        ),
    )


def _builder() -> InputBuilder:
    return InputBuilder(
        history_steps=HISTORY_STEPS,
        history_dt_s=HISTORY_DT_S,
        camera_stale_ms=CAMERA_STALE_MS,
        pose_stale_ms=POSE_STALE_MS,
        camera_pose_delta_ms=CAMERA_POSE_DELTA_MS,
        now_us_fn=lambda: NOW_US * 1000,  # the builder wants nanoseconds
    )


#: Fixed clock for the whole corpus.  Every case is measured against this, so
#: the ages are exact integers and the two sides must agree bit for bit.
NOW_US = 5_000_000_000

T0_US = NOW_US - 5_000_000


def _cases() -> list[tuple[str, object, object]]:
    """(name, window, history) - one entry per reachable branch."""
    cases: list[tuple[str, object, object]] = [
        (
            "fresh",
            _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US),
                    frame_ids=_frame_ids()),
            _history(t0_us=T0_US, arrival_us=NOW_US - 1000),
        ),
        (
            "fresh_no_frame_ids",
            _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US)),
            _history(t0_us=T0_US, arrival_us=NOW_US - 1000),
        ),
        (
            "fresh_heading_wrap",
            _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US),
                    frame_ids=_frame_ids()),
            _history(t0_us=T0_US, arrival_us=NOW_US - 1000, heading=359.95),
        ),
        (
            "fresh_pose_is_older_than_history",
            _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US),
                    frame_ids=_frame_ids()),
            # Newest sample is 0.9 s behind t0 - still inside the camera-lag
            # budget but outside the pose-history coverage, so the history
            # gate is what rejects this one.
            _history(t0_us=T0_US, start_offset_us=-900_000,
                     arrival_us=NOW_US - 1000),
        ),
        (
            "camera_stale",
            _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US, age_us=2_000_000),
                    frame_ids=_frame_ids()),
            _history(t0_us=T0_US, arrival_us=NOW_US - 1000),
        ),
        (
            "camera_arrivals_incomplete",
            _window(t0_us=T0_US,
                    frame_arrivals_us=_arrivals(T0_US, missing_camera=True),
                    frame_ids=_frame_ids()),
            _history(t0_us=T0_US, arrival_us=NOW_US - 1000),
        ),
        (
            "camera_arrivals_absent",
            _window(t0_us=T0_US, frame_arrivals_us=None, frame_ids=_frame_ids()),
            _history(t0_us=T0_US, arrival_us=NOW_US - 1000),
        ),
        (
            "pose_stale",
            _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US),
                    frame_ids=_frame_ids()),
            _history(t0_us=T0_US, arrival_us=NOW_US - 5_000_000),
        ),
        (
            "camera_window_stale",
            _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US),
                    frame_ids=_frame_ids()),
            # Arrivals are fresh, so every freshness gate passes; it is the
            # pose *sample* being 2 s ahead of the window's t0 that rejects.
            _history(t0_us=T0_US, start_offset_us=2_000_000,
                     arrival_us=NOW_US - 1000),
        ),
        (
            "camera_pose_delta_clock_skew",
            # Camera age exactly 1000 (passes, the gate is strict) so the
            # 1.5 s gap between the two arrivals is what rejects.
            _window(t0_us=T0_US,
                    frame_arrivals_us=_arrivals(T0_US, age_us=1_000_000),
                    frame_ids=_frame_ids()),
            # A pose arrival dated *ahead* of the clock the loop just read.
            # See test_the_delta_gate_needs_a_clock_anomaly: at the shipped
            # calibrations this is the only way the delta gate can fire, and
            # the branch still needs corpus coverage because a transcription
            # error there would otherwise be invisible.
            _history(t0_us=T0_US, arrival_us=NOW_US + 500_000),
        ),
    ]
    return cases


class ReferenceAndBuilderAgreeTests(unittest.TestCase):
    """The headline gate: identical reason and identical fields, every case."""

    def setUp(self) -> None:
        self.builder = _builder()

    def _compare(self, name: str, window, history) -> None:
        reference = _reference_build(window, history, NOW_US)
        built = self.builder.build(window, history)
        with self.subTest(case=name, field="rejection_reason"):
            self.assertEqual(built.rejection_reason, reference["reason"])
        for field in (
            "camera_age_ms",
            "pose_age_ms",
            "camera_pose_delta_ms",
            "camera_arrival_us",
            "pose_arrival_us",
            "latest_frame_ids",
        ):
            with self.subTest(case=name, field=field):
                self.assertEqual(getattr(built, field), reference[field])
        with self.subTest(case=name, field="latest_pose"):
            self.assertIs(built.latest_pose, reference["latest_pose"])
        if reference["reason"] is not None:
            return
        for field in ("camera_lag_ms",):
            with self.subTest(case=name, field=field):
                self.assertEqual(getattr(built, field), reference[field])
        with self.subTest(case=name, field="ego_xyz"):
            self.assertTrue(np.array_equal(built.ego_xyz, reference["ego_xyz"]))
        with self.subTest(case=name, field="ego_rot"):
            self.assertTrue(np.array_equal(built.ego_rot, reference["ego_rot"]))
        with self.subTest(case=name, field="reference_pose"):
            self.assertEqual(built.reference_pose, reference["ref"])
        with self.subTest(case=name, field="reference_payload_timestamp_us"):
            self.assertEqual(
                built.reference_payload_timestamp_us,
                reference["reference_payload_timestamp_us"],
            )
        with self.subTest(case=name, field="ego_history"):
            self.assertEqual(built.ego_history, reference["ego_history"])
        with self.subTest(case=name, field="camera_frame_ids"):
            self.assertEqual(built.camera_frame_ids, reference["camera_frame_ids"])
        with self.subTest(case=name, field="camera_frame_id"):
            self.assertEqual(built.camera_frame_id, reference["camera_frame_id"])

    def test_every_case_agrees_with_the_reference(self):
        for name, window, history in _cases():
            with self.subTest(case=name):
                self._compare(name, window, history)

    def test_the_corpus_covers_every_reachable_rejection(self):
        reasons = set()
        for name, window, history in _cases():
            built = self.builder.build(window, history)
            if built.rejection_reason is not None:
                reasons.add(built.rejection_reason)
        self.assertEqual(
            reasons,
            {
                "camera stale",
                "pose stale",
                "camera-pose delta",
                "camera_window_stale",
                "model_pose_history_unavailable",
            },
            "a rejection branch lost its corpus coverage",
        )

    def test_the_corpus_also_covers_an_accepted_cycle(self):
        accepted = [
            name
            for name, window, history in _cases()
            if self.builder.build(window, history).accepted
        ]
        self.assertGreaterEqual(len(accepted), 3, accepted)


class GateOrderTests(unittest.TestCase):
    """Which reason wins when several gates would fire is observable."""

    def setUp(self) -> None:
        self.builder = _builder()

    def test_freshness_beats_camera_lag(self):
        # Both the age checks and the lag check would reject; the operator's
        # log must name the freshness one, as it always has.
        window = _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US, age_us=2_000_000),
                         frame_ids=_frame_ids())
        history = _history(t0_us=T0_US, start_offset_us=3_000_000,
                           arrival_us=NOW_US - 1000)
        self.assertEqual(self.builder.build(window, history).rejection_reason,
                         REASON_CAMERA_STALE)

    def test_camera_age_beats_pose_age(self):
        window = _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US, age_us=2_000_000),
                         frame_ids=_frame_ids())
        history = _history(t0_us=T0_US, arrival_us=NOW_US - 5_000_000)
        self.assertEqual(self.builder.build(window, history).rejection_reason,
                         REASON_CAMERA_STALE)

    def test_camera_lag_uses_the_camera_budget_not_the_pose_budget(self):
        # A lag between the two budgets must be accepted: the gate compares
        # against camera_stale_ms (1000), not pose_stale_ms (800).
        window = _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US),
                         frame_ids=_frame_ids())
        history = _history(t0_us=T0_US, start_offset_us=900_000,
                           arrival_us=NOW_US - 1000)
        built = self.builder.build(window, history)
        self.assertEqual(built.camera_lag_ms, 900.0)
        self.assertNotEqual(built.rejection_reason, REASON_CAMERA_WINDOW_STALE)

    def test_the_delta_gate_needs_a_clock_anomaly_at_these_settings(self):
        """The camera-pose delta gate is defensive, not live.

        Both ages are measured from one clock read, so with non-negative ages
        the delta can never exceed ``max(camera_stale_ms, pose_stale_ms)``
        (1000 vs 800 -> 1000).  The gate fires only above 1000, so it needs an
        arrival dated ahead of that read to trip.

        This is a pin, not an endorsement: it records that the branch is
        unreachable in the field today, so that shrinking
        ``camera_pose_delta_ms`` (or raising ``pose_stale_ms`` above it) is a
        change someone makes deliberately.
        """
        self.assertGreaterEqual(CAMERA_POSE_DELTA_MS,
                                max(CAMERA_STALE_MS, POSE_STALE_MS))
        # Worst case with non-negative ages: a pose stamped at the instant the
        # loop read the clock, against the stalest camera a cycle can carry.
        worst = _history(t0_us=T0_US, arrival_us=NOW_US)
        window = _window(t0_us=T0_US,
                         frame_arrivals_us=_arrivals(T0_US, age_us=1_000_000),
                         frame_ids=_frame_ids())
        built = self.builder.build(window, worst)
        self.assertEqual(built.camera_pose_delta_ms, 1000.0)
        self.assertNotEqual(built.rejection_reason, REASON_CAMERA_POSE_DELTA)
        # One microsecond of clock skew past the budget is enough to fire it.
        skewed = _history(t0_us=T0_US, arrival_us=NOW_US + 500_000)
        built = self.builder.build(window, skewed)
        self.assertGreater(built.camera_pose_delta_ms, CAMERA_POSE_DELTA_MS)
        self.assertEqual(built.rejection_reason, REASON_CAMERA_POSE_DELTA)

    def test_rejection_reasons_keep_their_historical_spellings(self):
        # Spaces, not underscores: these strings go to the takeover engine and
        # to the field log.  See the module docstring of input_builder.
        self.assertEqual(REASON_CAMERA_STALE, "camera stale")
        self.assertEqual(REASON_POSE_STALE, "pose stale")
        self.assertEqual(REASON_CAMERA_POSE_DELTA, "camera-pose delta")
        self.assertEqual(REASON_CAMERA_WINDOW_STALE, "camera_window_stale")
        self.assertEqual(REASON_POSE_HISTORY_UNAVAILABLE,
                         "model_pose_history_unavailable")


class ModelInputSemanticsTests(unittest.TestCase):
    """Shapes and units are the model's contract, not this refactor's."""

    def setUp(self) -> None:
        self.builder = _builder()
        self.window = _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US),
                              frame_ids=_frame_ids())
        self.history = _history(t0_us=T0_US, arrival_us=NOW_US - 1000)

    def test_tensor_shapes_are_unchanged(self):
        built = self.builder.build(self.window, self.history)
        self.assertEqual(tuple(built.ego_xyz.shape), (1, 1, HISTORY_STEPS, 3))
        self.assertEqual(tuple(built.ego_rot.shape), (1, 1, HISTORY_STEPS, 3, 3))
        self.assertEqual(built.ego_xyz.dtype, torch.float32)
        self.assertEqual(built.ego_rot.dtype, torch.float32)

    def test_the_reference_pose_is_the_newest_aligned_sample(self):
        built = self.builder.build(self.window, self.history)
        self.assertEqual(built.reference_pose.timestamp_us, self.window.t0_us)

    def test_the_ego_history_has_one_row_per_step(self):
        built = self.builder.build(self.window, self.history)
        self.assertEqual(len(built.ego_history), HISTORY_STEPS)
        stamps = [row["timestamp_us"] for row in built.ego_history]
        self.assertEqual(stamps[-1], self.window.t0_us)
        self.assertEqual(
            [b - a for a, b in zip(stamps, stamps[1:])],
            [int(HISTORY_DT_S * 1e6)] * (HISTORY_STEPS - 1),
        )

    def test_a_rejected_cycle_reports_no_model_input(self):
        history = _history(t0_us=T0_US, arrival_us=NOW_US - 5_000_000)
        built = self.builder.build(self.window, history)
        self.assertFalse(built.accepted)
        self.assertIsNone(built.ego_xyz)
        self.assertIsNone(built.ego_rot)
        self.assertIsNone(built.reference_pose)
        self.assertEqual(built.ego_history, [])
        self.assertEqual(built.camera_frame_ids, [])

    def test_the_clock_is_injectable(self):
        # Nothing in the builder may read the wall clock: moving the injected
        # clock moves the ages and nothing else.
        later = InputBuilder(
            history_steps=HISTORY_STEPS,
            history_dt_s=HISTORY_DT_S,
            camera_stale_ms=CAMERA_STALE_MS,
            pose_stale_ms=POSE_STALE_MS,
            camera_pose_delta_ms=CAMERA_POSE_DELTA_MS,
            now_us_fn=lambda: (NOW_US + 3_000_000) * 1000,
        )
        built = later.build(self.window, self.history)
        self.assertEqual(built.rejection_reason, REASON_CAMERA_STALE)
        self.assertEqual(built.camera_age_ms, 3000.0)


class BuilderStaysFreeTests(unittest.TestCase):
    """The builder must not reach the model, the decision or the wire."""

    def test_it_does_not_import_the_model_decision_or_sender(self):
        import ast

        tree = ast.parse(
            (ROOT / "inference" / "input_builder.py").read_text(encoding="utf-8")
        )
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        for forbidden in ("app", "pipeline", "waypoint", "trajectory.takeover",
                          "camera.sync"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, roots)

    def test_it_does_not_import_alpamayo(self):
        # input_builder is the half of the model layer that must load without
        # torch's CUDA stack or the checkpoint, so it must not import the
        # module that owns them.  Checked over the AST, not the text: the
        # module docstring names inference.alpamayo on purpose.
        import ast

        tree = ast.parse(
            (ROOT / "inference" / "input_builder.py").read_text(encoding="utf-8")
        )
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertNotIn("inference.alpamayo", imported)
        self.assertFalse(
            [name for name in imported if name.startswith("alpamayo")],
            f"the input builder reaches a model runtime: {sorted(imported)}",
        )

    def test_it_takes_the_history_as_an_argument(self):
        # No constructor-held history: the object it reads must be the one the
        # caller hands it, or a test could not pin a window against a fixture.
        # The only "history" state allowed is the two calibration numbers.
        built = _builder()
        held = {name for name in vars(built) if "history" in name}
        self.assertEqual(held, {"history_steps", "history_dt_s"},
                         f"the builder holds history state: {sorted(vars(built))}")

    def test_the_result_is_frozen(self):
        built = _builder().build(
            _window(t0_us=T0_US, frame_arrivals_us=_arrivals(T0_US)),
            _history(t0_us=T0_US, arrival_us=NOW_US - 1000),
        )
        with self.assertRaises(Exception):
            built.rejection_reason = "nope"


if __name__ == "__main__":
    unittest.main()
