"""The published snapshot, readiness, and the two response shapes.

``health.py`` is a stdlib-only leaf, so everything below runs with no runner, no
camera, no cv2 and no GPU.  The truth table is built by starting from a snapshot
that passes every readiness check and breaking exactly one field at a time, so a
failure names the rule it broke rather than "readiness said False".

The response-shape tests pin **key order**, not just membership:
``tools/check_chain.sh`` reads these bodies by key, and an operator reading raw
``curl`` output should not see the fields move.
"""

from __future__ import annotations

import ast
import dataclasses
import types
import unittest
from dataclasses import asdict
from pathlib import Path

# NO cv2 shim here, deliberately.  ``health.py`` is a stdlib-only leaf and this
# file imports nothing that needs cv2, so installing a stub would only leak into
# later-imported test modules: with ``sys.modules["cv2"]`` set to an empty module,
# ``test_model_test_backend`` (which needs a *working* cv2) imports successfully
# and then fails in six places instead of failing once at import.  A stub here
# would also contradict what this file claims to prove.
import health  # noqa: E402

# A fixed instant, in nanoseconds, so no test here reads a clock.  1000 s.
NOW_NS = 1_000_000_000_000
NOW_S = NOW_NS / 1e9
NOW_US = NOW_NS // 1000


def _time_sync(**overrides) -> health.TimeSyncView:
    fields = {
        "camera_clock": "capture_time_ns",
        "config_error": None,
    }
    fields.update(overrides)
    return health.TimeSyncView(**fields)


def _snapshot(**overrides) -> health.RuntimeSnapshot:
    """A snapshot that passes every readiness check, then one field changed."""
    fields = {
        "version": health.SNAPSHOT_VERSION,
        "sequence": 7,
        "published_at_monotonic": NOW_S,
        "phase": health.PHASE_RUNNING,
        "started_at_monotonic": NOW_S - 30.0,
        "cycle_id": 7,
        "inference_count": 7,
        "model_loaded": True,
        # 100 ms old, inside the 800 ms budget.
        "last_pose_arrival_us": NOW_US - 100_000,
        # A window 0.5 s old, inside the 10 s readiness bound.
        "last_window_ok_at": NOW_S - 0.5,
        "pose_mode": "mock",
        "camera_stale_ms": 1000.0,
        "pose_stale_ms": 800.0,
        "window_stale_s": health.WINDOW_STALE_S,
        "startup_grace_s": health.STARTUP_GRACE_S,
        "time_sync": _time_sync(),
    }
    fields.update(overrides)
    return health.RuntimeSnapshot(**fields)


@dataclasses.dataclass(frozen=True)
class _Mutable:
    """A frozen dataclass that still holds a list - the trap the walker exists for."""

    values: list = dataclasses.field(default_factory=list)


def _holds_mutable(value, depth: int = 0) -> bool:
    """Whether ``value`` holds a mutable container anywhere in its graph."""
    if depth > 4:
        return False
    if isinstance(value, (list, dict, set, bytearray)):
        return True
    if isinstance(value, tuple):
        return any(_holds_mutable(v, depth + 1) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _holds_mutable(getattr(value, f.name), depth + 1)
            for f in dataclasses.fields(value)
        )
    return False


def _readiness(**overrides) -> health.Readiness:
    return health.evaluate_readiness(
        _snapshot(**overrides), now_monotonic_ns=NOW_NS, pipeline_alive=True
    )


class SnapshotPublishingTests(unittest.TestCase):
    """The publish mechanics the whole design leans on."""

    def test_the_empty_snapshot_says_so(self):
        empty = health.EMPTY_SNAPSHOT
        self.assertEqual(empty.sequence, 0)
        self.assertEqual(empty.phase, health.PHASE_CREATED)
        self.assertEqual(empty.version, health.SNAPSHOT_VERSION)
        # An operator reading a handler that ran before startup must be able to
        # tell "never published" from "published with zeroes".
        self.assertIsNone(health.snapshot_age_ms(empty, NOW_NS))

    def test_advance_bumps_the_sequence_and_the_version(self):
        first = health.advance(health.EMPTY_SNAPSHOT, inference_count=3)
        second = health.advance(first, inference_count=4)
        self.assertEqual(second.sequence, first.sequence + 1)
        self.assertEqual(second.version, health.SNAPSHOT_VERSION)
        self.assertEqual(second.inference_count, 4)
        self.assertGreater(second.published_at_monotonic, 0.0)

    def test_advance_carries_untouched_fields_forward(self):
        first = health.advance(health.EMPTY_SNAPSHOT, model_loaded=True, cycle_id=9)
        second = health.advance(first, inference_count=2)
        self.assertTrue(second.model_loaded)
        self.assertEqual(second.cycle_id, 9)
        self.assertEqual(second.inference_count, 2)

    def test_a_typo_in_a_publisher_is_a_loud_failure(self):
        # The point of routing every publish through advance: a misspelled field
        # would otherwise be accepted silently and never update.
        with self.assertRaises(TypeError):
            health.advance(health.EMPTY_SNAPSHOT, inference_cout=1)

    def test_a_published_snapshot_is_not_affected_by_the_next_publish(self):
        """Immutability, asserted rather than assumed.

        The reader takes one reference and uses it; if a second publish could
        mutate the object the first reader holds, every handler would be racy
        again despite the single-reference store.
        """
        first = health.advance(health.EMPTY_SNAPSHOT, inference_count=1, cycle_id=1)
        before = asdict(first)
        health.advance(first, inference_count=99, cycle_id=99, model_loaded=True)
        self.assertEqual(asdict(first), before)

    def test_the_snapshot_holds_no_mutable_container(self):
        """Deep immutability, walked over the real object graph.

        Walks the graph rather than ``asdict``: ``asdict`` recurses *into* the
        nested frozen dataclass and hands back a plain dict, which the walker
        would report as a mutable field that does not actually exist.
        """
        snapshot = _snapshot(last_takeover_decision=None, unhealthy_readers=("a",))
        for field in dataclasses.fields(snapshot):
            with self.subTest(field=field.name):
                self.assertFalse(
                    _holds_mutable(getattr(snapshot, field.name)),
                    f"{field.name} holds a mutable container; a reader could "
                    f"then see a half-updated value and the lockless design "
                    f"loses its guarantee",
                )

    def test_the_walker_itself_notices_a_mutable_container(self):
        """A positive control, so the immutability test above can actually fail."""
        for bad in ([1], {"a": 1}, {1, 2}, bytearray(b"x"), ([1],), _Mutable()):
            with self.subTest(value=bad):
                self.assertTrue(_holds_mutable(bad))
        for good in (1, "x", None, 1.5, (), ("a", "b"), _time_sync(), _snapshot()):
            with self.subTest(value=good):
                self.assertFalse(_holds_mutable(good))

    def test_a_stale_snapshot_ages_rather_than_reporting_zero(self):
        snapshot = health.RuntimeSnapshot(sequence=3, published_at_monotonic=NOW_S - 4.0)
        self.assertAlmostEqual(
            health.snapshot_age_ms(snapshot, NOW_NS), 4000.0, places=3
        )

    def test_uptime_needs_a_start(self):
        self.assertIsNone(health.uptime_s(health.EMPTY_SNAPSHOT, NOW_NS))
        self.assertAlmostEqual(
            health.uptime_s(_snapshot(started_at_monotonic=NOW_S - 12.0), NOW_NS),
            12.0,
            places=3,
        )

    def test_pose_age_needs_a_pose(self):
        self.assertIsNone(
            health.pose_age_ms(_snapshot(last_pose_arrival_us=None), NOW_NS)
        )
        self.assertAlmostEqual(
            health.pose_age_ms(
                _snapshot(last_pose_arrival_us=NOW_US - 250_000), NOW_NS
            ),
            250.0,
            places=3,
        )


class ThePublisherPopulatesWhatItPublishesTests(unittest.TestCase):
    """No field the publisher reads is a default that nothing ever updates.

    This exists because of a real defect.  ``_publish_snapshot`` passed
    ``last_window_ok_at=self._last_window_seen_at``, and that attribute was
    declared in ``__init__`` as ``None`` and then never assigned anywhere -
    a sentinel default with no writer.  Every unit test in this file passed,
    because they set the field on the snapshot directly; the only thing that
    noticed was the full-chain smoke test, where /ready's freshness check would
    have reported ``no_camera_window`` forever for a pipeline that was building
    windows perfectly well.

    So the rule is deliberately *not* "is it assigned somewhere" - the sentinel
    ``= None`` in ``__init__`` satisfies that, and it is exactly the shape of
    the bug.  The rule is: every field the publisher reads must have at least
    one assignment that is something *other* than a literal default inside
    ``__init__``.  A name written once as ``self.error = None`` in ``__init__``
    and once as ``self.error = None`` in the loop counts as written, because the
    loop's assignment is part of the cycle's real work; a name written only in
    ``__init__`` does not.

    Collaborators injected once at construction (``self.cameras``,
    ``self.settings``) are written by a call or a name, not by a literal, so
    they are unaffected by that distinction.

    Parsed, not imported: reading pipeline.py's source keeps this file free of
    cv2, torch and the camera stack, which is the whole reason health.py can be
    tested at all.
    """

    PACKAGE_ROOT = Path(__file__).resolve().parents[1]

    @staticmethod
    def _self_attributes(node) -> set[str]:
        """Direct ``self.X`` data references under *node*, read or written.

        ``self.X(...)`` is excluded: that is a method call, and a method needs no
        assignment.  Only the *base* of an attribute chain counts, so
        ``self.cameras.unhealthy_readers()`` reports ``cameras`` - a data
        reference - and not ``unhealthy_readers``.
        """
        called = {
            child.func.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "self"
        }
        return {
            child.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == "self"
        } - called

    @staticmethod
    def _assigned_names(node) -> set[str]:
        """``self.X`` names that *node* assigns to, by any assignment form."""
        names: set[str] = set()
        for item in ast.walk(node):
            if isinstance(item, ast.Assign):
                targets = item.targets
            elif isinstance(item, (ast.AugAssign, ast.AnnAssign)):
                targets = [item.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    names.add(target.attr)
        return names

    @staticmethod
    def _writer_tags(tree: ast.Module) -> dict[str, set[str]]:
        """Per attribute, which kinds of writer it has.

        ``"literal_default"`` is a literal assigned inside ``__init__`` - a
        sentinel, not work.  ``"real"`` is anything else: an assignment to a
        call or a name, an augmented assignment, or any literal assigned outside
        ``__init__`` (a method assigning a literal is doing the cycle's work,
        even if the value happens to be ``None``).

        Tracked per name rather than as sets of names, because a name can have
        both kinds of writer and the difference between "only a sentinel" and
        "a sentinel plus a real write" is the whole point.
        """
        constructors = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        ]
        in_constructor = {
            child for ctor in constructors for child in ast.walk(ctor)
        }

        tags: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            elif isinstance(node, ast.AugAssign):
                targets, value = [node.target], None
            else:
                continue
            is_default = value is not None and isinstance(value, ast.Constant)
            tag = (
                "literal_default"
                if is_default and node in in_constructor
                else "real"
            )
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    tags.setdefault(target.attr, set()).add(tag)
        return tags

    @classmethod
    def _populated_attributes(cls, tree: ast.Module) -> set[str]:
        """Names with at least one writer that is more than a sentinel."""
        return {
            name
            for name, kinds in cls._writer_tags(tree).items()
            if "real" in kinds
        }

    def test_no_publisher_input_is_a_default_that_nothing_updates(self):
        tree = ast.parse((self.PACKAGE_ROOT / "pipeline.py").read_text(encoding="utf-8"))
        publisher = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_publish_snapshot"
            ),
            None,
        )
        self.assertIsNotNone(publisher, "pipeline.py has no _publish_snapshot")

        never_populated = sorted(
            self._self_attributes(publisher) - self._populated_attributes(tree)
        )
        self.assertEqual(
            never_populated,
            [],
            "these are read by _publish_snapshot but nothing ever assigns them "
            "more than a literal default in __init__, so the snapshot field they "
            f"feed can never become true: {never_populated}",
        )

    def test_the_check_would_notice_a_default_with_no_writer(self):
        """A positive control on the predicate, not just on pipeline.py.

        Without this, a typo that made the analysis vacuous - matching nothing,
        or matching everything - would leave the test above green and worthless.
        """
        source = (
            "class T:\n"
            "    def __init__(self):\n"
            "        self.never = None\n"
            "        self.also_never: float | None = None\n"
            "        self.really = None\n"
            "        self.collaborator = object()\n"
            "    def loop(self):\n"
            "        self.really = None\n"
            "        if self.never and self.also_never and self.really:\n"
            "            pass\n"
        )
        tree = ast.parse(source)
        populated = self._populated_attributes(tree)
        self.assertIn("really", populated, "a literal assigned in a method is a write")
        self.assertIn("collaborator", populated, "a non-literal is not a default")
        self.assertNotIn("never", populated, "a lone None default is not a write")
        self.assertNotIn("also_never", populated, "annotated or not, same rule")


class ReadinessTruthTableTests(unittest.TestCase):
    def test_the_reference_snapshot_is_ready(self):
        result = _readiness()
        self.assertTrue(result.ready, result.as_dict())
        self.assertEqual(result.reason, "")
        self.assertEqual(result.failing, ())

    def test_a_shutting_down_pipeline_outranks_a_dead_one(self):
        # Order matters, not just membership: the operator's first question
        # about a 503 during a restart is "are you stopping", not "is it dead".
        result = _readiness(phase=health.PHASE_STOPPING)
        self.assertEqual(result.reason, "shutting_down")
        self.assertEqual(result.failing[0], "not_stopping")

    def test_a_stopped_pipeline_is_not_ready(self):
        self.assertEqual(_readiness(phase=health.PHASE_STOPPED).reason, "shutting_down")

    def test_a_dead_thread_is_reported_before_a_missing_start(self):
        result = health.evaluate_readiness(
            _snapshot(started_at_monotonic=None),
            now_monotonic_ns=NOW_NS,
            pipeline_alive=False,
        )
        self.assertEqual(result.reason, "pipeline_not_running")
        self.assertIn("pipeline_started", result.failing)

    def test_never_started_is_starting_not_broken(self):
        result = _readiness(started_at_monotonic=None)
        self.assertEqual(result.reason, "starting")
        self.assertIn("pipeline_started", result.failing)

    def test_no_window_inside_the_startup_budget_is_starting(self):
        """The checkpoint load is slow, and the deploy script's own gate is 60 s.

        Inside that budget this is still not ready - a consumer must not be told
        to proceed to a pipeline that has never seen a frame - but the reason it
        gives is the honest one.
        """
        result = _readiness(started_at_monotonic=NOW_S - 5.0, last_window_ok_at=None)
        self.assertFalse(result.ready)
        self.assertEqual(result.reason, "starting")
        self.assertIn("startup_complete", result.failing)
        detail = next(
            c for c in result.checks if c.name == "startup_complete"
        ).detail
        self.assertIn("grace_s=60", detail)

    def test_no_window_past_the_startup_budget_is_a_fault(self):
        result = _readiness(started_at_monotonic=NOW_S - 120.0, last_window_ok_at=None)
        self.assertFalse(result.ready)
        self.assertEqual(result.reason, "no_camera_window")

    def test_never_having_a_window_is_not_reported_twice(self):
        """The double-report guard.

        Check 4 owns "no window has EVER been built".  If check 5 also failed on
        ``last_window_ok_at is None``, the first multi-second inference - which
        happens before this iteration can publish - would be reported as a stale
        window instead of a starting pipeline, and the fix would look applied
        while the operator still saw the wrong verdict.
        """
        result = _readiness(started_at_monotonic=NOW_S - 1.0, last_window_ok_at=None)
        failing = list(result.failing)
        self.assertEqual(failing.count("startup_complete"), 1)
        self.assertNotIn("camera_window_fresh", failing)
        freshness = next(c for c in result.checks if c.name == "camera_window_fresh")
        self.assertTrue(freshness.ok)
        self.assertEqual(freshness.detail, "no_window_yet")

    def test_an_old_window_is_stale(self):
        result = _readiness(last_window_ok_at=NOW_S - 11.0)
        self.assertEqual(result.reason, "camera_window_stale")

    def test_a_slow_but_healthy_pipeline_does_not_flap(self):
        """Why the bound is the loop's 10 s watchdog and not ``camera_stale_ms``.

        ``camera_stale_ms`` (1000 ms) is a per-frame gate budget, and
        ``last_window_ok_at`` only advances once per loop iteration.  A healthy
        pipeline whose inference takes a few seconds - the normal case with the
        real checkpoint - would then be permanently "not ready", which is the
        exact false alarm that trains operators to ignore the endpoint.
        """
        result = _readiness(
            last_window_ok_at=NOW_S - 3.0, camera_stale_ms=1000.0
        )
        self.assertTrue(result.ready, result.as_dict())

    def test_no_pose_at_all_is_distinguished_from_a_stale_pose(self):
        self.assertEqual(
            _readiness(last_pose_arrival_us=None).reason, "pose_unavailable"
        )
        self.assertEqual(
            _readiness(last_pose_arrival_us=NOW_US - 900_000).reason, "pose_stale"
        )

    def test_a_pose_just_inside_the_budget_is_fresh(self):
        self.assertTrue(_readiness(last_pose_arrival_us=NOW_US - 800_000).ready)

    def test_a_loaded_model_is_required(self):
        result = _readiness(model_loaded=False)
        self.assertEqual(result.reason, "model_not_loaded")
        self.assertIn("model_loaded", result.failing)

    def test_a_time_sync_config_error_fails_readiness(self):
        result = _readiness(time_sync=_time_sync(config_error="bad pose_mode"))
        self.assertEqual(result.reason, "time_sync_config_error")

    def test_a_missing_time_sync_view_is_not_a_failure(self):
        self.assertTrue(_readiness(time_sync=None).ready)


class ReadinessWarningTests(unittest.TestCase):
    """Warnings must never flip the status code - that is their whole point."""

    def _warns(self, **overrides) -> tuple:
        result = _readiness(**overrides)
        self.assertTrue(result.ready, f"a warning became fatal: {result.as_dict()}")
        return result.warnings

    def test_unhealthy_readers_warn_without_failing(self):
        warnings = self._warns(unhealthy_readers=("udp://127.0.0.1:31101",))
        self.assertIn("readers_unhealthy:1", warnings)

    def test_a_silent_route_receiver_warns_without_failing(self):
        self.assertIn("route_receiver_not_listening", self._warns(route_receiver_listening=False))

    def test_a_route_receiver_error_warns_inline(self):
        warnings = self._warns(route_receiver_error="bind failed")
        self.assertIn("route_receiver_error:bind failed", warnings)

    def test_disabled_route_receiver_is_not_a_warning(self):
        # ``None`` means the receiver was never configured, which is a
        # deployment choice, not a fault.
        self.assertEqual(self._warns(route_receiver_listening=None), ())

    def test_a_listening_route_receiver_is_not_a_warning(self):
        self.assertEqual(self._warns(route_receiver_listening=True), ())

    def test_a_pipeline_error_warns_without_failing(self):
        self.assertIn("pipeline_error:boom", self._warns(last_error="boom"))

    def test_a_rejected_gate_warns_without_failing(self):
        warnings = self._warns(last_rejection_reason="camera_window_stale")
        self.assertIn("gate_rejected:camera_window_stale", warnings)

    def test_every_warning_can_be_present_at_once(self):
        warnings = self._warns(
            unhealthy_readers=("a", "b"),
            route_receiver_listening=False,
            route_receiver_error="bind failed",
            last_error="boom",
            last_rejection_reason="pose_stale",
        )
        self.assertEqual(len(warnings), 5)
        self.assertIn("readers_unhealthy:2", warnings)


class _Sources:
    """The live half of the two handlers, with no runner behind it."""

    @staticmethod
    def health_sources() -> health.HealthSources:
        return health.HealthSources(
            settings=types.SimpleNamespace(pose_mode="udp", camera_stale_ms=1000.0),
            camera_health={"0.0.0.0:30106#cam1": "HEALTHY"},
            camera_stats={"window_builds": 4, "refusals": {}, "readers": []},
            camera_cimg={"peer": "172.16.0.9:41234", "connected": True},
        )

    @staticmethod
    def status_sources(*, present: bool = True) -> health.StatusSources:
        if not present:
            return health.StatusSources(
                settings=types.SimpleNamespace(reference_source="planned_route"),
                runner_present=False,
            )
        return health.StatusSources(
            settings=types.SimpleNamespace(reference_source="planned_route"),
            runner_present=True,
            timestamp_manager=types.SimpleNamespace(),
            pose_listener=types.SimpleNamespace(
                _chassis_total=41,
                latest_chassis=lambda: types.SimpleNamespace(
                    timestamp_us=1_700_000_000_000_002,
                    speed_mps=2.5,
                    yaw_rate_rad_s=0.01,
                ),
                last_timestamp_us=1_700_000_000_000_003,
            ),
            history=types.SimpleNamespace(
                latest=lambda: types.SimpleNamespace(
                    timestamp_us=1_700_000_000_000_000,
                    payload_timestamp_us=1_700_000_000_000_001,
                )
            ),
            planned_route_cache=types.SimpleNamespace(snapshot=lambda: (None, 0)),
            route_tcp_receiver=types.SimpleNamespace(listening=True, last_error=None),
        )


# The /health prefix, in order.  tools/check_chain.sh reads camera_stats,
# inference_count and last_error by name.  The retired splitter peer is gone;
# the vehicle address lives on camera_cimg.peer.
HEALTH_KEYS_BEFORE_STAGE_3 = (
    "ok",
    "pipeline_alive",
    "pose_mode",
    "inference_count",
    "last_error",
    "camera_health",
    "camera_stats",
    "pose_source",
    "camera_stale_ms",
    "time_sync",
)

# The additive tail.  Additive on purpose: an operator's field order is
# unchanged, and a reader that indexes by name cannot break.
HEALTH_KEYS_ADDED = (
    "phase",
    "cycle_id",
    "snapshot_sequence",
    "snapshot_age_ms",
    "uptime_s",
    "model_loaded",
    "ready",
    # Direct-JPEG CIMG link report; {} unless that input mode is selected.
    "camera_cimg",
    # Appended with the multi-GPU pool.  Both are null in inline mode, so the
    # body of a single-GPU run is unchanged but for two nulls.
    "inference_pool",
    "inference_metrics",
)

# The /status body, frozen.  check_chain.sh reads chassis_packets and
# pose_timestamp_mode from it and computes the pose rate from their deltas, so
# nothing existing may move; the CIMG link report is appended at the end.
STATUS_KEYS = (
    "pose_timestamp_mode",
    "pose_last_ts_us",
    "pose_payload_ts_us",
    "chassis_last_ts_us",
    "chassis_speed_mps",
    "chassis_yaw_rate_rad_s",
    "chassis_packets",
    "pose_timestamp",
    "last_t0_us",
    "reference_source_mode",
    "planned_route_cache",
    "last_control_flag",
    "last_sent_control_flag",
    "last_takeover_decision",
    "last_num_waypoints",
    "last_cot",
    # Appended: the direct-JPEG CIMG link report.
    "camera_cimg",
    # Appended with the multi-GPU pool; both null in inline mode.  The same two
    # objects as on /health, from the same snapshot.
    "inference_pool",
    "inference_metrics",
)

PLANNED_ROUTE_KEYS = (
    "receiver_listening",
    "receiver_error",
    "available",
    "version",
    "route_id",
    "coordinate_frame",
    "num_points",
    "source_timestamp_us",
)

POSE_TIMESTAMP_KEYS = ("timestamp_us",)

TAKEOVER_KEYS = (
    "mean_lateral_error_m",
    "max_lateral_error_m",
    "takeover_count",
    "release_count",
    "decision_reason",
    "delay_s",
    "start_index",
    "remaining_points",
    "reference_source",
    "reference_fallback_reason",
)

TIME_SYNC_KEYS = (
    "camera_clock",
    "config_error",
)


class HealthBodyShapeTests(unittest.TestCase):
    def test_the_historical_keys_come_first_in_their_original_order(self):
        body = health.build_health_body(
            _snapshot(),
            _Sources.health_sources(),
            now_monotonic_ns=NOW_NS,
            pipeline_alive=True,
        )
        self.assertEqual(tuple(body)[: len(HEALTH_KEYS_BEFORE_STAGE_3)],
                         HEALTH_KEYS_BEFORE_STAGE_3)

    def test_the_additive_keys_are_appended_and_nothing_else(self):
        body = health.build_health_body(
            _snapshot(),
            _Sources.health_sources(),
            now_monotonic_ns=NOW_NS,
            pipeline_alive=True,
        )
        self.assertEqual(tuple(body)[len(HEALTH_KEYS_BEFORE_STAGE_3):],
                         HEALTH_KEYS_ADDED)

    def test_ok_stays_hardcoded_true(self):
        # Frozen wire value: the deploy script and the chain tool both treat any
        # 200 here as "uvicorn is up".  The verdict to act on is ``ready``.
        body = health.build_health_body(
            _snapshot(phase=health.PHASE_STOPPED, inference_count=0),
            _Sources.health_sources(),
            now_monotonic_ns=NOW_NS,
            pipeline_alive=False,
        )
        self.assertIs(body["ok"], True)
        self.assertIs(body["ready"], False)

    def test_the_live_half_is_the_live_objects(self):
        body = health.build_health_body(
            _snapshot(),
            _Sources.health_sources(),
            now_monotonic_ns=NOW_NS,
            pipeline_alive=True,
        )
        self.assertEqual(body["camera_stats"]["window_builds"], 4)
        self.assertEqual(body["camera_cimg"]["peer"], "172.16.0.9:41234")
        self.assertNotIn("camera_splitter_peer", body)
        self.assertEqual(body["pose_mode"], "udp")
        self.assertEqual(body["camera_stale_ms"], 1000.0)

    def test_the_cycle_half_is_the_snapshot(self):
        body = health.build_health_body(
            _snapshot(inference_count=42, last_error="boom", cycle_id=9),
            _Sources.health_sources(),
            now_monotonic_ns=NOW_NS,
            pipeline_alive=True,
        )
        self.assertEqual(body["inference_count"], 42)
        self.assertEqual(body["last_error"], "boom")
        self.assertEqual(body["cycle_id"], 9)
        self.assertEqual(body["snapshot_sequence"], 7)

    def test_the_time_sync_block_keeps_its_key_order(self):
        body = health.build_health_body(
            _snapshot(),
            _Sources.health_sources(),
            now_monotonic_ns=NOW_NS,
            pipeline_alive=True,
        )
        self.assertEqual(tuple(body["time_sync"]), TIME_SYNC_KEYS)

    def test_a_snapshot_with_no_time_sync_is_null_not_missing(self):
        body = health.build_health_body(
            _snapshot(time_sync=None),
            _Sources.health_sources(),
            now_monotonic_ns=NOW_NS,
            pipeline_alive=True,
        )
        self.assertIsNone(body["time_sync"])


class StatusBodyShapeTests(unittest.TestCase):
    def test_the_key_set_and_order_are_frozen(self):
        body = health.build_status_body(_snapshot(), _Sources.status_sources())
        self.assertEqual(tuple(body), STATUS_KEYS)

    def test_the_nested_blocks_keep_their_key_order(self):
        body = health.build_status_body(_snapshot(), _Sources.status_sources())
        self.assertEqual(tuple(body["planned_route_cache"]), PLANNED_ROUTE_KEYS)
        self.assertEqual(tuple(body["pose_timestamp"]), POSE_TIMESTAMP_KEYS)

    def test_the_vehicle_rate_fields_come_from_the_live_listener(self):
        body = health.build_status_body(_snapshot(), _Sources.status_sources())
        self.assertEqual(body["chassis_packets"], 41)
        self.assertEqual(body["chassis_speed_mps"], 2.5)
        self.assertEqual(body["pose_timestamp_mode"], "unix")

    def test_the_newest_pose_comes_from_the_history_not_the_listener(self):
        """Both are writers; mock mode has no listener at all.

        Reading the pose out of the listener would report nothing on a mock
        deployment and something stale on a real one, where the listener records
        the last packet it *decoded* rather than the newest pose in the history
        the loop actually gates on.
        """
        sources = _Sources.status_sources()
        sources = health.StatusSources(
            settings=sources.settings,
            runner_present=sources.runner_present,
            pose_listener=types.SimpleNamespace(
                _chassis_total=1,
                latest_chassis=lambda: None,
                last_timestamp_us=1,
            ),
            history=types.SimpleNamespace(
                latest=lambda: types.SimpleNamespace(
                    timestamp_us=777, payload_timestamp_us=778
                )
            ),
        )
        body = health.build_status_body(_snapshot(), sources)
        self.assertEqual(body["pose_last_ts_us"], 777)

    def test_the_cycle_half_is_the_snapshot(self):
        snapshot = _snapshot(
            last_t0_us=1_700_000_000_000_000,
            last_control_flag=1,
            last_sent_control_flag=0,
            last_num_waypoints=64,
            last_cot="c" * 500,
        )
        body = health.build_status_body(snapshot, _Sources.status_sources())
        self.assertEqual(body["last_t0_us"], 1_700_000_000_000_000)
        self.assertEqual(body["last_control_flag"], 1)
        self.assertEqual(body["last_sent_control_flag"], 0)
        self.assertEqual(body["last_num_waypoints"], 64)

    def test_cot_is_truncated_and_absent_cot_is_null(self):
        long_cot = health.build_status_body(
            _snapshot(last_cot="c" * 500), _Sources.status_sources()
        )["last_cot"]
        self.assertEqual(long_cot, "c" * 400 + "...")
        self.assertIsNone(
            health.build_status_body(
                _snapshot(last_cot=None), _Sources.status_sources()
            )["last_cot"]
        )

    def test_a_zero_waypoint_count_is_reported_as_zero_not_null(self):
        # The release frame: the trajectory was sent with count=0.  ``None``
        # here would mean "no runner", which is a different thing.
        body = health.build_status_body(
            _snapshot(last_num_waypoints=0), _Sources.status_sources()
        )
        self.assertEqual(body["last_num_waypoints"], 0)

    def test_a_takeover_decision_keeps_its_key_order(self):
        decision = types.SimpleNamespace(
            mean_lateral_error=1.25,
            max_lateral_error=1.75,
            takeover_count=2,
            release_count=0,
            decision_reason="takeover",
            delay_s=0.4,
            start_index=3,
            remaining_points=61,
            reference_source="planned_route",
            reference_fallback_reason=None,
        )
        body = health.build_status_body(
            _snapshot(last_takeover_decision=decision), _Sources.status_sources()
        )
        self.assertEqual(tuple(body["last_takeover_decision"]), TAKEOVER_KEYS)

    def test_no_runner_yields_the_null_shaped_body_and_never_raises(self):
        """``tests/test_waypoint_trimming.py`` calls this path directly.

        The snapshot on this path is the empty one - no runner ever published -
        which is what makes the two null-shaped fields null rather than zero.
        """
        body = health.build_status_body(
            health.EMPTY_SNAPSHOT, _Sources.status_sources(present=False)
        )
        self.assertEqual(tuple(body), STATUS_KEYS)
        self.assertIsNone(body["last_t0_us"])
        self.assertIsNone(body["last_control_flag"])
        self.assertIsNone(body["last_sent_control_flag"])
        self.assertIsNone(body["last_takeover_decision"])
        self.assertIsNone(body["pose_timestamp"])
        self.assertEqual(body["chassis_packets"], 0)
        self.assertIsNone(body["reference_source_mode"])
        self.assertEqual(body["last_num_waypoints"], 0)
        # The receiver was never constructed, so it is not listening - and
        # ``available`` is about the cache, which is empty.
        self.assertFalse(body["planned_route_cache"]["receiver_listening"])
        self.assertFalse(body["planned_route_cache"]["available"])

    def test_a_runner_that_has_never_run_reports_zero_waypoints(self):
        # No cycle has produced a trajectory yet, so the count is zero and the
        # control flags are still null.
        body = health.build_status_body(_snapshot(), _Sources.status_sources())
        self.assertIsNone(body["last_control_flag"])
        self.assertEqual(body["last_num_waypoints"], 0)


if __name__ == "__main__":
    unittest.main()
