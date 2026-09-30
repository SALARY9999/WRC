"""Which fields the HTTP handlers read live, and which come from the snapshot.

The Stage 3 contract is a split, not a wholesale swap:

* the **cycle-derived** half (``inference_count``, ``last_error``, the time-sync
  view, both control flags, the decision, the trajectory
  summary) comes from one published ``RuntimeSnapshot``, so a response describes
  a single cycle;
* the **rate-shaped** half (decoder counters, the vehicle's packet counters, the
  planned-route cache, ``is_alive()``) stays a live read, because freezing it
  would report "0 fps on all four cameras" exactly when an operator is trying to
  tell a dead decoder from a stuck pipeline - see health.py's docstring.

One test pins the whole split, so a future refactor that swaps a field from one
side to the other fails here with the field named.

WHY THIS FILE IS SEPARATE FROM ``test_health_snapshot.py``
    It imports ``app``, which imports ``cv2`` at module level, and the local
    environment has no cv2.  The stub is installed, used, and then torn down
    completely by ``cv2_stub`` - see that module's docstring, and
    ``ImportHygieneTests``, which asserts the tear-down from a fresh child
    interpreter.  Without it this file would silently change the outcome of
    ``test_model_test_backend`` (which needs a *working* cv2) purely because of
    where its own name falls in discovery order.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

# cv2 is absent locally but ``app`` imports it at module level, so a stub is
# needed to import ``app`` at all.  ``cv2_stub`` removes it again - and, unlike
# deleting the ``sys.modules`` entry alone, also drops every module that
# captured it while it was installed.  That second half is what protects a
# later ``import camera.sync``: ``camera.sync`` binds cv2 to its own global at import
# time, so a bare ``del sys.modules["cv2"]`` would leave it holding an empty
# module and turn "cv2 is not installed" into
# ``AttributeError: module 'cv2' has no attribute 'resize'``.
from _cv2_isolation import CV2_ABSENT_PRELUDE, cv2_stub  # noqa: E402

from fastapi import Response  # noqa: E402

import health  # noqa: E402
from test_health_snapshot import (  # noqa: E402
    HEALTH_KEYS_ADDED,
    HEALTH_KEYS_BEFORE_STAGE_3,
    STATUS_KEYS,
)

with cv2_stub():
    import app

HEALTH_KEYS = HEALTH_KEYS_BEFORE_STAGE_3 + HEALTH_KEYS_ADDED


class ImportHygieneTests(unittest.TestCase):
    """The invariant the cv2 shim above depends on, asserted rather than assumed."""

    def test_importing_this_module_leaves_no_cv2_stub_behind(self):
        """Checked in a subprocess, and that is deliberate.

        By the time the suite runs, a *later* test module may have installed its
        own cv2 stub, so an in-process ``'cv2' not in sys.modules`` assertion
        would fail for a reason that has nothing to do with this file.  A fresh
        interpreter that imports only this module measures the thing meant.

        Two things are asserted, because only the second one is the real
        invariant.  ``sys.modules`` being clean is necessary but not
        sufficient: the first version of this shim deleted the entry and still
        left ``camera.sync`` holding the stub.  So the probe also imports
        ``camera.sync`` and
        requires it to fail the honest way - with cv2 genuinely missing.

        The probe is given a machine without cv2 rather than reading the one it
        is on.  This file passes either way - the shim hides a real cv2 just as
        well as an absent one - but ``camera.sync`` would then import cleanly on
        the deployment server and the invariant would go unmeasured there.
        """
        probe = CV2_ABSENT_PRELUDE + (
            "import test_http_source_split, sys\n"
            "try:\n"
            "    import camera.sync\n"
            "except ModuleNotFoundError as exc:\n"
            "    print('cv2' in sys.modules, str(exc))\n"
            "else:\n"
            "    print('cv2' in sys.modules, 'camera.sync imported anyway')\n"
        )
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(ROOT / "tests"),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "False No module named 'cv2'",
            "importing this module left a cv2 stub reachable from the process, "
            "which changes the outcome of a later camera.sync import "
            "depending on file-name order",
        )


class _HandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = app._runner
        app._runner = None
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        app._runner = self._saved

    def _publish(
        self,
        *,
        model_loaded: bool = False,
        phase: str | None = None,
        **updates,
    ) -> app.PipelineRunner:
        """A real runner whose next publish carries ``updates``.

        Two of these are NOT ``setattr``-able, and both silently do nothing if
        you try:

        * ``model_loaded`` - the publisher reads ``self.model.loaded``, so it is
          set by swapping in a stub model;
        * ``phase`` - the publisher carries ``self.snapshot.phase`` forward, so a
          transition has to arrive as the publish kwarg.  Only lifecycle code
          sets it, which is the point.
        """
        runner = app.PipelineRunner(app._settings)
        runner.model = _FakeModel(model_loaded)
        for name, value in updates.items():
            setattr(runner, name, value)
        runner._publish_snapshot(**({"phase": phase} if phase is not None else {}))
        app._runner = runner
        self.addCleanup(self._release, runner)
        return runner

    @staticmethod
    def _release(runner: app.PipelineRunner) -> None:
        """Close what a constructed runner opens.

        ``PipelineRunner.__init__`` builds a ``WaypointSender``, which creates a
        UDP socket and - as of Stage 3 step 3 - still has no ``close()``; nothing
        anywhere closes it, in a test or in production.  That is the shutdown gap
        step 5 closes with ``WaypointSender.close()``.  Until then, a test that
        builds a runner has to release the socket itself or the suite leaks one
        file descriptor per construction, which is what the ResourceWarnings this
        cleanup removes were reporting.
        """
        sock = getattr(getattr(runner, "sender", None), "sock", None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


class _FakeModel:
    """Only ``loaded``, which is all the publisher asks of the model."""

    def __init__(self, loaded: bool) -> None:
        self.loaded = loaded


class NoRunnerTests(_HandlerTest):
    """The shape an operator sees before startup and after shutdown."""

    def test_health_still_answers_with_no_runner(self):
        body = app.health()
        self.assertEqual(tuple(body), HEALTH_KEYS)
        self.assertIs(body["ok"], True)
        self.assertIs(body["pipeline_alive"], False)
        self.assertEqual(body["inference_count"], 0)
        self.assertIsNone(body["last_error"])
        self.assertIsNone(body["time_sync"])
        self.assertEqual(body["camera_health"], {})
        self.assertEqual(body["camera_stats"], {})

    def test_status_still_answers_with_no_runner(self):
        # ``tests/test_waypoint_trimming.py`` already pins this path; repeated
        # here so both handlers are covered by one file's contract.
        body = app.status()
        self.assertEqual(tuple(body), STATUS_KEYS)
        self.assertIsNone(body["last_control_flag"])
        self.assertIsNone(body["last_sent_control_flag"])
        self.assertEqual(body["chassis_packets"], 0)

    def test_ready_is_503_when_nothing_has_been_started(self):
        response = Response()
        body = app.ready(response)
        self.assertEqual(response.status_code, 503)
        self.assertIs(body["ready"], False)
        # ``reason`` is the single first failure; ``failing`` names every failed
        # check.  Asserting the reason here pins the severity order.
        self.assertEqual(body["reason"], "pipeline_not_running")
        self.assertIn("pipeline_alive", body["failing"])


class HealthKeyContractTests(_HandlerTest):
    def test_the_response_is_the_historical_body_plus_the_additive_tail(self):
        self._publish(inference_count=7)
        body = app.health()
        self.assertEqual(tuple(body)[: len(HEALTH_KEYS_BEFORE_STAGE_3)],
                         HEALTH_KEYS_BEFORE_STAGE_3)
        self.assertEqual(tuple(body)[len(HEALTH_KEYS_BEFORE_STAGE_3):],
                         HEALTH_KEYS_ADDED)

    def test_status_gains_no_field(self):
        # check_chain.sh derives the pose rate from the deltas of these keys.
        self._publish(inference_count=7)
        self.assertEqual(tuple(app.status()), STATUS_KEYS)

    def test_the_nested_blocks_are_unchanged(self):
        self._publish(inference_count=7)
        body = app.status()
        self.assertEqual(tuple(body["planned_route_cache"])[0], "receiver_listening")
        self.assertIn("available", body["planned_route_cache"])
        self.assertIn("version", body["planned_route_cache"])


class TheSourceSplitTests(_HandlerTest):
    """One test per side of the table, so a swap names the field it moved."""

    def test_the_cycle_half_ignores_a_later_live_mutation(self):
        """The bug this refactor exists to fix.

        ``/health`` and ``/status`` used to read ~15 live attributes one at a
        time.  A cycle completing between two of those reads produced a view
        that never existed.  Now the cycle half is one object, so a mutation
        after the publish cannot appear in a response at all.
        """
        runner = self._publish(inference_count=7, last_control_flag=1)
        published = runner.snapshot
        runner.inference_count = 999
        runner.last_control_flag = 0
        body = app.health()
        self.assertEqual(body["inference_count"], 7,
                         "inference_count is not being read from the snapshot")
        status = app.status()
        self.assertEqual(status["last_control_flag"], 1,
                         "last_control_flag is not being read from the snapshot")
        self.assertEqual(body["snapshot_sequence"], published.sequence)

    def test_the_live_half_tracks_its_own_object_without_a_publish(self):
        """Kept live on purpose: see this file's docstring."""
        runner = self._publish(inference_count=7)
        runner.cameras.window_builds = 5
        body = app.health()
        self.assertEqual(body["camera_stats"]["window_builds"], 5,
                         "camera_stats must stay a live read: check_chain.sh "
                         "derives each decoder's fps from two samples of it")
        # And the snapshot half did NOT move when the live half did.
        self.assertEqual(body["inference_count"], 7)

    def test_both_flags_can_disagree_and_both_are_reported(self):
        # last_sent_control_flag differs from last_control_flag only for the
        # release frame sent when the trim expired - and the two must be able to
        # come from the SAME cycle, which is the whole point of publishing them
        # together.
        self._publish(last_control_flag=1, last_sent_control_flag=0)
        body = app.status()
        self.assertEqual(body["last_control_flag"], 1)
        self.assertEqual(body["last_sent_control_flag"], 0)

    def test_pipeline_alive_is_the_live_call_not_the_snapshot(self):
        runner = self._publish(inference_count=7, model_loaded=True)
        published = runner.snapshot
        for alive in (True, False):
            with self.subTest(alive=alive):
                app._runner = _ReadyRunner(runner, alive=alive)
                body = app.health()
                self.assertIs(body["pipeline_alive"], alive)
                # The snapshot half is identical either way, which is what makes
                # this a test of the source and not of the value.
                self.assertEqual(body["snapshot_sequence"], published.sequence)
                self.assertEqual(body["inference_count"], 7)
                self.assertIs(body["model_loaded"], True)

    def test_a_never_started_pipeline_is_not_alive(self):
        self._publish(inference_count=1)
        self.assertIs(app.health()["pipeline_alive"], False)


class ReadyEndpointTests(_HandlerTest):
    """The status code and the body must never disagree."""

    def _ready(self) -> tuple[Response, dict]:
        response = Response()
        return response, app.ready(response)

    def test_a_healthy_pipeline_is_200_and_says_ready(self):
        runner = self._publish(model_loaded=True)
        now = time.monotonic()
        runner.started_at_monotonic = now - 30.0
        # A window 0.5 s ago and a pose 50 ms ago: inside both budgets, so the
        # only thing that could still fail is the stubbed liveness.
        runner._last_window_seen_at = now - 0.5
        runner.history.add(
            app.GlobalPose(
                timestamp_us=1_700_000_000_000_000,
                latitude=28.0, longitude=113.0, altitude=0.0,
                v_north=0.0, v_east=0.0, v_up=0.0,
                roll_deg=0.0, pitch_deg=0.0, azimuth_deg=180.0,
                arrival_us=time.monotonic_ns() // 1000 - 50_000,
                payload_timestamp_us=1_700_000_000_000_001,
            )
        )
        # The state is set on the runner after the first publish, so publish
        # again - then stub ONLY the thread liveness, which is the one live
        # input a test cannot honestly provide without starting a real loop.
        runner._publish_snapshot()
        app._runner = _ReadyRunner(runner)
        response, body = self._ready()
        self.assertEqual(body["failing"], [], body)
        self.assertTrue(body["ready"], body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["reason"], "")
        # ``warnings`` is deliberately NOT asserted to be empty here.  A runner
        # that was built but never started has an enabled route receiver that is
        # not listening, which is a warning - and a warning must not flip the
        # verdict.  Asserting on the verdict keeps this test independent of the
        # local config's route_tcp_enabled.
        self.assertIsNone(body["route_receiver_error"])

    def test_the_body_carries_the_evidence_alongside_the_verdict(self):
        response, body = self._ready()
        self.assertEqual(response.status_code, 503)
        for key in ("ready", "reason", "failing", "warnings", "checks",
                    "phase", "cycle_id", "snapshot_sequence", "snapshot_age_ms",
                    "uptime_s", "model_loaded", "pipeline_alive", "window_age_s",
                    "pose_age_ms", "unhealthy_readers", "route_receiver_listening",
                    "route_receiver_error"):
            with self.subTest(key=key):
                self.assertIn(key, body)

    def test_every_check_is_reported_with_a_reason(self):
        _, body = self._ready()
        names = [c["name"] for c in body["checks"]]
        self.assertEqual(
            names,
            [
                "not_stopping",
                "pipeline_alive",
                "pipeline_started",
                "startup_complete",
                "camera_window_fresh",
                "pose_fresh",
                "model_loaded",
                "time_sync_configured",
            ],
        )
        for check in body["checks"]:
            with self.subTest(check=check["name"]):
                if not check["ok"]:
                    self.assertTrue(
                        check["reason"],
                        f"{check['name']} failed without a reason, so the "
                        f"verdict cannot be acted on",
                    )

    def test_a_shutting_down_pipeline_is_503_whatever_else_is_true(self):
        runner = self._publish(phase=health.PHASE_STOPPING, model_loaded=True)
        app._runner = _ReadyRunner(runner, alive=True)
        response, body = self._ready()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(body["reason"], "shutting_down")


class _ReadyRunner:
    """A real runner with only its thread liveness stubbed.

    ``/ready`` needs ``is_alive()``; constructing a real pipeline thread here
    would need a camera provider and a model.  Everything else - the snapshot,
    the history, the decision engine - stays the real object.
    """

    def __init__(self, runner: app.PipelineRunner, *, alive: bool = True) -> None:
        self._runner = runner
        self._alive = alive

    def __getattr__(self, name):
        return getattr(self._runner, name)

    def is_alive(self) -> bool:
        return self._alive


if __name__ == "__main__":
    unittest.main()
