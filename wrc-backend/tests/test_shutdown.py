"""Shutdown is bounded, ordered, and reports what it could not stop.

Why this file exists at all: before Stage 3 ``PipelineRunner.stop()`` set two
events and returned.  It did not wait for the loop and did not close the sender
socket, so "shutdown" and "the process is about to die" were the same statement.

So the assertions here are about *ordering and completeness*, not about tidiness:

* the CIMG reader is flagged BEFORE the pipeline drain, and joined on the reap
  budget so the next start can bind ``network.cimg``;
* the drain has its own budget, so the reap can never be starved by it;
* a component that did not stop is named in ``failed`` rather than only being
  visible as the next instance failing to bind;
* the sender socket is closed only once the loop is confirmed stopped, because
  a late ``send()`` on a closed socket surfaces as "pipeline cycle failed".

One environment note: ``pipeline`` imports cv2.  ``cv2_stub`` installs a stub
for the duration of the imports and then tears it down completely, so a later
``import cv2`` in another test module still fails exactly as it did before -
without that, this file would quietly change how a later ``import camera.sync``
fails (it reaches a real import and fails differently when cv2 is a stub).

Note the word *completely*.  Removing the ``sys.modules`` entry alone is not
enough, and an earlier version of this file did exactly that while claiming the
weaker property in a comment: ``pipeline`` and ``camera.sync`` bind cv2 to
their own globals at import time, so the stub stayed reachable from them and
a later ``import camera.sync`` failed with ``AttributeError: module 'cv2' has
no attribute 'resize'`` instead of ``ModuleNotFoundError``.  See
tests/_cv2_isolation.py.

Every cv2-dependent import belongs inside the block.  ``camera.sync`` is used by
``ComponentLevelTests`` and is imported here rather than inside the test for
that reason - a test-time import would re-execute the module after the stub is
gone and raise ``ModuleNotFoundError``.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

from _cv2_isolation import cv2_stub

# ``app -> pipeline -> camera.sync -> cv2``.  Stub, import, un-stub.
with cv2_stub():
    import app
    import camera.sync
    import pipeline

import health  # noqa: E402


class _Recorder:
    """A shared, ordered log of every call shutdown makes."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def note(self, event: str) -> None:
        self.events.append(event)

    def order(self, event: str) -> int:
        """Index of the first occurrence, or a large number if never called."""
        return self.events.index(event) if event in self.events else 10 ** 6


class _FakeCameras:
    def __init__(self, recorder: _Recorder, *, alive: tuple[str, ...] = ()) -> None:
        self._recorder = recorder
        self._alive = alive
        self.terminated = 0
        self.joins = 0

    # -- lifecycle surface ------------------------------------------------ #
    def start(self) -> None:
        self._recorder.note("cameras.start")

    # -- shutdown surface ------------------------------------------------- #
    def stop(self) -> None:
        self._recorder.note("cameras.stop")

    def join(self, timeout_s: float = 0.0) -> None:
        self.joins += 1
        self._recorder.note("cameras.join")

    def alive_readers(self) -> tuple[str, ...]:
        return self._alive

    # -- what _publish_snapshot reads ------------------------------------- #
    def unhealthy_readers(self) -> tuple[str, ...]:
        return ()


class _FakeSender:
    """Mirrors ``WaypointSender``: ``close()`` is idempotent.

    ``closes`` counts *calls* and ``closed`` records the state, so a test can
    tell "closed once, asked twice" (correct) from "closed twice" - which the
    real object's own guard prevents, and which a fake without the guard would
    happily allow.
    """

    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder
        self.closes = 0
        self.closed = False

    def close(self) -> None:
        self.closes += 1
        if self.closed:
            return
        self.closed = True
        self._recorder.note("sender.close")


class _FakeThread:
    """Shutdown surface of a started child, with the calls it should receive."""

    def __init__(
        self,
        recorder: _Recorder,
        name: str,
        *,
        raises: bool = False,
        ident: int | None = 1,
    ) -> None:
        self._recorder = recorder
        self._name = name
        self._raises = raises
        self.ident = ident
        self.stopped = 0
        self.joined = 0
        # ``_publish_snapshot`` reports the receiver's state every cycle.
        self.listening = False
        self.last_error: str | None = None

    def start(self) -> None:
        self.ident = 1
        self._recorder.note(f"{self._name}.start")

    def stop(self) -> None:
        self.stopped += 1
        self._recorder.note(f"{self._name}.stop")
        if self._raises:
            raise RuntimeError(f"{self._name} will not stop")

    def join(self, timeout: float | None = None) -> None:
        self.joined += 1
        self._recorder.note(f"{self._name}.join")


class _WedgedRunner(pipeline.PipelineRunner):
    """A loop that ignores the stop flag, to exercise the drain deadline."""

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.release = threading.Event()

    def run(self) -> None:  # noqa: D102 - deliberately not the real loop
        self.release.wait(30.0)


def _runner(cls=pipeline.PipelineRunner) -> pipeline.PipelineRunner:
    """Build a runner.  These tests replace the cameras after construction."""
    return cls(app._settings)


def _inject(
    runner: pipeline.PipelineRunner,
    recorder: _Recorder,
    *,
    alive: tuple[str, ...] = (),
    pose_raises: bool = False,
    pose_ident: int | None = 1,
    mock: bool = False,
    route: bool = True,
) -> dict[str, object]:
    cameras = _FakeCameras(recorder, alive=alive)
    sender = _FakeSender(recorder)
    # Release the real socket before replacing the object that owns it, so the
    # test does not leak a descriptor per construction (the very leak Stage 3
    # fixed by giving WaypointSender a close()).
    runner.sender.close()
    runner.cameras = cameras          # type: ignore[assignment]
    runner.sender = sender            # type: ignore[assignment]
    parts: dict[str, object] = {"cameras": cameras, "sender": sender}
    if route:
        parts["route"] = runner.route_tcp_receiver
        runner.route_tcp_receiver = _FakeThread(recorder, "route")  # type: ignore[assignment]
    if mock:
        runner.mock_pose = _FakeThread(recorder, "mock_pose")  # type: ignore[assignment]
    else:
        runner.pose_listener = _FakeThread(  # type: ignore[assignment]
            recorder, "pose_listener", raises=pose_raises, ident=pose_ident
        )
    return parts


class CleanShutdownTests(unittest.TestCase):
    """A stop with nothing wrong reports nothing wrong."""

    def setUp(self) -> None:
        self.recorder = _Recorder()
        self.runner = _runner()
        self.parts = _inject(self.runner, self.recorder)

    def tearDown(self) -> None:
        self.runner.stop(drain_s=0.1, reap_s=0.1)

    def test_nothing_is_reported_as_failed(self):
        report = self.runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertEqual(report.failed, ())
        self.assertFalse(report.budget_exceeded)

    def test_every_component_it_touched_is_accounted_for(self):
        report = self.runner.stop(drain_s=0.5, reap_s=0.5)
        for name in ("pipeline", "route_tcp_receiver", "pose_listener", "cameras"):
            with self.subTest(component=name):
                self.assertIn(name, report.components)
                self.assertIn(name, report.stopped)

    def test_the_report_serializes_for_the_log_line(self):
        report = self.runner.stop(drain_s=0.5, reap_s=0.5)
        body = report.as_dict()
        self.assertEqual(body["failed"], [])
        self.assertIn("elapsed_ms", body)
        self.assertIn("budget_exceeded", body)

    def test_the_sender_socket_is_released(self):
        sender = self.parts["sender"]
        self.runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertTrue(sender.closed)

    def test_the_phase_ends_at_stopped(self):
        self.runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertEqual(self.runner.snapshot.phase, health.PHASE_STOPPED)


class ShutdownOrderTests(unittest.TestCase):
    """The order is the safety property, so it is asserted, not assumed."""

    def setUp(self) -> None:
        self.recorder = _Recorder()
        self.runner = _runner()
        _inject(self.runner, self.recorder)

    def tearDown(self) -> None:
        self.runner.stop(drain_s=0.1, reap_s=0.1)

    def test_flags_are_set_before_any_component_is_waited_on(self):
        self.runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertLess(
            self.recorder.order("cameras.stop"),
            self.recorder.order("cameras.join"),
        )

    def test_the_other_children_are_stopped_before_the_cameras_are_reaped(self):
        self.runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertLess(
            self.recorder.order("pose_listener.stop"),
            self.recorder.order("cameras.join"),
        )
        self.assertLess(
            self.recorder.order("route.stop"),
            self.recorder.order("cameras.join"),
        )

    def test_the_sender_is_closed_after_the_drain_not_during_it(self):
        # A late send() on a closed socket raises OSError, which the loop's own
        # except Exception logs as "pipeline cycle failed" - so the close has to
        # come after the loop is confirmed stopped.
        self.runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertGreater(
            self.recorder.order("sender.close"),
            self.recorder.order("cameras.join"),
        )


class UncooperativeComponentTests(unittest.TestCase):
    """A component that will not stop is named, not swallowed."""

    def tearDown(self) -> None:
        for runner in getattr(self, "runners", []):
            runner.stop(drain_s=0.1, reap_s=0.1)

    def test_a_child_that_raises_on_stop_is_named(self):
        recorder = _Recorder()
        runner = _runner()
        self.runners = [runner]
        _inject(runner, recorder, pose_raises=True)
        with self.assertLogs("app", level="ERROR"):
            report = runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertIn("pose_listener", report.failed)
        self.assertNotIn("pose_listener", report.stopped)
        # Still in components: "tried and failed" must be distinguishable from
        # "never reached".
        self.assertIn("pose_listener", report.components)

    def test_the_rest_of_the_shutdown_still_happens(self):
        recorder = _Recorder()
        runner = _runner()
        self.runners = [runner]
        parts = _inject(runner, recorder, pose_raises=True)
        with self.assertLogs("app", level="ERROR"):
            report = runner.stop(drain_s=0.5, reap_s=0.5)
        # One child failing must not abandon the others.
        self.assertIn("cameras", report.stopped)
        self.assertEqual(parts["cameras"].joins, 1)
        self.assertEqual(parts["cameras"].terminated, 0)

    def test_a_reader_still_running_after_the_reap_is_named(self):
        recorder = _Recorder()
        runner = _runner()
        self.runners = [runner]
        _inject(runner, recorder, alive=("udp://127.0.0.1:31101",))
        with self.assertLogs("app", level="ERROR") as captured:
            report = runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertIn("cameras", report.failed)
        self.assertNotIn("cameras", report.stopped)
        self.assertIn(
            "udp://127.0.0.1:31101",
            "\n".join(captured.output),
            "the log line must name which reader survived, not just that one did",
        )

    def test_a_pipeline_that_ignores_the_deadline_is_named(self):
        recorder = _Recorder()
        runner = _runner(_WedgedRunner)
        self.runners = [runner]
        _inject(runner, recorder)
        runner.start()
        try:
            report = runner.stop(drain_s=0.05, reap_s=0.5)
        finally:
            runner.release.set()
            runner.join(5.0)
        self.assertIn("pipeline", report.failed)
        self.assertNotIn("pipeline", report.stopped)

    def test_the_sender_is_left_open_when_the_drain_did_not_confirm(self):
        # Closing it would be the very hazard the ordering exists to prevent.
        recorder = _Recorder()
        runner = _runner(_WedgedRunner)
        self.runners = [runner]
        parts = _inject(runner, recorder)
        runner.start()
        try:
            runner.stop(drain_s=0.05, reap_s=0.5)
        finally:
            runner.release.set()
            runner.join(5.0)
        self.assertFalse(
            parts["sender"].closed,
            "a sender closed while the loop may still be sending turns a late "
            "send into a 'pipeline cycle failed' traceback",
        )

    def test_the_reap_budget_is_separate_from_the_drain(self):
        # Two sub-budgets, not one pool: the reap joins the CIMG reader so the
        # next start can bind its port, and an overrunning drain must not be
        # able to consume that budget.
        recorder = _Recorder()
        runner = _runner(_WedgedRunner)
        self.runners = [runner]
        parts = _inject(runner, recorder)
        runner.start()
        try:
            report = runner.stop(drain_s=0.05, reap_s=0.5)
        finally:
            runner.release.set()
            runner.join(5.0)
        self.assertEqual(
            parts["cameras"].joins, 1,
            "the cameras were never reaped because the drain used the budget",
        )
        self.assertEqual(parts["cameras"].terminated, 0)
        self.assertIn("cameras", report.stopped)

    def test_the_report_stays_within_its_stated_budget(self):
        recorder = _Recorder()
        runner = _runner(_WedgedRunner)
        self.runners = [runner]
        _inject(runner, recorder)
        runner.start()
        try:
            report = runner.stop(drain_s=0.05, reap_s=0.5)
        finally:
            runner.release.set()
            runner.join(5.0)
        self.assertLess(report.elapsed_ms, 2000.0)
        self.assertFalse(report.budget_exceeded)


class NeverStartedTests(unittest.TestCase):
    """A runner that never started is stopped, not reported as broken.

    ``app._startup`` assigns ``_runner`` *before* it calls ``start()``, so a
    failure in between leaves an unstarted runner for ``app._shutdown``.  Every
    component then has to handle "there is no thread" without raising:
    ``Thread.join()`` raises ``RuntimeError`` for an unstarted thread, which
    would turn a startup failure into a bogus "did not stop" report.
    """

    def test_stopping_an_unstarted_runner_reports_success(self):
        runner = _runner()
        self.assertIsNone(runner.ident)
        logging.disable(logging.ERROR)
        try:
            report = runner.stop(drain_s=0.2, reap_s=0.2)
        finally:
            logging.disable(logging.NOTSET)
        self.assertEqual(report.failed, ())
        self.assertIn("pipeline", report.stopped)
        self.assertEqual(runner.snapshot.phase, health.PHASE_STOPPED)

    def test_the_unstarted_route_receiver_is_not_an_error(self):
        runner = _runner()
        if runner.route_tcp_receiver is None:
            self.skipTest("route receiver disabled in this configuration")
        logging.disable(logging.ERROR)
        try:
            report = runner.stop(drain_s=0.2, reap_s=0.2)
        finally:
            logging.disable(logging.NOTSET)
        self.assertNotIn("route_tcp_receiver", report.failed)


class IdempotenceTests(unittest.TestCase):
    def test_a_second_stop_changes_nothing(self):
        recorder = _Recorder()
        runner = _runner()
        _inject(runner, recorder)
        first = runner.stop(drain_s=0.2, reap_s=0.2)
        second = runner.stop(drain_s=0.2, reap_s=0.2)
        self.assertEqual(first.failed, ())
        self.assertEqual(second.failed, ())
        self.assertEqual(runner.snapshot.phase, health.PHASE_STOPPED)

    def test_the_real_sender_close_is_idempotent(self):
        # Pinned on WaypointSender itself, not on the fake above: the second
        # close is what a double shutdown would do, and it must not raise.
        from waypoint.sender import WaypointSender

        sender = WaypointSender(app._settings)
        self.assertFalse(sender.closed)
        sender.close()
        self.assertTrue(sender.closed)
        sender.close()          # must not raise
        self.assertTrue(sender.closed)

    def test_the_pre_stage_3_call_convention_still_works(self):
        # Every existing call site ignores the return value.  A caller doing
        # ``runner.stop()`` must keep working, with the module defaults.
        runner = _runner()
        _inject(runner, _Recorder())
        runner.stop()
        self.assertEqual(runner.snapshot.phase, health.PHASE_STOPPED)

    def test_the_defaults_come_from_the_health_module(self):
        runner = _runner()
        _inject(runner, _Recorder())
        report = runner.stop()
        self.assertEqual(report.drain_s, health.DRAIN_S)
        self.assertEqual(report.reap_s, health.REAP_S)


class ComponentLevelTests(unittest.TestCase):
    """The components' own stop paths, without a runner in the way."""

    def test_a_never_started_route_receiver_stops_without_raising(self):
        import vehicle.planned_route

        receiver = vehicle.planned_route.PlannedRouteTcpReceiver(None, "127.0.0.1", 0)
        self.assertIsNone(receiver.ident)
        receiver.stop()          # raised RuntimeError before Stage 3
        self.assertFalse(receiver.listening)

    def test_the_route_receiver_join_timeout_is_still_two_seconds(self):
        # tools/receive_route_test.py and tests/test_planned_route_tcp.py both
        # depend on this method returning inside that bound.
        import inspect
        import vehicle.planned_route

        source = inspect.getsource(vehicle.planned_route.PlannedRouteTcpReceiver.stop)
        self.assertIn("timeout=2.0", source)

    def test_a_reader_that_never_started_is_not_joined(self):
        class _NeverStarted(threading.Thread):
            def run(self) -> None:  # pragma: no cover - must never run
                raise AssertionError("started a thread that should not run")

        provider = camera.sync.CameraWindowProvider.__new__(
            camera.sync.CameraWindowProvider
        )
        provider.readers = [_NeverStarted(name="never-started")]
        provider._stats_thread = None
        provider.join(0.1)       # logged an ERROR traceback before Stage 3
        self.assertFalse(provider.readers[0].is_alive())


class StoppedIsAStatementAboutTheProcessTests(unittest.TestCase):
    """``PHASE_STOPPED`` means the loop is gone, not that stop() ran.

    This is the difference between a shutdown that reports what it achieved and
    one that reports what it attempted.  ``/ready`` reads the phase as evidence:
    check 1 (``not_stopping``) passes for anything other than "stopping" or
    "stopped", so publishing "stopped" over a still-running loop would make a
    process that is mid-cycle answer 200 on the one endpoint whose job is to say
    "do not send me traffic".
    """

    def tearDown(self) -> None:
        for runner in getattr(self, "runners", []):
            runner.stop(drain_s=0.1, reap_s=0.1)

    def _wedged(self):
        runner = _runner(_WedgedRunner)
        self.runners = [runner]
        _inject(runner, _Recorder())
        runner.start()
        return runner

    def test_a_drain_that_overran_does_not_publish_stopped(self):
        runner = self._wedged()
        try:
            report = runner.stop(drain_s=0.05, reap_s=0.5)
            self.assertIn("pipeline", report.failed)
            self.assertEqual(
                runner.snapshot.phase, health.PHASE_STOPPING,
                "the loop is still executing, so the published phase must stay "
                "where the truth is - 'stopped' is a statement about the thread",
            )
        finally:
            runner.release.set()
            runner.join(5.0)
            # Let the finally-released loop exit and the runner reach stopped,
            # so tearDown is a no-op rather than a second failed drain.
            runner.stop(drain_s=2.0, reap_s=0.5)
        self.assertEqual(runner.snapshot.phase, health.PHASE_STOPPED)

    def test_a_concluded_drain_still_publishes_stopped(self):
        # The other half of the same rule: the new branch must not have made
        # "stopped" unreachable for the ordinary case.
        runner = _runner()
        _inject(runner, _Recorder())
        runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertEqual(runner.snapshot.phase, health.PHASE_STOPPED)

    def test_a_failed_drain_leaves_ready_answering_503(self):
        # The phase is not decoration: it is what /ready acts on.  Driven
        # through the real handler, with the real wedged runner behind it.
        from fastapi import Response

        runner = self._wedged()
        try:
            report = runner.stop(drain_s=0.05, reap_s=0.5)
            self.assertIn("pipeline", report.failed)
            saved = app._runner
            app._runner = runner
            try:
                response = Response()
                body = app.ready(response)
            finally:
                app._runner = saved
            self.assertEqual(response.status_code, 503)
            self.assertEqual(body["reason"], "shutting_down")
        finally:
            runner.release.set()
            runner.join(5.0)
            runner.stop(drain_s=2.0, reap_s=0.5)

    def test_a_clean_stop_still_leaves_the_runner_cleared(self):
        saved = app._runner
        runner = _runner()
        _inject(runner, _Recorder())
        app._runner = runner
        try:
            app._shutdown()
            self.assertIsNone(app._runner)
        finally:
            app._runner = saved

    def test_a_failed_stop_keeps_the_runner_so_the_failure_stays_visible(self):
        # The whole reason the runner is not cleared on failure: it *is* the
        # observable state.  Dropping it leaves /health and /status answering
        # from the empty snapshot, indistinguishable from a process that never
        # started, and the operator loses the only handle on the component that
        # did not stop.
        saved = app._runner
        runner = self._wedged()
        app._runner = runner
        try:
            with self.assertLogs("app", level="ERROR") as captured:
                app._shutdown()
            self.assertIs(
                app._runner, runner,
                "_runner was cleared for a shutdown that did not complete",
            )
            self.assertIn(
                "pipeline stop incomplete", "\n".join(captured.output),
                "the incomplete shutdown must name itself in the log",
            )
        finally:
            runner.release.set()
            runner.join(5.0)
            runner.stop(drain_s=2.0, reap_s=0.5)
            app._runner = saved


class EveryChildIsJoinedOnlyIfItStartedTests(unittest.TestCase):
    """A thread that never ran is not joined, and a started one is.

    ``Thread.join()`` raises for an unstarted thread, and shutdown reaches that
    state whenever startup failed part-way.  Both directions are asserted: the
    guard must not become an excuse to skip the join for a thread that *is*
    running - that is the leak the whole bounded-reap design exists to close.
    """

    def tearDown(self) -> None:
        for runner in getattr(self, "runners", []):
            runner.stop(drain_s=0.1, reap_s=0.1)

    def test_a_started_mock_pose_provider_is_joined(self):
        recorder = _Recorder()
        runner = _runner()
        self.runners = [runner]
        parts = _inject(runner, recorder, mock=True)
        self.assertIsNotNone(parts["cameras"])
        runner.mock_pose.ident = 1            # started
        runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertEqual(runner.mock_pose.joined, 1)

    def test_an_unstarted_mock_pose_provider_is_not_joined(self):
        recorder = _Recorder()
        runner = _runner()
        self.runners = [runner]
        _inject(runner, recorder, mock=True)
        runner.mock_pose.ident = None         # never started
        report = runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertEqual(
            runner.mock_pose.joined, 0,
            "joining an unstarted thread raises RuntimeError, which would be "
            "reported as a component that failed to stop",
        )
        # Counted as stopped all the same: it is not running, and an ERROR
        # about it would be wrong.
        self.assertIn("mock_pose", report.stopped)

    def test_an_unstarted_pose_listener_is_not_joined(self):
        runner = _runner()
        self.runners = [runner]
        _inject(runner, _Recorder(), pose_ident=None)
        runner.pose_listener.ident = None
        runner.stop(drain_s=0.5, reap_s=0.5)
        self.assertEqual(runner.pose_listener.joined, 0)


if __name__ == "__main__":
    unittest.main()
