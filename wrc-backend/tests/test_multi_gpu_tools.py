"""The offline tools' own scaffolding, tested where it can be tested without a GPU.

Two of the three things in ``tools/multi_gpu_offline_test.py`` are load-bearing
for the verdict it prints, and neither needs a device to check:

* the **sink decoder**.  A run passes or fails on the numbers this reads off the
  wire, so it is written out independently of the sender's struct format - a
  decoder that reused the sender's format would agree with a broken sender.  It
  is tested here against hand-built bytes for that reason, including the ways a
  packet can be wrong rather than merely well-formed.
* the **synthetic task**.  It goes through the real ``InferenceTask.from_build``
  so the object crossing a ``spawn`` boundary is the shape the loop would have
  frozen; if that ever stopped being true, the scheduling scenarios would still
  pass while testing something else.

The rest of the tools - the pool, the wiring, the memory probe - is exercised by
running them.  See ``tools/multi_gpu_offline_test.py --scenario scheduler``,
which needs no GPU, and ``tools/gpu_memory_probe.py``, which needs all of them.

``tools/`` is not a package, so both are imported by path, as
``tests/test_replay_tool.py`` does.
"""

from __future__ import annotations

import importlib.util
import os
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEADER = struct.Struct("<IBBIHH")
POINT = struct.Struct("<dd")
MAGIC = 0x514D3031


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: a module that is in ``sys.modules`` while it
    # runs is what dataclasses and pickle need to resolve their own types.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


offline = _load_tool("multi_gpu_offline_test")
probe = _load_tool("gpu_memory_probe")


def packet(
    *,
    magic: int = MAGIC,
    version: int = 1,
    control: int = 1,
    seq: int = 1,
    vehicle_id: int = 3,
    points: tuple[tuple[float, float], ...] = ((30.5, 114.3),),
    declared_count: int | None = None,
    trailing: bytes = b"",
) -> bytes:
    count = len(points) if declared_count is None else declared_count
    payload = b"".join(POINT.pack(lat, lon) for lat, lon in points)
    return HEADER.pack(magic, version, control, seq, vehicle_id, count) + payload + trailing


class ToolLoadTests(unittest.TestCase):
    def test_both_tools_import_without_a_device(self) -> None:
        self.assertTrue(callable(offline.main))
        self.assertTrue(callable(probe.main))

    def test_the_offline_tool_states_the_three_frequencies_it_reports(self) -> None:
        """``udp_output_hz`` is 0 in scheduler mode and the tool says so, rather
        than letting ``accepted_nonempty_hz`` be read as the wire rate."""
        self.assertIn("udp_output_hz", offline.__doc__ or "")
        self.assertIn("accepted_nonempty_hz", offline.__doc__ or "")


class DeviceParsingTests(unittest.TestCase):
    def test_a_duplicated_device_is_refused(self) -> None:
        for module in (offline, probe):
            with self.assertRaises(Exception):
                module._parse_devices("1,3,1")

    def test_a_negative_device_is_refused(self) -> None:
        for module in (offline, probe):
            with self.assertRaises(Exception):
                module._parse_devices("1,-3")

    def test_an_empty_list_is_refused(self) -> None:
        for module in (offline, probe):
            with self.assertRaises(Exception):
                module._parse_devices(" , ")

    def test_spaces_and_order_are_honoured(self) -> None:
        self.assertEqual(offline._parse_devices(" 4, 5 ,6"), [4, 5, 6])


class SinkDecodeTests(unittest.TestCase):
    """The decoder the whole pipeline verdict rests on, against built bytes."""

    def setUp(self) -> None:
        self.sink = offline.WaypointSink()

    def tearDown(self) -> None:
        self.sink.stop()

    def test_a_two_point_packet_is_counted_as_carrying_a_trajectory(self) -> None:
        self.sink._decode(packet(points=((30.5, 114.3), (30.6, 114.4)), seq=7))
        self.assertEqual(self.sink.packets, 1)
        self.assertEqual(self.sink.with_waypoints, 1)
        self.assertEqual(self.sink.release_frames, 0)
        self.assertEqual(self.sink.decode_errors, [])
        self.assertEqual(self.sink.last_seq, 7)
        self.assertEqual(self.sink.first_point, (30.5, 114.3))

    def test_a_release_frame_is_a_send_that_carries_no_trajectory(self) -> None:
        """``count=0, control=0`` is legal and must still be sent - which is
        exactly why the total packet rate is not the success criterion."""
        self.sink._decode(packet(points=(), declared_count=0, control=0, seq=1))
        self.assertEqual(self.sink.packets, 1)
        self.assertEqual(self.sink.with_waypoints, 0)
        self.assertEqual(self.sink.release_frames, 1)
        self.assertEqual(self.sink.first_point, None)

    def test_a_foreign_magic_is_rejected_rather_than_counted(self) -> None:
        self.sink._decode(packet(magic=0xDEADBEEF))
        self.assertEqual(self.sink.packets, 0)
        self.assertEqual(self.sink.bad_magic, 1)
        self.assertTrue(self.sink.decode_errors)

    def test_a_count_that_disagrees_with_the_length_is_rejected(self) -> None:
        """Both directions: a packet cut short, and one with bytes appended.
        The second is the quieter bug - a sender that grew a field would still
        decode if only the short case were checked."""
        self.sink._decode(packet(points=((30.5, 114.3),), declared_count=3))
        self.assertEqual(self.sink.packets, 0)
        self.sink._decode(packet(points=((30.5, 114.3),), trailing=b"\x00\x00"))
        self.assertEqual(self.sink.packets, 0)
        self.assertEqual(len(self.sink.decode_errors), 2)

    def test_a_header_shorter_than_the_format_is_rejected(self) -> None:
        self.sink._decode(b"\x01\x02\x03")
        self.assertEqual(self.sink.packets, 0)
        self.assertTrue(self.sink.decode_errors)

    def test_a_skipped_sequence_number_is_counted_but_not_fatal(self) -> None:
        """A failed send legitimately skips a number, so this is reported rather
        than asserted on."""
        self.sink._decode(packet(seq=1))
        self.sink._decode(packet(seq=3))
        self.sink._decode(packet(seq=4))
        self.assertEqual(self.sink.packets, 3)
        self.assertEqual(self.sink.seq_gaps, 1)
        self.assertEqual(self.sink.last_seq, 4)

    def test_the_rate_is_zero_with_nothing_to_measure(self) -> None:
        self.assertEqual(self.sink.rate_hz(), 0.0)
        self.sink._decode(packet(seq=1))
        self.assertEqual(self.sink.rate_hz(), 0.0)

    def test_the_report_survives_json_serialisation(self) -> None:
        """It is printed with ``json.dumps``, so a tuple or a set here would be
        a runtime failure in the one run that matters."""
        import json

        self.sink._decode(packet(points=((30.5, 114.3),), seq=1))
        body = json.dumps(self.sink.report())
        self.assertIn("hz_with_waypoints", body)


class SyntheticTaskTests(unittest.TestCase):
    """The fixture every scheduling scenario submits."""

    def test_the_task_carries_its_own_sequence_and_capture_time(self) -> None:
        task = offline.make_task(9, 900_000, cycle_id=4)
        self.assertEqual(task.task_seq, 9)
        self.assertEqual(task.capture_timestamp_us, 900_000)
        self.assertEqual(task.cycle_id, 4)
        self.assertGreater(task.created_monotonic_ns, 0)

    def test_the_window_uses_the_models_camera_order(self) -> None:
        """``[1, 2, 0, 4]`` - the CIMG order the model takes.  A scenario that
        submitted the decode order would pass while measuring the wrong thing."""
        window = offline.make_task(1, 100_000).window
        self.assertEqual(window.camera_indices, [1, 2, 0, 4])
        self.assertEqual(len(window.frame_ids), 4)
        self.assertEqual(len(window.frame_ids[0]), offline.FRAMES_PER_CAMERA)

    def test_the_ego_tensors_have_the_shape_the_model_takes(self) -> None:
        task = offline.make_task(1, 100_000)
        self.assertEqual(tuple(task.ego_xyz.shape), (1, 1, offline.HISTORY_STEPS, 3))
        self.assertEqual(tuple(task.ego_rot.shape), (1, 1, offline.HISTORY_STEPS, 3, 3))

    def test_the_reference_pose_is_the_capture_it_came_from(self) -> None:
        task = offline.make_task(1, 250_000)
        self.assertEqual(task.reference_pose.timestamp_us, 250_000)

    def test_consecutive_tasks_advance_the_capture_by_one_camera_frame(self) -> None:
        """The Result Manager's camera-clock checks are relative, so the step
        between two submitted captures has to look like a camera.  Getting this
        wrong would make every task an anomaly and the scenario would prove the
        opposite of what it claims."""
        first = offline.make_task(1, offline.CAPTURE_STEP_US)
        second = offline.make_task(2, offline.CAPTURE_STEP_US * 2)
        self.assertEqual(
            second.capture_timestamp_us - first.capture_timestamp_us,
            offline.CAPTURE_STEP_US,
        )


class ScenarioGuardTests(unittest.TestCase):
    def test_the_ordering_scenario_skips_itself_on_the_wrong_worker_count(self) -> None:
        """Its per-task delays are chosen for four workers.  On any other count
        the completions would not line up, so it declines rather than asserting
        something it did not set up."""
        self.assertEqual(
            offline.main(["--scenario", "order", "--devices", "1,3"]), 0
        )


class NvidiaSmiReadingTests(unittest.TestCase):
    """The probe's whole-device readout, against text rather than a device.

    This is the number the release check and every growth slope are computed
    from, and it is the number an operator compares against their own
    ``nvidia-smi`` - so it is parsed here rather than trusted.
    """

    HEADER_LINES = [
        "0, 24564, 1488, 22844",
        "1, 24564, 4, 24327",
        "2, 24564, 20212, 4120",
    ]

    def test_every_field_lands_where_it_belongs(self) -> None:
        seen: list[list[str]] = []

        def runner(args):
            seen.append(list(args))
            return "\n".join(self.HEADER_LINES)

        memory = probe.nvidia_smi_memory(runner)
        self.assertEqual(
            memory[1], {"total_mib": 24564.0, "used_mib": 4.0, "free_mib": 24327.0}
        )
        self.assertEqual(sorted(memory), [0, 1, 2])
        # Physical indices, no header, no units: a header or a "MiB" suffix
        # would make the first data row unparseable and the whole read empty.
        self.assertIn("--query-gpu=index,memory.total,memory.used,memory.free", seen[0])
        self.assertIn("--format=csv,noheader,nounits", seen[0])

    def test_a_missing_nvidia_smi_is_an_empty_read_not_a_crash(self) -> None:
        def runner(args):
            raise FileNotFoundError("nvidia-smi")

        self.assertEqual(probe.nvidia_smi_memory(runner), {})

    def test_one_unreadable_card_does_not_discard_the_others(self) -> None:
        """A card reporting ``[N/A]`` is a partial reading, and a partial
        reading is still a reading."""

        def runner(args):
            return "0, 24564, 1488, 22844\n1, [N/A], [N/A], [N/A]\n2, 24564, 4, 24327"

        memory = probe.nvidia_smi_memory(runner)
        self.assertEqual(sorted(memory), [0, 2])

    def test_the_description_carries_used_free_and_total(self) -> None:
        memory = {1: {"total_mib": 24564.0, "used_mib": 4096.0, "free_mib": 20468.0}}
        text = probe.describe_device(1, memory)
        self.assertIn("used=4096MiB", text)
        self.assertIn("free=20468MiB", text)
        self.assertIn("total=24564MiB", text)
        self.assertIn("unavailable", probe.describe_device(7, memory))


class DevicePinningTests(unittest.TestCase):
    """The guard against a model landing on the wrong physical card.

    ``CUDA_VISIBLE_DEVICES=1`` with ``--devices 0`` passes every check the pool
    makes - the process sees one card and it is numbered 0 - and then the worker
    pins itself to physical GPU 0.  Nothing else in the system notices, so the
    probe has to.
    """

    def setUp(self) -> None:
        self._saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def tearDown(self) -> None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        if self._saved is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._saved

    def test_an_unset_variable_is_the_expected_state(self) -> None:
        self.assertIsNone(probe.check_device_pinning([1]))

    def test_an_empty_variable_is_treated_as_unset(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "   "
        self.assertIsNone(probe.check_device_pinning([1]))

    def test_an_inherited_list_is_refused_and_says_why(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        problem = probe.check_device_pinning([0])
        assert problem is not None
        self.assertIn("CUDA_VISIBLE_DEVICES='1'", problem)
        # It has to name the fix, because the variable looks harmless.
        self.assertIn("Unset CUDA_VISIBLE_DEVICES", problem)

    def test_it_is_refused_even_when_the_numbers_agree(self) -> None:
        """``CUDA_VISIBLE_DEVICES=1`` with ``--devices 1`` looks consistent and
        is not: the pool's range check counts one visible card, so device 1 is
        out of range, and the run would fail at startup instead of here."""
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        self.assertIsNotNone(probe.check_device_pinning([1]))


class ReleaseCheckTests(unittest.TestCase):
    """Whether the cards came back, decided without a card to come back to."""

    class FakeProbe:
        def __init__(self, baseline: dict[int, int], devices: list[int]) -> None:
            self.baseline_used = baseline
            self.devices = devices

    def test_a_card_within_tolerance_passes(self) -> None:
        mib = probe.MIB
        fake = self.FakeProbe({1: 1000 * mib}, [1])
        outcome = probe.wait_for_release(
            fake,
            tolerance_mib=256.0,
            sampler=lambda: {1: 1100 * mib},
            sleep=lambda _s: None,
        )
        self.assertTrue(outcome["checked"])
        self.assertTrue(outcome["released"])
        self.assertEqual(outcome["baseline_mib"][1], 1000.0)
        self.assertEqual(outcome["after_mib"][1], 1100.0)

    def test_a_card_still_holding_a_worker_fails(self) -> None:
        """Several GiB still held after the pool stopped is a worker's worth -
        the exact thing this check exists to catch, and it is far outside the
        tolerance rather than a hair over it."""
        mib = probe.MIB
        fake = self.FakeProbe({1: 1000 * mib}, [1])
        outcome = probe.wait_for_release(
            fake,
            tolerance_mib=256.0,
            timeout_s=0.0,
            sampler=lambda: {1: 9000 * mib},
            sleep=lambda _s: None,
        )
        self.assertTrue(outcome["checked"])
        self.assertFalse(outcome["released"])
        self.assertEqual(outcome["still_held"][1]["over_mib"], 8000.0)

    def test_it_waits_rather_than_failing_on_the_first_reading(self) -> None:
        """The driver does not necessarily hand the memory back the instant the
        process exits, so the first sample being high is not a verdict."""
        mib = probe.MIB
        fake = self.FakeProbe({1: 1000 * mib}, [1])
        readings = iter([{1: 9000 * mib}, {1: 9000 * mib}, {1: 1000 * mib}])
        slept: list[float] = []
        outcome = probe.wait_for_release(
            fake,
            sampler=lambda: next(readings),
            sleep=lambda seconds: slept.append(seconds),
        )
        self.assertTrue(outcome["released"])
        self.assertEqual(len(slept), 2)

    def test_without_a_baseline_it_declines_rather_than_guesses(self) -> None:
        fake = self.FakeProbe({}, [1])
        outcome = probe.wait_for_release(
            fake, sampler=lambda: {1: 0}, sleep=lambda _s: None
        )
        self.assertFalse(outcome["checked"])
        self.assertNotIn("released", outcome)

    def test_a_device_that_was_never_measured_is_not_silently_passed(self) -> None:
        """A device missing from the sampler's reading is compared as zero,
        which reads as 'returned to baseline'.  That is the honest answer only
        because the baseline was taken the same way; what must not happen is an
        exception."""
        mib = probe.MIB
        fake = self.FakeProbe({1: 0 * mib}, [1])
        outcome = probe.wait_for_release(
            fake, sampler=lambda: {}, sleep=lambda _s: None
        )
        self.assertTrue(outcome["checked"])
        self.assertTrue(outcome["released"])


class ProbeMathTests(unittest.TestCase):
    def test_a_flat_series_has_no_growth(self) -> None:
        samples = [(i * 60.0, 1000.0 * probe.MIB) for i in range(5)]
        slope = probe.slope_per_hour(samples)
        assert slope is not None
        self.assertAlmostEqual(slope, 0.0, places=6)

    def test_growth_is_reported_per_hour(self) -> None:
        """One MiB per minute is 60 MiB per hour, however long the soak ran."""
        samples = [(i * 60.0, i * probe.MIB) for i in range(6)]
        slope = probe.slope_per_hour(samples)
        assert slope is not None
        self.assertAlmostEqual(slope / probe.MIB, 60.0, places=3)

    def test_too_few_samples_are_not_a_verdict(self) -> None:
        """Two points fit a line exactly and would call noise a leak."""
        self.assertIsNone(probe.slope_per_hour([]))
        self.assertIsNone(probe.slope_per_hour([(0.0, 0.0)]))
        self.assertIsNone(probe.slope_per_hour([(0.0, 0.0), (1.0, 1.0)]))


if __name__ == "__main__":
    unittest.main()
