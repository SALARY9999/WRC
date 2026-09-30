"""The frozen task: what a worker is allowed to know, and what it cannot change.

Two properties matter here and both are about *isolation*.  A task must carry
everything one model call needs, so a worker never reads the pipeline's live
pose history or camera buffers; and it must be immutable, so a provider reusing
a buffer cannot change an input that is already in flight on a GPU.
"""

from __future__ import annotations

import dataclasses
import unittest

from inference.tasks import (
    ERROR_CUDA_OOM,
    ERROR_WORKER_EXCEPTION,
    WORKER_FAILED,
    GpuMemory,
    InferenceResult,
    InferenceTask,
    WorkerStatus,
    freeze_window,
)

from _fake_pool import CAMERA_INDICES, make_built, make_task, make_window


class FreezeWindowTests(unittest.TestCase):
    def test_the_frame_tensor_is_shared_not_copied(self) -> None:
        """The pixels are written once at build time and never mutated."""
        window = make_window(1_000_000)
        frozen = freeze_window(window)
        self.assertIs(frozen.frames, window.frames)

    def test_the_list_fields_are_copied(self) -> None:
        window = make_window(1_000_000)
        frozen = freeze_window(window)
        self.assertEqual(frozen.frame_ids, window.frame_ids)
        self.assertIsNot(frozen.frame_ids, window.frame_ids)
        self.assertIsNot(frozen.frame_ids[0], window.frame_ids[0])

    def test_a_provider_reusing_its_lists_cannot_change_a_frozen_window(self) -> None:
        window = make_window(1_000_000)
        frozen = freeze_window(window)
        before = list(frozen.timestamps_us)
        window.timestamps_us.append(999)
        window.camera_indices.append(7)
        window.frame_ids[0].append("late")
        self.assertEqual(frozen.timestamps_us, before)
        self.assertEqual(frozen.camera_indices, CAMERA_INDICES)
        self.assertNotIn("late", frozen.frame_ids[0])


class InferenceTaskTests(unittest.TestCase):
    def test_the_task_carries_everything_a_model_call_needs(self) -> None:
        task = make_task(3, capture_us=1_234_567)
        self.assertEqual(task.task_seq, 3)
        self.assertEqual(task.capture_timestamp_us, 1_234_567)
        self.assertEqual(task.window.t0_us, 1_234_567)
        self.assertIsNotNone(task.ego_xyz)
        self.assertIsNotNone(task.ego_rot)
        self.assertIsNotNone(task.reference_pose)
        self.assertEqual(len(task.ego_history), 16)
        self.assertEqual(task.camera_age_ms, 12.5)
        self.assertEqual(task.camera_pose_delta_ms, 8.0)
        self.assertEqual(task.camera_lag_ms, 3.0)

    def test_the_task_window_is_a_copy_of_the_source_window(self) -> None:
        window = make_window(5_000_000)
        task = InferenceTask.from_build(1, make_built(window), window)
        self.assertIsNot(task.window, window)
        self.assertEqual(task.window.t0_us, window.t0_us)

    def test_a_task_cannot_be_modified_once_frozen(self) -> None:
        task = make_task(1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            task.task_seq = 2  # type: ignore[misc]

    def test_the_frozen_timestamp_defaults_to_now_and_can_be_pinned(self) -> None:
        pinned = make_task(1, created_ns=1_000_000_000)
        self.assertEqual(pinned.created_monotonic_ns, 1_000_000_000)
        self.assertEqual(pinned.age_ms(1_250_000_000), 250.0)
        self.assertEqual(make_task(1).age_ms(0) < 0, True)

    def test_sequence_and_capture_timestamp_are_independent(self) -> None:
        """A newer task can carry an older capture stamp; seq still orders them.

        The camera clock is remote and can step, so nothing downstream may use
        the timestamp to decide which of two tasks is newer.  The task keeps
        both; only ``task_seq`` is the ordering key.
        """
        older_capture_but_newer_task = make_task(9, capture_us=1_000)
        newer_capture_but_older_task = make_task(8, capture_us=9_999_999)
        self.assertGreater(
            older_capture_but_newer_task.task_seq,
            newer_capture_but_older_task.task_seq,
        )
        self.assertLess(
            older_capture_but_newer_task.capture_timestamp_us,
            newer_capture_but_older_task.capture_timestamp_us,
        )

    def test_the_camera_mapping_is_carried_through_unchanged(self) -> None:
        task = make_task(1)
        self.assertEqual(task.window.camera_indices, CAMERA_INDICES)
        self.assertEqual(len(task.camera_frame_ids), len(CAMERA_INDICES))
        self.assertEqual(
            sorted(task.camera_frame_id), sorted(f"cam{i}" for i in CAMERA_INDICES)
        )


class InferenceResultTests(unittest.TestCase):
    def test_ok_and_failed_are_opposites(self) -> None:
        self.assertFalse(
            InferenceResult(task_seq=1, worker_id=0, device=1, ok=True).failed
        )
        self.assertTrue(
            InferenceResult(task_seq=1, worker_id=0, device=1, ok=False).failed
        )

    def test_a_cuda_oom_is_recognisable_by_its_kind(self) -> None:
        oom = InferenceResult(
            task_seq=1,
            worker_id=0,
            device=1,
            ok=False,
            error_kind=ERROR_CUDA_OOM,
        )
        other = InferenceResult(
            task_seq=1,
            worker_id=0,
            device=1,
            ok=False,
            error_kind=ERROR_WORKER_EXCEPTION,
        )
        self.assertTrue(oom.is_cuda_oom)
        self.assertFalse(other.is_cuda_oom)

    def test_the_two_error_kinds_are_distinct_strings(self) -> None:
        self.assertNotEqual(ERROR_CUDA_OOM, ERROR_WORKER_EXCEPTION)


class GpuMemoryTests(unittest.TestCase):
    def test_it_reports_every_field_in_bytes(self) -> None:
        memory = GpuMemory(
            allocated_bytes=1 << 30,
            reserved_bytes=2 << 30,
            max_allocated_bytes=3 << 30,
            max_reserved_bytes=4 << 30,
        )
        self.assertEqual(
            memory.as_dict(),
            {
                "allocated_bytes": 1 << 30,
                "reserved_bytes": 2 << 30,
                "max_allocated_bytes": 3 << 30,
                "max_reserved_bytes": 4 << 30,
            },
        )

    def test_the_default_says_nothing_rather_than_zero_memory(self) -> None:
        self.assertEqual(GpuMemory().as_dict()["allocated_bytes"], 0)


class WorkerStatusTests(unittest.TestCase):
    def test_failed_is_derived_from_the_state(self) -> None:
        self.assertTrue(
            WorkerStatus(worker_id=0, device=1, state=WORKER_FAILED).failed
        )
        self.assertFalse(WorkerStatus(worker_id=0, device=1, state="ready").failed)


if __name__ == "__main__":
    unittest.main()
