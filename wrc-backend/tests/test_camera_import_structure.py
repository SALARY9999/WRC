"""Structural guards for the camera package.

``camera.cimg`` and ``camera.cimg_source`` must stay importable without
``camera.sync`` or OpenCV.  ``CameraWindowProvider`` always builds one CIMG
receiver.  The checks that import ``camera.sync`` run in a subprocess with a
cv2 stub, because the local interpreter does not have OpenCV.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_CV2_STUB = """
import sys, types
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
"""


def _run(body: str, *, with_cv2_stub: bool = False) -> str:
    script = (_CV2_STUB if with_cv2_stub else "") + textwrap.dedent(body)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"subprocess failed ({completed.returncode})\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )
    return completed.stdout


class ImportStructureTests(unittest.TestCase):
    def test_frame_entry_is_one_class_not_two_copies(self):
        out = _run("""
            import camera.models
            import camera.sync
            print(camera.sync.FrameEntry is camera.models.FrameEntry)
            print(camera.sync.CameraWindow is camera.models.CameraWindow)
        """, with_cv2_stub=True)
        same_entry, same_window = out.split()
        self.assertEqual(same_entry, "True")
        self.assertEqual(same_window, "True")

    def test_the_mpegts_decoder_module_is_gone(self):
        out = _run("""
            try:
                import camera.source
            except ModuleNotFoundError as exc:
                print("missing", exc.name)
            else:
                print("present")
        """)
        self.assertEqual(out.split(), ["missing", "camera.source"])

    def test_the_protocol_layer_stays_stdlib_only(self):
        out = _run("""
            import sys
            import camera.cimg
            print([m for m in ("numpy", "torch", "cv2") if m in sys.modules] or "none")
        """)
        self.assertEqual("none", out.strip())

    def test_the_receiver_does_not_import_the_synchronizer(self):
        out = _run("""
            import sys
            import camera.cimg_source
            print("camera.sync" in sys.modules, "cv2" in sys.modules)
        """)
        sync_loaded, cv2_loaded = out.split()
        self.assertEqual("False", sync_loaded)
        self.assertEqual("False", cv2_loaded)

    def test_the_receiver_shares_the_model_frame_type(self):
        out = _run("""
            import camera.cimg_source, camera.models
            print(camera.cimg_source.CimgConfig.__module__)
            print(camera.cimg_source.FrameEntry is camera.models.FrameEntry)
        """)
        module, one_class = out.split()
        self.assertEqual("camera.cimg_source", module)
        self.assertEqual("True", one_class)


class CimgProviderTests(unittest.TestCase):
    """The window provider builds the CIMG receiver directly."""

    _PROBE = """
        import sys
        from camera.cimg_source import CimgConfig
        import camera.sync
        from time_sync import TimestampManager
        tm = TimestampManager()
        config = None if sys.argv[1] == "none" else CimgConfig(port=0, group_fps=5.0)
        try:
            provider = camera.sync.CameraWindowProvider(
                config, frames_per_camera=4, target_fps=5.0, timestamp_manager=tm)
        except ValueError as exc:
            print("ValueError", "CimgConfig" in str(exc))
        else:
            print("OK", len(provider.buffers), provider.period_us,
                  "x".join(str(v) for v in provider.target_size),
                  len(provider.readers), type(provider.readers[0]).__name__,
                  ",".join(str(i) for i in provider.camera_indices))
    """

    def _probe(self, argument: str) -> list[str]:
        completed = subprocess.run(
            [sys.executable, "-c", _CV2_STUB + textwrap.dedent(self._PROBE), argument],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"cimg probe {argument!r} failed\n--- stderr ---\n{completed.stderr}"
            )
        return completed.stdout.split()

    def test_the_provider_builds_one_reader_and_four_buffers(self):
        mode, buffers, period_us, target_size, readers, reader_class, indices = (
            self._probe("on")
        )
        self.assertEqual("OK", mode)
        self.assertEqual("4", buffers)
        self.assertEqual("200000", period_us)
        self.assertEqual("576x320", target_size)
        self.assertEqual("1", readers)
        self.assertEqual("CimgCameraSource", reader_class)
        self.assertEqual("1,2,0,4", indices)

    def test_a_missing_config_is_refused(self):
        self.assertEqual(["ValueError", "True"], self._probe("none"))

    def test_startup_spawns_no_subprocess(self):
        out = _run("""
            import subprocess, time
            import camera.sync
            from camera.cimg_source import CimgConfig
            from time_sync import TimestampManager

            def refuse(*args, **kwargs):
                raise AssertionError(f"subprocess spawned: {args!r}")

            subprocess.Popen = refuse
            provider = camera.sync.CameraWindowProvider(
                CimgConfig(host="127.0.0.1", port=0),
                timestamp_manager=TimestampManager(),
            )
            provider.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if provider.readers[0].connection_state == "listening":
                    break
                time.sleep(0.01)
            print(provider.readers[0].connection_state,
                  type(provider.readers[0]).__name__)
            provider.stop()
            provider.join(1.0)
        """, with_cv2_stub=True)
        state, reader_class = out.split()
        self.assertEqual("listening", state)
        self.assertEqual("CimgCameraSource", reader_class)


if __name__ == "__main__":
    unittest.main()
