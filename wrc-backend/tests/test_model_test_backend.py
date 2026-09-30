"""Offline API checks with real image decoding, without loading GPU weights."""
import base64
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError

import test_backend as backend


class OfflineTests(unittest.TestCase):
    def geographic_sample(self):
        data = json.loads((Path(__file__).resolve().parents[1] /
                           'examples/model_test_sample.json').read_text(encoding='utf-8'))
        images = self.sample().cameras[0].images_base64
        for camera in data['cameras']:
            camera.pop('image_paths')
            camera['images_base64'] = images
        return backend.InferenceRequest(**data)

    def test_supplied_geographic_history_matches_realtime_conversion(self):
        sample = self.geographic_sample()
        _, xyz, rot = backend.prepare(sample)
        expected_xyz, expected_rot, _ = backend.build_ego_arrays([
            backend.PoseSample(p.timestamp_us, p.latitude, p.longitude, p.altitude,
                               p.heading_deg) for p in sample.ego_history])
        np.testing.assert_allclose(xyz.numpy(), expected_xyz.numpy())
        np.testing.assert_allclose(rot.numpy(), expected_rot.numpy())
        np.testing.assert_allclose(xyz[0, 0, -1].numpy(), [0, 0, 0])
        # The first frame carries the x/y the inference package recorded for it;
        # recomputing from lat/lon must land in the same place.  Comparing
        # against the sample instead of a literal copied out of it is what keeps
        # the two from drifting apart unnoticed - the previous literal had.
        recorded_xy = [
            sample.ego_history[0].x_m,
            sample.ego_history[0].y_m,
        ]
        np.testing.assert_allclose(xyz[0, 0, 0, :2].numpy(), recorded_xy, atol=1e-3)
        # Rounded diagnostic xy must not override geographic input.
        sample.ego_history[0].x_m = 999
        sample.ego_history[0].y_m = 999
        _, actual, _ = backend.prepare(sample)
        np.testing.assert_allclose(actual.numpy(), expected_xyz.numpy())

    def test_geographic_history_validation(self):
        sample = self.geographic_sample()
        payload = sample.model_dump()
        for field, value in [('latitude', 91), ('longitude', float('nan')),
                             ('altitude', float('inf'))]:
            altered = json.loads(json.dumps(payload))
            altered['ego_history'][0][field] = value
            with self.assertRaises(ValidationError):
                backend.InferenceRequest(**altered)
        with self.assertRaises(ValidationError):
            backend.InferenceRequest(nav_text=sample.nav_text, cameras=sample.cameras)
        with self.assertRaises(ValidationError):
            backend.InferenceRequest(nav_text=sample.nav_text, cameras=sample.cameras,
                                     ego_history=sample.ego_history, history=self.sample().history)

    def sample(self):
        _, image = cv2.imencode('.jpg', np.zeros((32, 48, 3), dtype=np.uint8))
        encoded = base64.b64encode(image.tobytes()).decode()
        stamps = [1000000 + i * 100000 for i in range(16)]
        return backend.InferenceRequest(
            nav_text='Turn left',
            cameras=[backend.CameraInput(camera_index=k, timestamps_us=stamps[-4:],
                                         images_base64=[encoded] * 4) for k in [4, 2, 1, 0]],
            history=[backend.HistoryPose(timestamp_us=t, xyz=[i * .1, 0, 0], yaw_rad=0)
                     for i, t in enumerate(stamps)])

    def test_shapes_and_latest_origin(self):
        window, xyz, rot = backend.prepare(self.sample())
        self.assertEqual(tuple(window.frames.shape), (4, 4, 3, 320, 576))
        self.assertEqual(window.camera_indices, [0, 1, 2, 4])
        self.assertEqual(tuple(xyz.shape), (1, 1, 16, 3))
        np.testing.assert_allclose(xyz[0, 0, -1].numpy(), [0, 0, 0])
        np.testing.assert_allclose(xyz[0, 0, 0].numpy(), [-1.5, 0, 0])
        np.testing.assert_allclose(rot[0, 0, -1].numpy(), np.eye(3))

    def test_yaw_frame(self):
        sample = self.sample()
        for p in sample.history:
            p.yaw_rad = np.pi / 2
        _, xyz, _ = backend.prepare(sample)
        np.testing.assert_allclose(xyz[0, 0, 0].numpy(), [0, 1.5, 0], atol=1e-6)

    def test_invalid_input_never_calls_model_and_releases_lock(self):
        for kind in ['image', 'timestamp', 'camera', 'finite']:
            sample = self.sample()
            if kind == 'image':
                sample.cameras[0].images_base64[0] = 'invalid!'
            elif kind == 'timestamp':
                sample.cameras[0].timestamps_us[0] += 50000
            elif kind == 'camera':
                sample.cameras[0].camera_index = 0
            else:
                sample.history[0].xyz[0] = float('nan')
            with patch.object(backend.runner, 'infer') as model:
                with self.assertRaises(HTTPException) as error:
                    backend.infer(sample)
                self.assertEqual(error.exception.status_code, 422)
                model.assert_not_called()
                self.assertFalse(backend.inference_lock.locked())

    def test_busy(self):
        backend.inference_lock.acquire()
        try:
            with self.assertRaises(HTTPException) as error:
                backend.infer(self.sample())
            self.assertEqual(error.exception.status_code, 409)
        finally:
            backend.inference_lock.release()


if __name__ == '__main__':
    unittest.main()
