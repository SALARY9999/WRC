"""Isolated offline inference API. Never starts the vehicle pipeline."""
from __future__ import annotations

import base64
import binascii
import logging
import os
import threading
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from app import AlpamayoRunner, CameraWindow, PoseSample, Settings, build_ego_arrays, yaw_to_matrix


class HistoryPose(BaseModel):
    timestamp_us: int = Field(ge=0)
    xyz: list[float] = Field(min_length=3, max_length=3)
    yaw_rad: float


class GeographicHistoryPose(BaseModel):
    timestamp_us: int = Field(ge=0)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    altitude: float = Field(allow_inf_nan=False)
    heading_deg: float = Field(allow_inf_nan=False)
    # Recorded diagnostic coordinates; conversion uses geographic pose fields.
    x_m: float | None = Field(default=None, allow_inf_nan=False)
    y_m: float | None = Field(default=None, allow_inf_nan=False)


class CameraInput(BaseModel):
    camera_index: int
    timestamps_us: list[int] = Field(min_length=4, max_length=4)
    images_base64: list[str] = Field(min_length=4, max_length=4)


class InferenceRequest(BaseModel):
    nav_text: str = Field(min_length=1, max_length=4096)
    cameras: list[CameraInput] = Field(min_length=4, max_length=4)
    ego_history: list[GeographicHistoryPose] | None = Field(default=None, min_length=16, max_length=16)
    history: list[HistoryPose] | None = Field(default=None, min_length=16, max_length=16)

    @model_validator(mode="after")
    def one_history_format(self):
        if (self.ego_history is None) == (self.history is None):
            raise ValueError("provide exactly one of ego_history or history")
        return self


app = FastAPI(title="Alpamayo offline model test", version="1.0")
settings = Settings(mock_mode=False, frames_per_camera=4, num_traj_samples=1)
# Only model loading settings are inherited. No realtime pipeline is constructed.
settings.checkpoint = os.environ.get("ALPAMAYO_CHECKPOINT", settings.checkpoint)
runner = AlpamayoRunner(settings)
inference_lock = threading.Lock()


def prepare(request: InferenceRequest):
    if not request.nav_text.strip():
        raise ValueError("nav_text must not be blank")
    if sorted(c.camera_index for c in request.cameras) != [0, 1, 2, 4]:
        raise ValueError("camera indices must be 0(left), 1(front), 2(right), 4(back)")
    history = request.ego_history if request.ego_history is not None else request.history
    stamps = [p.timestamp_us for p in history]
    if any(abs(b - a - 100000) > 1000 for a, b in zip(stamps, stamps[1:])):
        raise ValueError("history must be oldest first at 10Hz (100000us +/-1000us)")
    if request.ego_history is not None:
        samples = [PoseSample(p.timestamp_us, p.latitude, p.longitude, p.altitude,
                              p.heading_deg) for p in history]
        # Same heading semantics and conversion as the realtime backend.
        ego_xyz, ego_rot, _ = build_ego_arrays(samples)
    else:
        xyz = np.asarray([p.xyz for p in history], dtype=np.float64)
        yaw = np.asarray([p.yaw_rad for p in history], dtype=np.float64)
        if not np.isfinite(xyz).all() or not np.isfinite(yaw).all():
            raise ValueError("history coordinates and yaw must be finite")
        xyz = (xyz - xyz[-1]) @ yaw_to_matrix(float(yaw[-1]))
        rot = np.stack([yaw_to_matrix(float(y - yaw[-1])) for y in yaw])
        ego_xyz = torch.tensor(xyz, dtype=torch.float32)[None, None]
        ego_rot = torch.tensor(rot, dtype=torch.float32)[None, None]
    frame_stamps = stamps[-4:]
    cameras = sorted(request.cameras, key=lambda c: c.camera_index)
    frames = []
    for camera in cameras:
        if any(abs(a - b) > 1000 for a, b in zip(camera.timestamps_us, frame_stamps)):
            raise ValueError("camera timestamps must match the last 4 history timestamps +/-1000us")
        decoded = []
        for encoded in camera.images_base64:
            if len(encoded) > 8 * 1024 * 1024:
                raise ValueError("each Base64 image must be <=8MiB")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("invalid image Base64; supply raw Base64 without data URL prefix") from exc
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR) if raw else None
            if image is None:
                raise ValueError("image cannot be decoded")
            image = cv2.cvtColor(cv2.resize(image, (576, 320)), cv2.COLOR_BGR2RGB)
            decoded.append(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1))
        frames.append(torch.stack(decoded))
    window = CameraWindow(t0_us=stamps[-1], frames=torch.stack(frames),
                          camera_indices=[0, 1, 2, 4], timestamps_us=frame_stamps)
    return window, ego_xyz, ego_rot


@app.get("/health")
def health():
    return {"mode": "offline_test", "model_loaded": runner._model is not None,
            "busy": inference_lock.locked(), "waypoint_sending": False}


@app.post("/infer")
def infer(request: InferenceRequest):
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(409, "model is busy; retry after the current request")
    try:
        try:
            window, xyz, rot = prepare(request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        settings.navigation_text = request.nav_text.strip()
        started = time.perf_counter()
        p = request.ego_history[-1] if request.ego_history is not None else None
        ref = (PoseSample(p.timestamp_us, p.latitude, p.longitude, p.altitude, p.heading_deg)
               if p is not None else PoseSample(window.t0_us, 0, 0, 0, 0))
        result = runner.infer(window, xyz, rot, ref)
        if not np.isfinite(result.positions).all() or not np.isfinite(result.rotations).all():
            raise RuntimeError("model returned non-finite trajectory")
        return {"cot": result.cot, "positions": result.positions.tolist(),
                "rotations": result.rotations.tolist(), "frame": "latest_ego",
                "trajectory_dt_s": settings.trajectory_dt_s, "t0_us": window.t0_us,
                "elapsed_s": time.perf_counter() - started,
                "prompt_text": result.prompt_text,
                "input_ids_sha256": result.input_ids_sha256}
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("offline inference failed")
        raise HTTPException(500, "model inference failed; inspect test service logs") from exc
    finally:
        inference_lock.release()
