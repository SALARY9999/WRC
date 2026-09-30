#!/usr/bin/env python3
"""Check that the inference package writer records real frame ids."""

import json
import os
import sys
import time

import numpy as np
import torch

BACKEND = "/data/users_data/wanghaohao/701/alpamayo_infer_backend"
PKG_DIR = "/data/users_data/wanghaohao/capture/pkg_test"
sys.path.insert(0, BACKEND)
os.environ["ALPAMAYO_INFER_PACKAGE_DIR"] = PKG_DIR

import app  # noqa: E402


def main() -> int:
    settings = app.Settings.from_env()
    runner = app.PipelineRunner(settings)
    frames = torch.zeros((4, 4, 3, 320, 576), dtype=torch.uint8)
    frame_ids = [
        [f"cam0:{1000 + i}" for i in range(4)],
        [f"cam1:{2000 + i}" for i in range(4)],
        [f"cam2:{3000 + i}" for i in range(4)],
        [f"cam4:{4000 + i}" for i in range(4)],
    ]
    window = app.CameraWindow(
        t0_us=123456789,
        frames=frames,
        camera_indices=[0, 1, 2, 4],
        timestamps_us=[123456789 - 3, 123456789 - 2, 123456789 - 1, 123456789],
        frame_ids=frame_ids,
    )
    record = {
        "send_timestamp": time.time(),
        "event": "infer_record",
        "camera_frame_id": {"left": "cam0:1003", "front": "cam1:2003"},
    }
    path = runner._save_infer_package(window, record)
    print(f"package={path}")
    with open(os.path.join(path, "infer.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"camera_frame_id={data['camera_frame_id']}")
    print(f"first_image={data['images'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
