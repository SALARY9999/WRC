"""Direct Alpamayo 1.5 inference on a captured package, without backend imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

CAMERAS = [(0, "left"), (1, "front"), (2, "right"), (4, "back")]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_from_ego_rotation(heading_deg: float, np):
    """Columns are vehicle forward, left and up expressed in ENU coordinates."""
    heading = math.radians(heading_deg)
    return np.array(
        [[math.sin(heading), -math.cos(heading), 0.0],
         [math.cos(heading),  math.sin(heading), 0.0],
         [0.0,                0.0,               1.0]],
        dtype=np.float64,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", default=os.environ.get(
        "ALPAMAYO_CHECKPOINT",
        "/data/users_data/wanghaohao/701/model_weights/Alpamayo-1.5-10B"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    package = args.package.resolve()
    output = args.output or package / "official_model_result.json"
    record_path = package / "infer.json"
    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    history = record.get("ego_history")
    if not isinstance(history, list) or len(history) != 16:
        raise ValueError("infer.json must contain exactly 16 ego_history entries")
    required = ("timestamp_us", "latitude", "longitude", "altitude",
                "heading_deg", "x_m", "y_m")
    for index, pose in enumerate(history):
        if any(key not in pose for key in required):
            raise ValueError(f"ego_history[{index}] is missing required fields")
        if any(isinstance(pose[key], bool) or not isinstance(pose[key], (int, float))
               or not math.isfinite(pose[key]) for key in required):
            raise ValueError(f"ego_history[{index}] contains a non-finite value")
    timestamps = [pose["timestamp_us"] for pose in history]
    if any(b - a != 100000 for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("ego_history must be oldest-first at exactly 10 Hz")
    if abs(history[-1]["x_m"]) > 1e-4 or abs(history[-1]["y_m"]) > 1e-4:
        raise ValueError("the final x_m/y_m must be the local origin")

    # x_m/y_m are already relative to the latest pose in the latest ego frame.
    # z follows the official loader: world z minus t0 z (yaw does not change z).
    t0_altitude = history[-1]["altitude"]
    history_xyz_np = np.array(
        [[pose["x_m"], pose["y_m"], pose["altitude"] - t0_altitude]
         for pose in history], dtype=np.float32)

    # Official convention: R_local_i = inverse(R_world_t0) @ R_world_i.
    t0_rotation = world_from_ego_rotation(history[-1]["heading_deg"], np)
    history_rot_np = np.stack([
        t0_rotation.T @ world_from_ego_rotation(pose["heading_deg"], np)
        for pose in history
    ]).astype(np.float32)
    np.testing.assert_allclose(history_rot_np[-1], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(
        history_rot_np @ np.swapaxes(history_rot_np, -1, -2),
        np.broadcast_to(np.eye(3), history_rot_np.shape), atol=1e-6)

    # Independently check rounded local xy against latitude/longitude conversion.
    ref = history[-1]
    heading = math.radians(ref["heading_deg"])
    xy_errors = []
    for pose in history:
        east = ((pose["longitude"] - ref["longitude"]) * 111320.0
                * math.cos(math.radians(ref["latitude"])))
        north = (pose["latitude"] - ref["latitude"]) * 110540.0
        x = east * math.sin(heading) + north * math.cos(heading)
        y = -east * math.cos(heading) + north * math.sin(heading)
        xy_errors.append(math.hypot(x - pose["x_m"], y - pose["y_m"]))
    if max(xy_errors) > 0.02:
        raise ValueError(f"x_m/y_m disagree with geographic pose by {max(xy_errors):.3f}m")

    frames, image_manifest = [], []
    for camera_index, camera_name in CAMERAS:
        for temporal_index in range(4):
            image_path = package / "images" / f"{camera_name}_t{temporal_index}.jpg"
            if not image_path.is_file():
                image_path = package / f"{camera_name}_t{temporal_index}.jpg"
            raw = np.fromfile(image_path, dtype=np.uint8)
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"cannot decode {image_path}")
            rgb = cv2.cvtColor(cv2.resize(bgr, (576, 320)), cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
            image_manifest.append({
                "camera_index": camera_index,
                "camera_name": camera_name,
                "temporal_index": temporal_index,
                "file": str(image_path),
                "sha256": sha256_file(image_path),
            })
    image_frames = torch.stack(frames).reshape(4, 4, 3, 320, 576)
    camera_indices = torch.tensor([camera for camera, _ in CAMERAS], dtype=torch.int64)
    ego_history_xyz = torch.from_numpy(history_xyz_np).unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.from_numpy(history_rot_np).unsqueeze(0).unsqueeze(0)

    # This follows the official test_inference.py call path exactly: no nav_text.
    messages = helper.create_message(
        frames=image_frames.flatten(0, 1), camera_indices=camera_indices)
    load_started = time.perf_counter()
    model = Alpamayo1_5.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    model_load_s = time.perf_counter() - load_started
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    input_ids_sha256 = hashlib.sha256(
        inputs["input_ids"].cpu().numpy().astype(np.int32).tobytes()).hexdigest()
    model_inputs = helper.to_device({
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }, "cuda")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=0.98, temperature=0.6,
            num_traj_samples=1, max_generation_length=256, return_extra=True)
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - inference_started

    pred_xyz_np = pred_xyz.float().cpu().numpy()
    pred_rot_np = pred_rot.float().cpu().numpy()
    positions = pred_xyz_np[0, 0, 0]
    rotations = pred_rot_np[0, 0, 0]
    if positions.shape != (64, 3) or not np.isfinite(positions).all():
        raise ValueError(f"unexpected pred_xyz shape/content: {pred_xyz_np.shape}")
    extra_json = {key: np.asarray(value).tolist() for key, value in extra.items()}
    cot = str(np.asarray(extra["cot"]).reshape(-1)[0]) if "cot" in extra else ""
    try:
        source_commit = subprocess.check_output(
            ["git", "-C", str(Path(helper.__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        source_commit = None
    result = {
        "validation_type": "direct_official_alpamayo_1_5",
        "business_backend_imported": False,
        "navigation_text": None,
        "checkpoint": args.checkpoint,
        "alpamayo_source_commit": source_commit,
        "seed": args.seed,
        "sampling": {"top_p": 0.98, "temperature": 0.6,
                     "num_traj_samples": 1, "max_generation_length": 256},
        "model_load_s": model_load_s,
        "inference_s": inference_s,
        "cot": cot,
        "pred_xyz_shape": list(pred_xyz_np.shape),
        "pred_rot_shape": list(pred_rot_np.shape),
        "positions": positions.tolist(),
        "rotations": rotations.tolist(),
        "raw_extra": extra_json,
        "frame": "latest_ego",
        "trajectory_dt_s": 0.1,
        "t0_us": timestamps[-1],
        "reference_pose": {key: ref[key] for key in
                           ("timestamp_us", "latitude", "longitude", "altitude", "heading_deg")},
        "ego_history_xyz": history_xyz_np.tolist(),
        "ego_history_rot": history_rot_np.tolist(),
        "max_xy_crosscheck_error_m": max(xy_errors),
        "camera_indices": camera_indices.tolist(),
        "image_manifest": image_manifest,
        "input_ids_shape": list(inputs["input_ids"].shape),
        "input_ids_sha256": input_ids_sha256,
        "source_infer_json": str(record_path),
        "source_infer_json_sha256": sha256_file(record_path),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "cot": cot,
                      "points": len(positions), "inference_s": inference_s,
                      "model_load_s": model_load_s}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
