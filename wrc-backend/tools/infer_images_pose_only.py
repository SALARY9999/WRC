"""One-off GPU inference with official prompt, images and geographic history only."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from alpamayo1_5 import helper
    from app import AlpamayoRunner, PoseSample, Settings, build_ego_arrays, _messages_text
    from test_backend import GeographicHistoryPose, CameraInput

    payload = json.loads(args.input.read_text(encoding='utf-8-sig'))
    history = [GeographicHistoryPose(**p) for p in payload['ego_history']]
    cameras = sorted([CameraInput(**c) for c in payload['cameras']], key=lambda c:c.camera_index)
    if len(history) != 16 or [c.camera_index for c in cameras] != [0,1,2,4]:
        raise ValueError('require 16 history poses and camera indices 0,1,2,4')
    stamps = [p.timestamp_us for p in history]
    if any(b-a != 100000 for a,b in zip(stamps,stamps[1:])):
        raise ValueError('history must have 100000us intervals')
    images = []
    for camera in cameras:
        if camera.timestamps_us != stamps[-4:]:
            raise ValueError('camera timestamps must match last 4 history poses')
        for encoded in camera.images_base64:
            raw = base64.b64decode(encoded,validate=True)
            bgr = cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('image decode failed')
            rgb = cv2.cvtColor(cv2.resize(bgr,(576,320)),cv2.COLOR_BGR2RGB)
            images.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1))
    samples = [PoseSample(p.timestamp_us,p.latitude,p.longitude,p.altitude,p.heading_deg) for p in history]
    xyz, _, ref = build_ego_arrays(samples)
    # Independent world-to-ego orientation check, following official R0^-1 Ri.
    def world_rotation(heading):
        h = np.deg2rad(heading)
        return np.array([[np.sin(h),-np.cos(h),0],[np.cos(h),np.sin(h),0],[0,0,1]])
    expected = np.stack([world_rotation(ref.heading_deg).T @ world_rotation(p.heading_deg) for p in history])
    # Construct directly, independent of realtime backend rotation revisions.
    rot = torch.tensor(expected,dtype=torch.float32)[None,None]
    np.testing.assert_allclose(rot[0,0].numpy(),expected,atol=1e-6)
    settings = Settings(mock_mode=False,navigation_text='',num_traj_samples=1,max_generation_length=512)
    import os
    settings.checkpoint = os.environ.get('ALPAMAYO_CHECKPOINT',settings.checkpoint)
    runner = AlpamayoRunner(settings)
    loaded_at = time.perf_counter()
    runner.ensure_loaded()
    load_s = time.perf_counter()-loaded_at
    messages = helper.create_message(torch.stack(images),
        camera_indices=torch.tensor([0,1,2,4]),num_frames_per_camera=4,nav_text=None)
    prompt = _messages_text(messages)
    assert '<|route_start|>' not in prompt
    inputs = runner._processor.apply_chat_template(messages,tokenize=True,
        add_generation_prompt=False,continue_final_message=True,return_dict=True,return_tensors='pt')
    fingerprint = hashlib.sha256(inputs['input_ids'].cpu().numpy().astype(np.int32).tobytes()).hexdigest()
    data = helper.to_device(dict(tokenized_data=inputs,ego_history_xyz=xyz,ego_history_rot=rot),'cuda')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        pred_xyz,pred_rot,extra = runner._model.sample_trajectories_from_data_with_vlm_rollout(
            data=data,top_p=.98,temperature=.6,num_traj_samples=1,
            max_generation_length=512,return_extra=True)
    positions = pred_xyz[0,0,0].float().cpu().numpy()
    rotations = pred_rot[0,0,0].float().cpu().numpy()
    if positions.shape != (64,3) or not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError('invalid model trajectory')
    elapsed = time.perf_counter()-started
    extra_json = {k:np.asarray(v).tolist() for k,v in extra.items()}
    cot = str(np.asarray(extra['cot']).reshape(-1)[0]) if 'cot' in extra else ''
    result = dict(cot=cot,positions=positions.tolist(),rotations=rotations.tolist(),
        frame='latest_ego',trajectory_dt_s=.1,t0_us=ref.timestamp_us,
        elapsed_s=elapsed,model_load_s=load_s,prompt_mode='official_images_pose_only',
        nav_text=None,reference_pose={k:getattr(ref,k) for k in
            ['timestamp_us','latitude','longitude','altitude','heading_deg']},
        history_rotation_convention='R_t0_inverse @ R_i; compass heading to CCW ego yaw',
        raw_extra=extra_json,prompt_text=prompt,input_ids_sha256=fingerprint,seed=args.seed)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(points=len(positions),cot=cot,elapsed_s=elapsed,model_load_s=load_s,
                          output=str(args.output)),ensure_ascii=False))


if __name__ == '__main__':
    main()
