"""Load Alpamayo 1.5 and run one deterministic synthetic-input GPU inference."""
from __future__ import annotations

import argparse
import json
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--max-generation-length", type=int, default=32)
    args = parser.parse_args()

    import numpy as np
    import torch
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    helper.BASE_PROCESSOR_NAME = args.processor
    # Transformers serializes a config by constructing a default instance.
    # Alpamayo's default instance performs an unrelated remote processor lookup;
    # the real checkpoint config is complete, so skip that comparison offline.
    Alpamayo1_5.config_class.has_no_defaults_at_init = True
    frames = torch.full((4, 4, 3, 320, 576), 127, dtype=torch.uint8)
    camera_indices = torch.tensor([0, 1, 2, 4], dtype=torch.int64)
    ego_xyz = torch.zeros((1, 1, 16, 3), dtype=torch.float32)
    ego_rot = torch.eye(3, dtype=torch.float32).repeat(1, 1, 16, 1, 1)
    messages = helper.create_message(frames.flatten(0, 1), camera_indices)

    before_free, total = torch.cuda.mem_get_info(0)
    load_started = time.perf_counter()
    model = Alpamayo1_5.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    after_load_free, _ = torch.cuda.mem_get_info(0)

    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    data = helper.to_device(
        {
            "tokenized_data": tokenized,
            "ego_history_xyz": ego_xyz,
            "ego_history_rot": ego_rot,
        },
        "cuda",
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=data,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=args.max_generation_length,
            return_extra=True,
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    after_infer_free, _ = torch.cuda.mem_get_info(0)

    xyz = pred_xyz.float().cpu().numpy()
    rot = pred_rot.float().cpu().numpy()
    if xyz.shape[-2:] != (64, 3) or not np.isfinite(xyz).all():
        raise RuntimeError(f"invalid trajectory output: shape={xyz.shape}")
    if rot.shape[-2:] != (3, 3) or not np.isfinite(rot).all():
        raise RuntimeError(f"invalid rotation output: shape={rot.shape}")
    print(json.dumps({
        "status": "ok",
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "memory_total_bytes": total,
        "memory_model_bytes": before_free - after_load_free,
        "memory_peak_observed_bytes": before_free - min(after_load_free, after_infer_free),
        "pred_xyz_shape": list(xyz.shape),
        "pred_rot_shape": list(rot.shape),
        "cot": str(np.asarray(extra.get("cot", "")).reshape(-1)[0]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
