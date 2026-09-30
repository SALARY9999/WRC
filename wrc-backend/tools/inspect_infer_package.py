#!/usr/bin/env python3
"""看一眼最近的推理包：时间戳模式、帧标识（PTS）是否齐全、时间域量级。

用法：
    /data/users_data/wanghaohao/701/alpasim/.venv.bak/bin/python tools/inspect_infer_package.py [包目录或个数]
默认检查 /data/users_data/wanghaohao/infer_packages 下最新的包；
给数字 N 则统计最近 N 个包；给路径则只看该包。
"""

from __future__ import annotations

import json
import os
import sys

BASE = "/data/users_data/wanghaohao/infer_packages"


def newest(n: int) -> list[str]:
    names = [
        os.path.join(BASE, x)
        for x in os.listdir(BASE)
        if os.path.isdir(os.path.join(BASE, x))
    ]
    names.sort(key=os.path.getmtime, reverse=True)
    return names[:n]


def summarize(pkg: str) -> dict:
    path = os.path.join(pkg, "infer.json")
    if not os.path.exists(pkg) and os.path.exists(pkg + "/infer.json"):
        path = pkg + "/infer.json"
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    imgs = d.get("images") or []
    ref = d.get("reference_pose") or {}
    ids = [i.get("frame_id") for i in imgs]
    return {
        "pkg": os.path.basename(path.rsplit("/", 1)[0]),
        "camera_clock": d.get("camera_clock", d.get("camera_timestamp_mode")),
        "pose_timestamp_mode": d.get("pose_timestamp_mode"),
        "camera_t0_us": d.get("camera_t0_us"),
        "ref_ts_us": ref.get("timestamp_us"),
        "ref_payload_ts_us": ref.get("payload_timestamp_us"),
        "images": len(imgs),
        "null_ids": sum(1 for x in ids if not x),
        "sample": [
            (i.get("camera_label"), i.get("temporal_index"), i.get("frame_id"),
             i.get("frame_timestamp_us"), i.get("frame_arrival_us"))
            for i in imgs[:3]
        ],
        "frame_ids": [i.get("frame_id") for i in imgs],
        "labels": [i.get("camera_label") for i in imgs],
    }


def pts_analysis(frame_ids: list, labels: list) -> None:
    """按相机分组，检查 PTS 是否单调、步长是否≈9000 ticks(100ms) → 验证 90 kHz 时基。"""
    per: dict[str, list[int]] = {}
    for label, fid in zip(labels, frame_ids):
        if not fid or ":" not in str(fid):
            continue
        _cam, _, pts = str(fid).partition(":")
        try:
            per.setdefault(str(label), []).append(int(pts))
        except ValueError:
            continue
    if not per:
        print("  没有任何可解析的 cam<ID>:<PTS>，无法做时基判断")
        return
    for label, pts_list in sorted(per.items()):
        deltas = [pts_list[i + 1] - pts_list[i] for i in range(len(pts_list) - 1)]
        mono = all(d >= 0 for d in deltas)
        ms = [round(d / 90.0, 1) for d in deltas]
        print(
            f"  {label}: pts={pts_list} 步长={deltas} ticks ≈{ms} ms  "
            f"单调={'是' if mono else '否'}"
        )
    all_deltas = [d for v in per.values() for d in
                  (v[i + 1] - v[i] for i in range(len(v) - 1))]
    if all_deltas:
        avg = sum(all_deltas) / len(all_deltas)
        print(
            f"  平均步长 {avg:.1f} ticks ≈ {avg / 90:.1f} ms "
            f"（若≈9000 ticks/100 ms，说明时基就是 90 kHz，线性换算系数正确）"
        )


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    if os.path.isdir(arg):
        packages = [arg]
    else:
        packages = newest(int(arg))
    if not packages:
        print("没有找到推理包")
        return 1
    total_null = 0
    total_imgs = 0
    for pkg in packages:
        s = summarize(pkg)
        total_null += s["null_ids"]
        total_imgs += s["images"]
        print(f"\n包 {s['pkg']}")
        print(f"  camera_clock          = {s['camera_clock']}")
        print(f"  pose_timestamp_mode   = {s['pose_timestamp_mode']}")
        print(f"  camera_t0_us          = {s['camera_t0_us']}   (≈{(s['camera_t0_us'] or 0)/1e6:.3f} s)")
        print(f"  reference_pose.ts_us  = {s['ref_ts_us']}   (≈{(s['ref_ts_us'] or 0)/1e6:.3f} s)")
        print(f"  payload_timestamp_us  = {s['ref_payload_ts_us']}")
        print(f"  图片数={s['images']}  缺 frame_id 的帧数={s['null_ids']}")
        for label, t, fid, ts, arr in s["sample"]:
            print(f"    {label} t={t} frame_id={fid} ts_us={ts} arrival_us={arr}")
        pts_analysis(s["frame_ids"], s["labels"])
    print(f"\n合计：{len(packages)} 个包 / {total_imgs} 帧，缺 frame_id 共 {total_null} 帧")
    if total_imgs:
        print(f"缺帧率 = {total_null / total_imgs * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
