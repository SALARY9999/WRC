#!/usr/bin/env python3
"""Read-only replay of the trajectory crop over recorded inference packages.

Re-runs ``trim_waypoints_to_latest_pose`` against the pose the crop actually
consumed and checks the recorded send.  It opens nothing but the package JSON
files, writes nothing, and never touches the network:

    python tools/replay_trajectory_postprocess.py /data/.../infer_packages

Per package it runs two checks:

``idempotent``
             Re-crops the *already sent* waypoints.  A trajectory that was
             correctly cropped once must not lose more points when cropped
             again against the same pose.  It cannot see a crop that was wrong
             the first time.

``send``     Rebuilds the WGS84 waypoints from the re-cropped array and
             compares them with what the package says was sent, within the
             rounding the package applies.

Which pose the crop is replayed against
    ``trajectory_postprocess_input`` (new packages) records the pose and the
    reference pose the crop *actually consumed*; the tool uses those.  When
    the section is present it is authoritative, **including when
    its ``latest_pose`` is null** - a crop that ran without a pose must be
    replayed without one, because falling back to the record's ``pose_latest``
    would replay a crop that never happened.  Packages without the section fall
    back to ``pose_latest`` / ``reference_pose`` exactly as before, so
    historical data keeps replaying.  The note on each line says which was used:
    ``crop-input`` or ``legacy-pose_latest``.

Exit status is 0 only when every package passes every check it can run.
Differences are printed one per line and are never tolerated: the whole point
is to surface them for a human.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from trajectory.coordinates import local_xy_to_latlon  # noqa: E402
from trajectory.postprocess import trim_waypoints_to_latest_pose  # noqa: E402
from vehicle.models import (  # noqa: E402
    GlobalPose,
    PoseSample,
    vehicle_azimuth_for_compass,
)

# The package rounds waypoints to 8 decimals before writing them.
_ROUND_PLACES = 8


def _reference_pose(payload: dict) -> PoseSample:
    return PoseSample(
        int(payload["timestamp_us"]),
        float(payload["latitude"]),
        float(payload["longitude"]),
        float(payload["altitude"]),
        float(payload["heading_deg"]),
    )


def _global_pose(payload) -> GlobalPose | None:
    if not isinstance(payload, dict):
        return None
    heading = float(payload.get("heading_deg") or 0.0)
    return GlobalPose(
        timestamp_us=int(payload["timestamp_us"]),
        latitude=float(payload["latitude"]),
        longitude=float(payload["longitude"]),
        altitude=float(payload["altitude"]),
        v_north=0.0,
        v_east=0.0,
        v_up=0.0,
        roll_deg=0.0,
        pitch_deg=0.0,
        azimuth_deg=vehicle_azimuth_for_compass(heading),
        payload_timestamp_us=(
            int(payload["payload_timestamp_us"])
            if payload.get("payload_timestamp_us") is not None
            else None
        ),
        arrival_us=(
            int(payload["arrival_us"])
            if payload.get("arrival_us") is not None
            else None
        ),
    )


def _crop_inputs(record: dict, notes: list[str]):
    """The pose / reference pose / latency the crop consumed.

    Prefers the recorded ``trajectory_postprocess_input``.  When that section
    exists it is authoritative even if ``latest_pose`` is null, because the
    record's ``pose_latest`` is a *later* snapshot taken when the package was
    written, after the crop had already run.  Only packages predating the
    section fall back to the top-level keys.
    """
    section = record.get("trajectory_postprocess_input")
    if not isinstance(section, dict):
        notes.append("legacy-pose_latest")
        return (
            _global_pose(record.get("pose_latest")),
            record.get("reference_pose"),
            record.get("inference_latency_ms"),
        )
    notes.append("crop-input")
    if section.get("latest_pose") != record.get("pose_latest"):
        # Expected and not a problem: the vehicle moved between the crop and
        # the write.  Recorded so a reader knows the two are not comparable.
        notes.append("pose-moved")
    return (
        _global_pose(section.get("latest_pose")),
        section.get("reference_pose") or record.get("reference_pose"),
        section.get("inference_latency_ms", record.get("inference_latency_ms")),
    )


def _as_trajectory(rows) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 2:
        # The package stores xyz; a legacy two-column record is padded so the
        # crop sees the same shape it would have seen live.
        array = np.column_stack((array, np.zeros(len(array))))
    return array


def _check_send(
    positions: np.ndarray, record: dict, ref: PoseSample
) -> list[str]:
    sent = record.get("output_waypoints") or []
    if len(sent) != len(positions):
        return [
            f"send: recorded {len(sent)} waypoints but the crop yields "
            f"{len(positions)}"
        ]
    if not len(positions):
        return []
    lat, lon = local_xy_to_latlon(
        positions[:, 0], positions[:, 1],
        ref.latitude, ref.longitude, ref.heading_deg,
    )
    differences = []
    for index, (la, lo) in enumerate(sent):
        if not math.isclose(lat[index], la, abs_tol=10 ** -_ROUND_PLACES):
            differences.append(
                f"send: waypoint {index} latitude {lat[index]!r} != recorded {la!r}"
            )
        if not math.isclose(lon[index], lo, abs_tol=10 ** -_ROUND_PLACES):
            differences.append(
                f"send: waypoint {index} longitude {lon[index]!r} != recorded {lo!r}"
            )
        if len(differences) >= 5:
            differences.append("send: ... further differences suppressed")
            break
    return differences


def replay_package(path: Path) -> tuple[list[str], list[str]]:
    """Returns (failures, notes) for one package."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: unreadable package: {exc}"], []

    failures: list[str] = []
    notes: list[str] = []
    latest, ref_payload, _latency = _crop_inputs(record, notes)
    try:
        ref = _reference_pose(ref_payload)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"{path}: reference_pose unusable: {exc}"], []
    if latest is None:
        notes.append("no pose from the crop; it will fall back")

    sent_rows = record.get("output_waypoints_local_xyz") or []
    if sent_rows:
        sent = _as_trajectory(sent_rows)
        reference = trim_waypoints_to_latest_pose(sent, latest, ref)
        if reference.applied and len(sent) and reference.removed_waypoint_count:
            failures.append(
                f"{path}: idempotent: re-cropping the sent waypoints removed "
                f"{reference.removed_waypoint_count} more of {len(sent)}"
            )
        for difference in _check_send(sent, record, ref):
            failures.append(f"{path}: {difference}")
        notes.append("send-checked")

    recorded_count = record.get("num_waypoints")
    if recorded_count is not None and len(sent_rows) != int(recorded_count):
        failures.append(
            f"{path}: num_waypoints={recorded_count} but "
            f"output_waypoints_local_xyz holds {len(sent_rows)}"
        )
    return failures, notes


def collect_packages(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(target.glob("*/infer.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("packages", type=Path,
                        help="package root directory, or a single infer.json")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write a machine-readable summary here")
    parser.add_argument("--quiet", action="store_true",
                        help="print only failures and the summary")
    args = parser.parse_args(argv)

    if not args.packages.exists():
        print(f"no such path: {args.packages}", file=sys.stderr)
        return 2
    packages = collect_packages(args.packages)
    if not packages:
        print(f"no infer.json found under {args.packages}", file=sys.stderr)
        return 2

    all_failures: list[str] = []
    tally = {
        "send-checked": 0,
        "crop-input": 0, "legacy-pose_latest": 0,
    }
    for path in packages:
        failures, notes = replay_package(path)
        all_failures.extend(failures)
        for note in notes:
            if note in tally:
                tally[note] += 1
        if not args.quiet:
            status = "FAIL" if failures else "ok"
            print(f"[{status}] {path.parent.name} ({', '.join(notes) or '-'})")

    summary = {
        "packages": len(packages),
        "send_checked": tally["send-checked"],
        # Which pose the crop was replayed against: the recorded one, or the
        # pre-Stage-2 fallback.
        "crop_input": tally["crop-input"],
        "legacy_pose_latest": tally["legacy-pose_latest"],
        "failures": len(all_failures),
    }
    print(
        f"\n{len(packages)} package(s): "
        f"send-checked={tally['send-checked']} "
        f"crop-input={tally['crop-input']} "
        f"legacy-pose_latest={tally['legacy-pose_latest']} "
        f"failures={len(all_failures)}"
    )
    for failure in all_failures:
        print(f"  {failure}")
    if args.json is not None:
        args.json.write_text(
            json.dumps({"summary": summary, "failures": all_failures}, indent=2),
            encoding="utf-8",
        )
    return 1 if all_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
