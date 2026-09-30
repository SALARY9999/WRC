"""Exercise the real PGX01 backend with loopback CIMG and pose packets.

Run only with config/pgx01-test.json. Its listeners and waypoint destination
are all loopback addresses, so no vehicle is contacted.
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import time
import urllib.error
import urllib.request

from PIL import Image

from camera.cimg import IMAGE_HEADER, MAGIC, PACKET_HEADER, PROTOCOL_VERSION


def health(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
        return json.load(response)


def summary(body: dict) -> str:
    pool = body.get("inference_pool") or {}
    return json.dumps({
        "pipeline_alive": body.get("pipeline_alive"),
        "inference_count": body.get("inference_count"),
        "last_error": body.get("last_error"),
        "pool_ready_workers": pool.get("ready_workers"),
        "pool_failed_workers": pool.get("failed_workers"),
        "pool_error": pool.get("error"),
        "inference_metrics": body.get("inference_metrics"),
    }, ensure_ascii=False)


def jpeg_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (576, 320), (127, 127, 127)).save(output, format="JPEG", quality=80)
    return output.getvalue()


def cimg_packet(sequence: int, capture_ns: int, jpeg: bytes) -> bytes:
    records = b"".join(
        IMAGE_HEADER.pack(camera_id, 0, sequence, capture_ns + camera_id * 1000, len(jpeg))
        + jpeg
        for camera_id in (1, 2, 3, 4)
    )
    return PACKET_HEADER.pack(MAGIC, PROTOCOL_VERSION, 4, capture_ns, len(records)) + records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-port", type=int, default=18090)
    parser.add_argument("--cimg-port", type=int, default=31106)
    parser.add_argument("--pose-port", type=int, default=17001)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--require-workers",
        type=int,
        default=1,
        help="Require this many distinct workers to complete an inference",
    )
    args = parser.parse_args()
    if args.require_workers < 1:
        parser.error("--require-workers must be at least 1")
    deadline = time.monotonic() + args.timeout

    last: dict = {}
    while time.monotonic() < deadline:
        try:
            last = health(args.http_port)
            pool = last.get("inference_pool") or {}
            if last.get("pipeline_alive") and pool.get("ready_workers", 0) >= 1:
                break
            if pool.get("failed_workers", 0):
                raise RuntimeError(f"model worker failed: {summary(last)}")
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    else:
        raise RuntimeError(f"backend/model did not become ready: {summary(last)}")

    baseline = int(last.get("inference_count", 0))
    jpeg = jpeg_bytes()
    pose = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with socket.create_connection(("127.0.0.1", args.cimg_port), timeout=5) as camera:
            camera.settimeout(5)
            sequence = 0
            while time.monotonic() < deadline:
                now_ns = time.time_ns()
                timestamp = now_ns / 1_000_000_000
                vehicle_state = {
                    "type": "vehicle_state",
                    "timestamp": timestamp,
                    "pose": {
                        "latitude": 31.1190589,
                        "longitude": 114.4639576 + sequence * 0.0000002,
                        "height": 0.0,
                        "vNorth": 0.0,
                        "vEast": 2.0,
                        "vUp": 0.0,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "azimuth": 90.0,
                    },
                }
                pose.sendto(
                    json.dumps(vehicle_state).encode("utf-8"),
                    ("127.0.0.1", args.pose_port),
                )
                camera.sendall(cimg_packet(sequence, now_ns - 50_000_000, jpeg))
                if sequence % 10 == 0:
                    last = health(args.http_port)
                    workers = (last.get("inference_pool") or {}).get("workers") or []
                    completed_workers = sum(
                        int(worker.get("infer_count") or 0) > 0 for worker in workers
                    )
                    if (
                        int(last.get("inference_count", 0)) > baseline
                        and completed_workers >= args.require_workers
                    ):
                        print(json.dumps({
                            "status": "ok",
                            "backend_inference_count": last["inference_count"],
                            "completed_workers": completed_workers,
                            "inference_pool": last.get("inference_pool"),
                            "inference_metrics": last.get("inference_metrics"),
                            "last_error": last.get("last_error"),
                        }, ensure_ascii=False))
                        return 0
                sequence += 1
                time.sleep(0.1)
    finally:
        pose.close()
    raise RuntimeError(f"no backend inference completed: {summary(last)}")


if __name__ == "__main__":
    raise SystemExit(main())
