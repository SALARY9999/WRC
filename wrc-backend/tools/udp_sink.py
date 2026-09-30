#!/usr/bin/env python3
"""Vehicle-side UDP receiver that prints waypoint packet summaries."""

import socket
import struct
import sys
import time


def waypoint_text(payload: bytes) -> str:
    if len(payload) < 14:
        return f"bytes={len(payload)}"
    magic, version, control, seq, vehicle_id, n = struct.unpack(
        "<IBBIHH", payload[:14]
    )
    if len(payload) < 14 + 16 * n:
        return f"truncated n={n} bytes={len(payload)}"
    first = struct.unpack("<dd", payload[14:30])
    return (
        f"magic=0x{magic:08X} ver={version} control={control} "
        f"seq={seq} vehicle={vehicle_id} n={n} first=({first[0]:.7f},{first[1]:.7f})"
    )


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7010
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(1.0)
    print(f"listening UDP 0.0.0.0:{port}", flush=True)
    total = 0
    last = time.monotonic()
    while True:
        try:
            data, _addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        total += 1
        now = time.monotonic()
        if now - last >= 1.0 or total == 1:
            print(f"pkts={total} {waypoint_text(data)}", flush=True)
            last = now


if __name__ == "__main__":
    raise SystemExit(main())
