#!/usr/bin/env python3
"""Send synthetic vehicle_state JSON to backend UDP 7001 for inference test."""

import json
import socket
import time


HOST = "127.0.0.1"
PORT = 7001
FPS = 10.0


def build(ts: float) -> bytes:
    msg = {
        "type": "vehicle_state",
        "timestamp": ts,
        "pose": {
            "latitude": 28.25,
            "longitude": 113.16,
            "height": 20.0,
            "azimuth": 180.0,
            "vNorth": 0.0,
            "vEast": 0.0,
        },
        "chassis": {"cur_spd": 0.0, "yaw_rate": 0.0},
    }
    return json.dumps(msg).encode()


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    end = time.time() + 120
    n = 0
    while time.time() < end:
        sock.sendto(build(time.time()), (HOST, PORT))
        n += 1
        time.sleep(1.0 / FPS)
    print(f"sent {n} mock vehicle_state packets", flush=True)


if __name__ == "__main__":
    main()
