#!/usr/bin/env python3
"""Verify the UDP pose listener binds only the configured ports."""

import sys
import time

BACKEND = "/data/users_data/wanghaohao/701/alpamayo_infer_backend"
sys.path.insert(0, BACKEND)

import app  # noqa: E402


def main() -> int:
    listener = app.UdpPoseListener(app.PoseHistory())
    listener.start()
    time.sleep(1.0)
    print("bound=", [s.getsockname() for s in listener._sockets])
    listener.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
