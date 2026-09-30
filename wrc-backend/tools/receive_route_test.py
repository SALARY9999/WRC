"""Bounded real-route reception test; cache is isolated from vehicle control."""
import argparse
import logging
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vehicle.planned_route import PlannedRouteTcpReceiver
from trajectory.takeover import PlannedRouteCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30006)
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if args.background:
        log_path = pathlib.Path(__file__).resolve().parent / "route_receive_test.log"
        with log_path.open("ab") as log:
            child = subprocess.Popen(
                [sys.executable, __file__, "--port", str(args.port),
                 "--duration", str(args.duration)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True,
            )
        print(f"test_pid={child.pid} log={log_path} duration={args.duration}s", flush=True)
        return
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    receiver = PlannedRouteTcpReceiver(PlannedRouteCache(), port=args.port)
    receiver.start()
    try:
        deadline = time.monotonic() + args.duration
        while receiver.is_alive() and time.monotonic() < deadline:
            time.sleep(.2)
    finally:
        receiver.stop()
        logging.info("route reception test ended: updates=%d error=%s",
                     receiver.updates, receiver.last_error)


if __name__ == "__main__":
    main()
