#!/usr/bin/env bash
# Local full-chain bring-up: CIMG on a free loopback port +
# mock pose -> Alpamayo inference -> a waypoint sink on a free UDP port.
#
# This script must not stop or bind the production listeners:
#   HTTP 8090, pose UDP 7001, CIMG TCP 30106, route TCP 30006.
# Waypoints are sent to the vehicle; this script does not bind 7010 or 17100.
# It chooses its own HTTP and waypoint ports, disables the route listener,
# and only signals processes it started itself.
#
# Run as: sudo -n bash tools/run_full_chain_test.sh
set -euo pipefail

TEST_ROOT=/data/users_data/wanghaohao/full_chain_test
BACKEND=/data/users_data/wanghaohao/701/alpamayo_infer_backend
PY="$BACKEND/../alpasim/.venv.bak/bin/python"
CHECKPOINT=/data/users_data/wanghaohao/701/model_weights/Alpamayo-1.5-10B
GPU=${ALPAMAYO_TEST_GPU:-1}
LOG="$TEST_ROOT/backend.log"

# Production endpoints.  A chosen test port that lands on one of these is
# refused rather than reused.
PRODUCTION_PORTS="8090 7001 30106 30006"

is_production_port() {
    local port="$1"
    local known
    for known in $PRODUCTION_PORTS; do
        if [ "$port" = "$known" ]; then
            return 0
        fi
    done
    return 1
}

# kind is STREAM or DGRAM.  The socket is closed before the caller binds, so
# this is a hint, not a reservation; the production-port check is what matters.
free_port() {
    local kind="$1"
    local port=""
    local attempt
    for attempt in 1 2 3 4 5 6 7 8; do
        port="$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_${kind}); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")"
        if ! is_production_port "$port"; then
            printf '%s\n' "$port"
            return 0
        fi
    done
    printf 'refusing: could not find a non-production %s port\n' "$kind" >&2
    exit 1
}

stop_own() {
    local pidfile="$1"
    if [ -f "$pidfile" ]; then
        local pid
        pid="$(cat "$pidfile" 2>/dev/null || true)"
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
}

mkdir -p "$TEST_ROOT/snapshots" "$TEST_ROOT/run"

# Only this test's previous processes.  Do not pkill uvicorn or anything
# bound to a production port: that pattern used to stop the vehicle backend.
stop_own "$TEST_ROOT/run/backend.pid"
stop_own "$TEST_ROOT/run/waypoint_sink.pid"
sleep 1

HTTP_PORT="$(free_port STREAM)"
CIMG_PORT="$(free_port STREAM)"
WAYPOINT_PORT="$(free_port DGRAM)"
if is_production_port "$HTTP_PORT" || is_production_port "$CIMG_PORT" || is_production_port "$WAYPOINT_PORT"; then
    printf 'refusing: selected a production port http=%s cimg=%s waypoint=%s\n' \
        "$HTTP_PORT" "$CIMG_PORT" "$WAYPOINT_PORT" >&2
    exit 1
fi

setsid nohup python3 /data/users_data/wanghaohao/port_sink.py "$WAYPOINT_PORT" \
    > "$TEST_ROOT/waypoint_sink.log" 2>&1 < /dev/null &
echo $! > "$TEST_ROOT/run/waypoint_sink.pid"

cd "$BACKEND"
# Freshness thresholds come from config/config.json.  Route TCP is off so this
# process does not bind production port 30006.  Pose is mock, so 7001 stays free.
env PYTHONPATH="$BACKEND:/data/users_data/wanghaohao/701/alpamayo1.5/src" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    ALPAMAYO_MOCK_MODE=1 \
    ALPAMAYO_POSE_MODE=mock \
    ALPAMAYO_CIMG_HOST=127.0.0.1 \
    ALPAMAYO_CIMG_PORT="$CIMG_PORT" \
    ALPAMAYO_CIMG_CAMERA_INDICES=1,2,0,4 \
    ALPAMAYO_SNAPSHOT_DIR="$TEST_ROOT/snapshots" \
    ALPAMAYO_SNAPSHOT_INTERVAL=2.0 \
    ALPAMAYO_INFER_PACKAGE_DIR="$TEST_ROOT/infer_packages" \
    ALPAMAYO_WAYPOINT_HOST=127.0.0.1 \
    ALPAMAYO_WAYPOINT_PORT="$WAYPOINT_PORT" \
    ALPAMAYO_ROUTE_TCP_ENABLED=false \
    ALPAMAYO_LOG_LEVEL=INFO \
    ALPAMAYO_CHECKPOINT="$CHECKPOINT" \
    ALPAMAYO_CONTROL_FLAG=0 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup "$PY" -m uvicorn app:app --host 127.0.0.1 --port "$HTTP_PORT" \
    > "$LOG" 2>&1 < /dev/null &
echo $! > "$TEST_ROOT/run/backend.pid"

echo "full-chain test started: http=127.0.0.1:${HTTP_PORT} cimg=127.0.0.1:${CIMG_PORT} waypoint=127.0.0.1:${WAYPOINT_PORT} log=$LOG"
