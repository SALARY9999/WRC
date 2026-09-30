#!/usr/bin/env bash
set -euo pipefail

# PGX01 backend container.
#
#   bash run_backend_real.sh            live listeners, waypoints to the vehicle
#   bash run_backend_real.sh worker     this machine only: model HTTP on port 8092
#   bash run_backend_real.sh test       loopback smoke test, then remove the container
#   bash run_backend_real.sh status     [live|test|worker]
#   bash run_backend_real.sh logs       [live|test|worker]
#   bash run_backend_real.sh stop       [live|test|worker]
#
# PGX01 start reads config/pgx01.json and dispatches to
#   http://192.168.50.3:8092
#   http://192.168.50.4:8092
#   http://192.168.50.5:8092
# Run `worker` on those three machines first. They do not receive the vehicle
# and they do not send waypoints.
#
# Live listeners:
#   CIMG TCP   0.0.0.0:30106          vehicle -> this host
#   pose UDP   0.0.0.0:7001           vehicle -> this host
#   route TCP  0.0.0.0:30006          vehicle -> this host
#   waypoints  168.130.182.10:7010    this host -> vehicle
#
# config/pgx01.json still says result_max_age_ms=2500. Inference on this GPU
# takes about 2.8–5.5 s, so that limit drops every result before a datagram
# is sent. Live start overrides the two age limits to 10000 ms in the process
# environment and does not edit the JSON file. Takeover (control=1) is still
# decided from planned-route deviation, not from ALPAMAYO_CONTROL_FLAG.

ACTION="${1:-start}"
KIND="${2:-live}"
ROOT="${DATA_ROOT:-/home/xz-01/data/user_data/wanghaohao/701}"
IMAGE="${IMAGE_NAME:-alpamayo-pgx:1.0}"
ARCHIVE="${IMAGE_ARCHIVE:-$ROOT/offline/alpamayo-pgx-1.0.tar.zst}"
CONTAINER_ROOT="$ROOT"
LIVE_NAME=alpamayo-pgx01-backend
TEST_NAME=alpamayo-pgx01-test
WORKER_NAME=alpamayo-pgx-worker

usage() {
    echo "usage: $0 {start|worker|test|status|logs|stop} [live|test|worker]" >&2
    exit 2
}

if ! docker info >/dev/null 2>&1; then
    if [ "${PGX_DOCKER_REEXEC:-0}" = 1 ]; then
        echo "docker socket is not accessible" >&2
        exit 1
    fi
    if getent group docker | grep -qw "$(id -un)"; then
        export PGX_DOCKER_REEXEC=1
        exec sg docker -c "bash $(printf '%q ' "$0" "$@")"
    fi
    echo "docker socket is not accessible and $(id -un) is not in the docker group" >&2
    exit 1
fi

container_name() {
    case "$1" in
        live) echo "$LIVE_NAME" ;;
        test) echo "$TEST_NAME" ;;
        worker) echo "$WORKER_NAME" ;;
        *) usage ;;
    esac
}

ensure_runtime() {
    test "$(uname -m)" = aarch64 || { echo "PGX01 requires ARM64" >&2; exit 1; }
    for dir in alpamayo1.5 model_weights model_assets wrc-backend; do
        test -d "$ROOT/$dir" || { echo "missing $ROOT/$dir" >&2; exit 1; }
    done
    test -f "$ROOT/model_weights/config.json" || { echo "checkpoint config missing" >&2; exit 1; }
    test -d "$ROOT/model_assets/Qwen3-VL-2B-Instruct" || { echo "processor assets missing" >&2; exit 1; }
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        test -f "$ARCHIVE" || { echo "missing image $IMAGE and archive $ARCHIVE" >&2; exit 1; }
        echo "importing $IMAGE from $ARCHIVE"
        bash "$ROOT/pgx-deploy/import-image.sh" "$ARCHIVE"
        docker image inspect "$IMAGE" >/dev/null
    fi
}

ports_free() {
    local port
    for port in "$@"; do
        if ss -ltn 2>/dev/null | grep -E ":${port}[[:space:]]" >/dev/null; then
            echo "port $port is already in use" >&2
            exit 1
        fi
    done
}

start_container() {
    local kind="$1" name config
    local -a age_env=() cmd=(python app.py)
    name="$(container_name "$kind")"
    if docker container inspect "$name" >/dev/null 2>&1; then
        echo "container $name already exists; inspect it before starting another" >&2
        exit 1
    fi
    ensure_runtime
    if [ "$kind" = test ]; then
        config="$ROOT/wrc-backend/config/pgx01-test.json"
        ports_free 31106 17001 18090
    elif [ "$kind" = worker ]; then
        config="$ROOT/wrc-backend/config/pgx01.json"
        ports_free 8092
        cmd=(python -m inference.http_worker --host 0.0.0.0 --port 8092)
        echo "network worker listening on 0.0.0.0:8092"
    else
        config="$ROOT/wrc-backend/config/pgx01.json"
        ports_free 30106 7001 30006 8090
        age_env+=(
            -e "ALPAMAYO_RESULT_MAX_AGE_MS=${ALPAMAYO_RESULT_MAX_AGE_MS:-10000}"
            -e "ALPAMAYO_MAX_CAPTURE_LAG_MS=${ALPAMAYO_MAX_CAPTURE_LAG_MS:-10000}"
            -e "ALPAMAYO_NETWORK_INFER_TIMEOUT_S=${ALPAMAYO_NETWORK_INFER_TIMEOUT_S:-10}"
        )
        echo "live backend: cimg=0.0.0.0:30106 pose=0.0.0.0:7001 route=0.0.0.0:30006 waypoint=168.130.182.10:7010"
        echo "network workers: 192.168.50.3:8092 192.168.50.4:8092 192.168.50.5:8092"
        echo "result_max_age_ms=${ALPAMAYO_RESULT_MAX_AGE_MS:-10000} max_capture_lag_ms=${ALPAMAYO_MAX_CAPTURE_LAG_MS:-10000} network_infer_timeout_s=${ALPAMAYO_NETWORK_INFER_TIMEOUT_S:-10}"
    fi
    test -f "$config" || { echo "missing $config" >&2; exit 1; }
    mkdir -p "$ROOT/pgx-runtime/hf_home" "$ROOT/pgx-runtime/snapshots" "$ROOT/pgx-runtime/infer_packages" "$ROOT/pgx-runtime/vehicle_trace"

    docker run -d \
        --name "$name" --gpus all --ipc=host --network host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --user "$(id -u):$(id -g)" \
        -v "$ROOT/alpamayo1.5:$CONTAINER_ROOT/alpamayo1.5:ro" \
        -v "$ROOT/model_weights:$CONTAINER_ROOT/model_weights:ro" \
        -v "$ROOT/model_assets:$CONTAINER_ROOT/model_assets:ro" \
        -v "$ROOT/wrc-backend:$CONTAINER_ROOT/wrc-backend:ro" \
        -v "$ROOT/pgx-runtime:$CONTAINER_ROOT/pgx-runtime" \
        -e "PYTHONPATH=$CONTAINER_ROOT/alpamayo1.5/src:$CONTAINER_ROOT/wrc-backend" \
        -e "ALPAMAYO_CONFIG=$CONTAINER_ROOT/wrc-backend/config/$(basename "$config")" \
        -e "ALPAMAYO_CHECKPOINT=$CONTAINER_ROOT/model_weights" \
        -e "ALPAMAYO_PROCESSOR_PATH=$CONTAINER_ROOT/model_assets/Qwen3-VL-2B-Instruct" \
        -e ALPAMAYO_MOCK_MODE=0 -e ALPAMAYO_POSE_MODE=udp \
        -e ALPAMAYO_CIMG_CAMERA_INDICES=1,2,0,4 \
        -e ALPAMAYO_CONTROL_FLAG=0 \
        -e ALPAMAYO_WAYPOINT_MAGIC=0x514D3031 -e ALPAMAYO_VEHICLE_ID=1 \
        -e "ALPAMAYO_SNAPSHOT_DIR=$CONTAINER_ROOT/pgx-runtime/snapshots" \
        -e "ALPAMAYO_INFER_PACKAGE_DIR=$CONTAINER_ROOT/pgx-runtime/infer_packages" \
        -e "ALPAMAYO_VEHICLE_TRACE_DIR=$CONTAINER_ROOT/pgx-runtime/vehicle_trace" \
        -e "HF_HOME=$CONTAINER_ROOT/pgx-runtime/hf_home" \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e HOME=/tmp \
        "${age_env[@]}" \
        -w "$CONTAINER_ROOT/wrc-backend" \
        "$IMAGE" "${cmd[@]}"
    echo "started $name ($kind)"
}

case "$ACTION" in
    start)
        start_container live
        ;;
    worker)
        start_container worker
        ;;
    test)
        start_container test
        cleanup_test() {
            local result=$?
            if [ "$result" -eq 0 ]; then
                docker stop "$TEST_NAME" >/dev/null
                docker rm "$TEST_NAME" >/dev/null
            else
                docker logs --tail 80 "$TEST_NAME" 2>&1 || true
                echo "test failed; $TEST_NAME was kept for inspection (stop with '$0 stop test')" >&2
            fi
        }
        trap cleanup_test EXIT
        docker exec "$TEST_NAME" python "$CONTAINER_ROOT/wrc-backend/tools/pgx01_backend_smoke.py" \
            --http-port 18090 --cimg-port 31106 --pose-port 17001 --timeout 300
        ;;
    status)
        name="$(container_name "$KIND")"
        docker ps --filter "name=^/${name}$" --format '{{.Names}} {{.Status}} {{.Image}}'
        ;;
    logs)
        docker logs --tail 100 "$(container_name "$KIND")"
        ;;
    stop)
        name="$(container_name "$KIND")"
        docker stop "$name"
        docker rm "$name"
        ;;
    *)
        usage
        ;;
esac
