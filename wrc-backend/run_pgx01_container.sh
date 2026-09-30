#!/usr/bin/env bash
set -euo pipefail

# PGX01 container launcher. The existing run_backend_real.sh targets an older
# x86 server; this script leaves it and the production config untouched.
ACTION="${1:-}"
ROOT="${DATA_ROOT:-/home/xz-01/data/user_data/wanghaohao/701}"
IMAGE="${IMAGE_NAME:-alpamayo-pgx:1.0}"
CONTAINER_ROOT=/home/xz-01/data/user_data/wanghaohao/701
LIVE_NAME=alpamayo-pgx01-backend
TEST_NAME=alpamayo-pgx01-test

usage() {
  echo "usage: $0 {test|start|status|logs|stop} [live|test]" >&2
  exit 2
}

check_files() {
  test "$(uname -m)" = aarch64 || { echo 'PGX01 requires ARM64' >&2; exit 1; }
  for dir in alpamayo1.5 model_weights model_assets wrc-backend; do
    test -d "$ROOT/$dir" || { echo "missing $ROOT/$dir" >&2; exit 1; }
  done
  test -f "$ROOT/model_weights/config.json" || { echo 'checkpoint config missing' >&2; exit 1; }
  test -d "$ROOT/model_assets/Qwen3-VL-2B-Instruct" || { echo 'processor assets missing' >&2; exit 1; }
  docker image inspect "$IMAGE" >/dev/null
}

container_name() {
  case "${1:-live}" in
    live) echo "$LIVE_NAME" ;;
    test) echo "$TEST_NAME" ;;
    *) usage ;;
  esac
}

start_container() {
  local kind="$1" name config
  name="$(container_name "$kind")"
  if docker container inspect "$name" >/dev/null 2>&1; then
    echo "container $name already exists; inspect it before starting another" >&2
    exit 1
  fi
  check_files
  if [ "$kind" = test ]; then
    config="$ROOT/wrc-backend/config/pgx01-test.json"
  else
    config="$ROOT/wrc-backend/config/pgx01.json"
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
    -w "$CONTAINER_ROOT/wrc-backend" \
    "$IMAGE" python app.py
  echo "started $name ($kind); use '$0 status $kind' and '$0 logs $kind'"
}

case "$ACTION" in
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
  start)
    start_container live
    ;;
  status)
    name="$(container_name "${2:-live}")"
    docker ps --filter "name=^/${name}$" --format '{{.Names}} {{.Status}} {{.Image}}'
    ;;
  logs)
    docker logs --tail 100 "$(container_name "${2:-live}")"
    ;;
  stop)
    name="$(container_name "${2:-live}")"
    docker stop "$name"
    docker rm "$name"
    ;;
  *) usage ;;
esac
