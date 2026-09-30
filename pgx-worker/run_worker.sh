#!/usr/bin/env bash
set -euo pipefail

# Standalone HTTP inference Worker for PGX02-04. No camera, scheduler or UDP.
ACTION="${1:-start}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DATA_ROOT="${DATA_ROOT:-$(dirname -- "$SCRIPT_DIR")}"
IMAGE_NAME="${IMAGE_NAME:-alpamayo-pgx:1.0}"
CONTAINER_NAME="${CONTAINER_NAME:-alpamayo-pgx-worker}"
WORKER_PORT="${WORKER_PORT:-8092}"

# Keep the paths used by the PGX01 checkpoint/config inside the container.
# The host-side DATA_ROOT may have a different username or location.
CONTAINER_ROOT=/home/xz-01/data/user_data/wanghaohao/701

usage() {
  echo "usage: bash $0 {start|check|status|logs|stop}" >&2
  exit 2
}

lan_ip() {
  if [[ -n "${WORKER_BIND_IP:-}" ]]; then
    printf '%s\n' "$WORKER_BIND_IP"
    return
  fi
  local addresses
  addresses="$(ip -o -4 addr show | awk '$4 ~ /^192[.]168[.]50[.]/ {split($4, a, "/"); print a[1]}')"
  if [[ "$(printf '%s\n' "$addresses" | sed '/^$/d' | wc -l)" -ne 1 ]]; then
    echo 'Set WORKER_BIND_IP to this PGX LAN address (for example 192.168.50.3).' >&2
    exit 1
  fi
  printf '%s\n' "$addresses"
}

check_inputs() {
  [[ "$(uname -m)" == aarch64 ]] || { echo 'ARM64/aarch64 host required' >&2; exit 1; }
  for path in \
    "$SCRIPT_DIR/inference/__init__.py" \
    "$SCRIPT_DIR/inference/http_worker.py" \
    "$SCRIPT_DIR/inference/alpamayo.py" \
    "$DATA_ROOT/alpamayo1.5/src/alpamayo1_5" \
    "$DATA_ROOT/model_weights/config.json" \
    "$DATA_ROOT/model_assets/Qwen3-VL-2B-Instruct"; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
  done
  docker image inspect "$IMAGE_NAME" >/dev/null || {
    echo "Docker image not imported: $IMAGE_NAME" >&2
    exit 1
  }
  [[ "$(docker image inspect --format '{{.Architecture}}' "$IMAGE_NAME")" == arm64 ]] || {
    echo "Docker image is not ARM64: $IMAGE_NAME" >&2
    exit 1
  }
}

case "$ACTION" in
  start)
    check_inputs
    if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
      echo "Container already exists: $CONTAINER_NAME (inspect or stop it first)" >&2
      exit 1
    fi
    BIND_IP="$(lan_ip)"
    docker run -d \
      --name "$CONTAINER_NAME" --gpus all --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -p "$BIND_IP:$WORKER_PORT:8092" \
      -v "$DATA_ROOT/alpamayo1.5:$CONTAINER_ROOT/alpamayo1.5:ro" \
      -v "$DATA_ROOT/model_weights:$CONTAINER_ROOT/model_weights:ro" \
      -v "$DATA_ROOT/model_assets:$CONTAINER_ROOT/model_assets:ro" \
      -v "$SCRIPT_DIR:$CONTAINER_ROOT/pgx-worker:ro" \
      -e "PYTHONPATH=$CONTAINER_ROOT/alpamayo1.5/src:$CONTAINER_ROOT/pgx-worker" \
      -e "ALPAMAYO_PROCESSOR_PATH=$CONTAINER_ROOT/model_assets/Qwen3-VL-2B-Instruct" \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -w "$CONTAINER_ROOT/pgx-worker" \
      "$IMAGE_NAME" python -m inference.http_worker \
        --host 0.0.0.0 --port 8092 --checkpoint "$CONTAINER_ROOT/model_weights"
    echo "Started $CONTAINER_NAME on $BIND_IP:$WORKER_PORT; model loading may take several minutes."
    echo "Check with: sudo bash $SCRIPT_DIR/run_worker.sh check"
    ;;
  check)
    docker exec "$CONTAINER_NAME" python -c \
      'import json,sys,urllib.request; d=json.load(urllib.request.urlopen("http://127.0.0.1:8092/health", timeout=5)); print(json.dumps(d)); sys.exit(0 if d.get("ready") else 1)'
    ;;
  status)
    docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}} {{.Status}} {{.Image}}'
    ;;
  logs)
    docker logs --tail 100 "$CONTAINER_NAME"
    ;;
  stop)
    docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 || {
      echo "Container does not exist: $CONTAINER_NAME" >&2
      exit 1
    }
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
    echo 'Stopped and removed the Worker container; model files and image were not removed.'
    ;;
  *) usage ;;
esac
