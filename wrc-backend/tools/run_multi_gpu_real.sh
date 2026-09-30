#!/bin/bash
# Real workers on the cards this test was given, nothing sent to a vehicle.
#
# Scope, stated so it is checkable: real Alpamayo 1.5 10B on four physical GPUs,
# read-only with respect to the checkpoint, no camera opened, no pose listener
# bound, no production script started or edited.  With --scenario pipeline the
# only traffic is waypoint UDP to a 127.0.0.1 ephemeral port this run opened
# itself.  Nothing here writes outside the staging directory.
set -u

STAGE="$(cd "$(dirname "$0")/.." && pwd)"
PY=/data/users_data/wanghaohao/701/alpasim/.venv.bak/bin/python
SRC=/data/users_data/wanghaohao/701/alpamayo1.5/src
CKPT=/data/users_data/wanghaohao/701/model_weights/Alpamayo-1.5-10B

SCENARIO="${1:?usage: run_multi_gpu_real.sh <scenario> <label> [extra args...]}"
LABEL="${2:?usage: run_multi_gpu_real.sh <scenario> <label> [extra args...]}"
shift 2

# The cards the live nvidia-smi showed free.  GPU 0 and 2 are held by other
# users' jobs and are deliberately absent: this test does not queue behind them
# and does not evict them.
DEVICES="1,3,4,5"
OUT="$STAGE/results/$LABEL.log"
SAMPLES="$STAGE/results/${LABEL}_gpu_samples.csv"

echo "host=$(hostname) user=$(whoami) devices=$DEVICES scenario=$SCENARIO"
echo "start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "nvidia-smi before:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# Sampler over *every* card, not just the four under test - the same reason the
# single-GPU probe watches all eight: a path that quietly took a context on
# someone else's card would look clean from the four it was pointed at.
(
  while true; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | tr '\n' ',' | sed "s/^/$(date '+%s'),/" >> "$SAMPLES"
    echo >> "$SAMPLES"
    sleep 5
  done
) &
SAMPLER=$!

# CUDA_VISIBLE_DEVICES is deliberately NOT set, and is explicitly unset below
# rather than inherited.  Every worker pins the physical index it was configured
# with and then loads on cuda:0; inheriting a visible-devices list would
# renumber the cards underneath --devices, so --devices 1 could land the model
# on physical GPU 0 while the log said 1.
sudo -n env -u CUDA_VISIBLE_DEVICES \
    PYTHONPATH="$STAGE:$SRC" \
    ALPAMAYO_MOCK_MODE=0 \
    ALPAMAYO_CHECKPOINT="$CKPT" \
    PYTHONDONTWRITEBYTECODE=1 \
    "$PY" "$STAGE/tools/multi_gpu_offline_test.py" \
        --scenario "$SCENARIO" \
        --devices "$DEVICES" \
        --real \
        --infer-ms 1138 \
        --seconds 60 \
        --rate-hz 10 \
        "$@" 2>&1 | tee "$OUT"
code=${PIPESTATUS[0]}

kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null

echo
echo "exit=$code"
echo "end=$(date '+%Y-%m-%d %H:%M:%S')"
echo "nvidia-smi after:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
exit "$code"
