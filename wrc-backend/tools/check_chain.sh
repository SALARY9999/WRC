#!/usr/bin/env bash
# One-shot health check for the whole chain (run on the server):
#   vehicle CIMG push -> 4 JPEG queues -> window -> inference -> waypoint
#
#   bash tools/check_chain.sh
#
# Prints one section per hop plus a verdict, so a test session can tell
# "vehicle is not pushing" from "decoder died" from "no pose" in ~5 seconds.
set -uo pipefail

BACKEND=/data/users_data/wanghaohao/701/alpamayo_infer_backend
BLOG=/data/users_data/wanghaohao/backend.log
HURL=http://127.0.0.1:8090/health
SURL=http://127.0.0.1:8090/status
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

ok()   { printf '  [ OK ] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; }

echo "=== 无人车链路自检 $(date '+%F %T') ==="

# --------------------------------------------------------------------------- #
echo
echo "[1/6] CIMG 图片源：车辆 -> 后端 TCP :30106"
listening=no
if ss -ltn 2>/dev/null | grep ':30106 ' >/dev/null; then
    listening=yes
    owner=$(ss -ltnp 2>/dev/null | awk '$4 ~ /:30106$/ {print $0}' | head -1)
    ok "后端 CIMG 端口 30106 已监听${owner:+：$owner}"
else
    bad "TCP 30106 没有监听 —— 后端 CIMG 接收器未启动"
fi
peer=$(ss -tn 2>/dev/null | awk '$4 ~ /:30106$/ {print $5}' | head -1)
if [ -n "$peer" ]; then
    ok "车辆已连接: $peer"
else
    bad "车辆未连接 30106 —— 车辆端没有推送 CIMG 图片"
fi

# --------------------------------------------------------------------------- #
echo
echo "[2/6] CIMG 四路图片（来自 /health，采样 2 s 算实时 fps）"
curl -s -m 3 "$HURL" -o "$TMP/health1.json" || true
curl -s -m 3 "$SURL" -o "$TMP/status1.json" || true
sleep 2
curl -s -m 3 "$HURL" -o "$TMP/health2.json" || true
curl -s -m 3 "$SURL" -o "$TMP/status2.json" || true

python3 - "$TMP/health1.json" "$TMP/health2.json" "$TMP/bad_cams" <<'PY'
import json, sys

def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

h1, h2 = load(sys.argv[1]), load(sys.argv[2])
if not h2:
    print("  [FAIL] 后端 /health 无响应（uvicorn 未启动，或仍在加载 10B 模型）")
    open(sys.argv[3], "w").close()
    raise SystemExit(0)

cimg = h2.get("camera_cimg") or {}
bad_cams = []
if not cimg:
    print("  [FAIL] /health 没有 camera_cimg 状态")
    bad_cams.append("link")
else:
    print(
        f"  CIMG: state={cimg.get('connection_state')} "
        f"peer={cimg.get('peer') or 'none'} queue={cimg.get('queue_depth')} "
        f"groups={cimg.get('groups_total')} errors={cimg.get('protocol_errors')}"
    )
    if not cimg.get("connected"):
        print("  [FAIL] CIMG 接收器已启动，但车辆尚未连接")
        bad_cams.append("link")

s1 = {str(r.get("camera_index")): r for r in (h1 or {}).get("camera_stats", {}).get("readers", [])}
s2 = {str(r.get("camera_index")): r for r in h2.get("camera_stats", {}).get("readers", [])}
if len(s2) != 4:
    print(f"  [FAIL] 相机状态数量应为 4，实际为 {len(s2)}")
    bad_cams.append(f"count={len(s2)}")
for camera_index, st in s2.items():
    prev = s1.get(camera_index, {}).get("frames_total", st["frames_total"])
    fps = (st["frames_total"] - prev) / 2.0
    age = st["frame_age_ms"]
    age_s = "n/a" if age is None else f"{age:.0f}ms"
    line = (
        f"  camera_index={camera_index}: {st['health']:<10s} "
        f"in={fps:5.1f} fps  frame_age={age_s:>7s}  "
        f"frames={st['frames_total']}  queue={st.get('queue_depth', 'n/a')}"
    )
    if st["health"] != "HEALTHY" or (age is not None and age > 1000):
        bad_cams.append(camera_index)
        print(line + "   <== 异常")
        if st.get("last_error"):
            print(f"         last_error: {st['last_error'][:140]}")
    else:
        print(line)
ws = h2.get("camera_stats", {})
print(
    f"  窗口: builds={ws.get('window_builds')}  "
    f"last_window_age={ws.get('last_window_age_ms') or 'n/a'} ms  "
    f"refusals={ws.get('refusals')}"
)
print(
    f"  推理: count={h2.get('inference_count')}  "
    f"last_error={h2.get('last_error')}"
)
with open(sys.argv[3], "w", encoding="utf-8") as fh:
    fh.write(",".join(bad_cams))
PY
bad_cams=$(cat "$TMP/bad_cams" 2>/dev/null || true)

# --------------------------------------------------------------------------- #
echo
echo "[3/6] 位姿（车辆 -> PGX UDP :7001）"
python3 - "$TMP/status1.json" "$TMP/status2.json" <<'PY'
import json, re, sys

def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

s1, s2 = load(sys.argv[1]), load(sys.argv[2])
if not s2:
    print("  [FAIL] /status 无响应")
    raise SystemExit(0)
rate = (s2.get("chassis_packets", 0) - (s1 or {}).get("chassis_packets", 0)) / 2.0
print(f"  时间轴口径: {s2.get('pose_timestamp_mode')}")
print(f"  位姿速率: {rate:.1f} 包/秒   累计: {s2.get('chassis_packets')}")
print(
    f"  车速: {s2.get('chassis_speed_mps')} m/s   "
    f"yaw_rate: {s2.get('chassis_yaw_rate_rad_s')} rad/s"
)
if rate < 1.0:
    print("  [FAIL] 没有位姿进来：确认车辆把 UDP 7001 发到 PGX 的局域网 IP")
PY

# --------------------------------------------------------------------------- #
echo
echo "[4/6] 推理 / 航点（后端日志）"
last_inf=$(grep "inference sent" "$BLOG" 2>/dev/null | tail -1)
last_out=$(grep "waypoint-out" "$BLOG" 2>/dev/null | tail -1)
last_pkg=$(grep "infer-package-saved" "$BLOG" 2>/dev/null | tail -1)
[ -n "$last_inf" ] && ok "${last_inf#*INFO \[app\] }" || warn "本日志内还没有成功推理"
[ -n "$last_out" ] && printf '  [ OK ] %s\n' "${last_out#*INFO \[app\] }"
[ -n "$last_pkg" ] && printf '  [ OK ] %s\n' "${last_pkg#*INFO \[app\] }"
errs=$(grep -c "pipeline cycle failed\|no camera window for" "$BLOG" 2>/dev/null || true)
if [ "${errs:-0}" -gt 0 ]; then
    warn "日志里有 ${errs} 条链路错误"
    grep "pipeline cycle failed\|no camera window for" "$BLOG" | tail -1 | cut -c1-200
fi

# --------------------------------------------------------------------------- #
echo
echo "[5/6] 下行（PGX -> 车辆 UDP :7010）"
if [ -n "$last_inf" ]; then
    ok "航点由后端直接发往 network.waypoint（车辆 UDP 7010）"
else
    warn "backend.log 里还没有 inference sent"
fi

# --------------------------------------------------------------------------- #
echo
echo "[6/6] 结论"
if [ -z "$peer" ]; then
    bad "车辆没有向 30106 推送 CIMG 图片 —— 当前无法形成相机窗口"
elif [ "$listening" = "no" ]; then
    bad "后端 CIMG 监听端口未就绪"
elif [ -n "$bad_cams" ]; then
    bad "CIMG 相机异常: $bad_cams（查看 backend.log 的 cimg 状态）"
else
    ok "CIMG 图片链路正常（4 路都在出帧）"
    if [ -n "$last_inf" ]; then
        ok "推理与航点都在产出"
    else
        warn "还没有推理产出：确认位姿是否有数据、经纬度是否为 0"
    fi
fi
echo
echo "提示：重启后端会断开车辆的 CIMG TCP 连接，车辆端需要具备自动重连。"
