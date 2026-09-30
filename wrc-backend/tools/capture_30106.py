#!/usr/bin/env python3
"""把无人车发到 TCP 30106 的数据原样录下来（只记录，不解析）。

两种模式：
  1) listen（默认）：自己占住 30106 收数据。要求该端口没有被后端占用。
  2) tap：用 tcpdump 旁路抓同一份流量（链路照常跑）：
         python3 capture_30106.py --tap 60

用法：
    python3 capture_30106.py              # 一直录，Ctrl-C 停
    python3 capture_30106.py 60           # 录 60 秒
    python3 capture_30106.py --tap 60     # 旁路抓 60 秒
    python3 capture_30106.py --port 30199 # 换端口（本地测试用）

输出目录默认 /data/users_data/wanghaohao/capture/30106/<时间戳>/
  - *.bin   车端发的原始字节，一个连接一个文件
  - *.log   连接时间 / 断开时间 / 字节数
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time

DEFAULT_OUT = (
    r"F:\wrc\capture_30106"
    if sys.platform == "win32"
    else "/data/users_data/wanghaohao/capture/30106"
)


def log(path: str, line: str) -> None:
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def listen_mode(host: str, port: int, seconds: float, outdir: str) -> int:
    run_dir = os.path.join(outdir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    logfile = os.path.join(run_dir, "capture.log")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(4)
    log(logfile, f"[{time.strftime('%F %T')}] listening on {host}:{port}")
    log(logfile, f"[{time.strftime('%F %T')}] 输出目录: {run_dir}")

    deadline = time.time() + seconds if seconds > 0 else None
    index = 0
    try:
        while deadline is None or time.time() < deadline:
            srv.settimeout(1.0)
            try:
                conn, peer = srv.accept()
            except socket.timeout:
                continue

            index += 1
            name = f"{index:03d}_{peer[0]}-{peer[1]}_{port}.bin"
            path = os.path.join(run_dir, name)
            log(logfile, f"[{time.strftime('%F %T')}] #{index} 连接来自 {peer[0]}:{peer[1]}")

            total = 0
            started = time.time()
            last_report = started
            conn.settimeout(1.0)
            with open(path, "wb") as fh:
                while deadline is None or time.time() < deadline:
                    try:
                        chunk = conn.recv(262144)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    fh.write(chunk)
                    fh.flush()
                    total += len(chunk)
                    now = time.time()
                    if now - last_report >= 5.0:
                        mbps = total * 8 / 1e6 / max(now - started, 1e-6)
                        log(logfile,
                            f"    收到 {total} 字节 ({total/1e6:.2f} MB), "
                            f"{mbps:.2f} Mbps, 正在写入 {name}")
                        last_report = now
            conn.close()

            elapsed = max(time.time() - started, 1e-6)
            log(logfile,
                f"[{time.strftime('%F %T')}] #{index} 断开, 共 {total} 字节 "
                f"({total/1e6:.2f} MB / {elapsed:.1f}s), 文件: {path}")
    except KeyboardInterrupt:
        log(logfile, f"[{time.strftime('%F %T')}] 手动停止")
    finally:
        srv.close()
    log(logfile, f"结束。原始数据在 {run_dir}")
    return 0


def tap_mode(port: int, seconds: float, outdir: str) -> int:
    """不动拆分器，用 tcpdump 把 30106 的流量旁路存成 pcap。"""
    run_dir = os.path.join(outdir, "tap_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    pcap = os.path.join(run_dir, f"port{port}.pcap")

    cmd = ["tcpdump", "-i", "any", "-nn", "-s0", "-w", pcap, f"tcp port {port}"]
    if os.geteuid() == 0:
        prefix: list[str] = []
    else:
        prefix = ["sudo", "-n"]

    print(f"[{time.strftime('%F %T')}] 旁路抓包 {seconds:.0f}s -> {pcap}", flush=True)
    try:
        subprocess.run(prefix + ["timeout", str(int(seconds) + 1)] + cmd,
                       check=False, timeout=seconds + 30)
    except FileNotFoundError:
        print("找不到 tcpdump/timeout，请在服务器上运行", file=sys.stderr)
        return 1

    size = os.path.getsize(pcap) if os.path.exists(pcap) else 0
    # 顺便报一下车端连接和收到的字节数（不解析内容，只看有没有数据）
    ss = subprocess.run(["ss", "-tni", "state", "established",
                         f"( sport = :{port} )"],
                        capture_output=True, text=True)
    print(f"抓包结束: {pcap} ({size} 字节)")
    print("30106 连接状态:")
    print("\n".join(ss.stdout.splitlines()[:4]) or "  （当前没有已建立连接）")
    if size <= 24:
        print("pcap 是空的 —— 这段时间车端没有向 30106 发数据。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seconds", nargs="?", type=float, default=0,
                        help="录多少秒，0 或省略 = 一直录到 Ctrl-C")
    parser.add_argument("--port", type=int, default=30106)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--outdir", default=DEFAULT_OUT)
    parser.add_argument("--tap", action="store_true",
                        help="不停拆分器，用 tcpdump 旁路抓（链路不受影响）")
    args = parser.parse_args()

    if args.tap:
        if args.seconds <= 0:
            args.seconds = 60
        return tap_mode(args.port, args.seconds, args.outdir)
    return listen_mode(args.host, args.port, args.seconds, args.outdir)


if __name__ == "__main__":
    sys.exit(main())
