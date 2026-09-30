#!/usr/bin/env python3
"""CIMG four-camera JPEG receiver for first-stage link validation only."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Callable

MAGIC = b"CIMG"
PROTOCOL_VERSION = 1
EXPECTED_IMAGE_COUNT = 4
PACKET_HEADER = struct.Struct("!4sHHQI")
IMAGE_HEADER = struct.Struct("!HHIQI")
DEFAULT_MAX_PAYLOAD = 64 * 1024 * 1024
DEFAULT_MAX_JPEG = 16 * 1024 * 1024


class ProtocolError(ValueError):
    """The peer sent a packet that cannot be safely parsed."""


class TruncatedRead(EOFError):
    def __init__(self, expected: int, received: int) -> None:
        super().__init__(f"需要 {expected} 字节，实际只收到 {received} 字节")
        self.received = received


@dataclasses.dataclass(frozen=True)
class ImageRecord:
    camera_id: int
    sequence: int
    capture_time_ns: int
    jpeg_size: int
    width: int
    height: int
    image_format: str = "JPEG"
    jpeg_data: bytes = dataclasses.field(default=b"", repr=False, compare=False)


@dataclasses.dataclass(frozen=True)
class ImageGroup:
    group_time_ns: int
    payload_size: int
    images: tuple[ImageRecord, ...]


def recv_exact(conn: socket.socket, size: int) -> bytes:
    """Read exactly size bytes, handling normal TCP fragmentation."""
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise TruncatedRead(size, len(data))
        data.extend(chunk)
    return bytes(data)


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Validate JPEG framing and return (width, height) without extra packages."""
    if len(data) < 4 or data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
        raise ProtocolError("图片不是完整 JPEG（SOI/EOI 标记无效）")
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
           0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    pos = 2
    while pos < len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker == 0xDA:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            continue
        if pos + 2 > len(data):
            break
        segment_len = struct.unpack_from("!H", data, pos)[0]
        if segment_len < 2 or pos + segment_len > len(data):
            raise ProtocolError("JPEG 段长度无效")
        if marker in sof:
            if segment_len < 7:
                raise ProtocolError("JPEG SOF 段过短")
            height, width = struct.unpack_from("!HH", data, pos + 3)
            if width <= 0 or height <= 0:
                raise ProtocolError("JPEG 图片尺寸无效")
            return width, height
        pos += segment_len
    raise ProtocolError("JPEG 中未找到尺寸信息")


def parse_payload(payload: bytes, image_count: int, max_jpeg: int) -> tuple[ImageRecord, ...]:
    records: list[ImageRecord] = []
    offset = 0
    seen: set[int] = set()
    for index in range(image_count):
        if len(payload) - offset < IMAGE_HEADER.size:
            raise ProtocolError(f"第 {index + 1} 条图片记录头不完整")
        camera_id, reserved, sequence, capture_ns, jpeg_size = IMAGE_HEADER.unpack_from(payload, offset)
        offset += IMAGE_HEADER.size
        if reserved != 0:
            raise ProtocolError(f"相机 {camera_id} 保留字段不是 0")
        if camera_id not in range(1, EXPECTED_IMAGE_COUNT + 1):
            raise ProtocolError(f"非法相机编号 {camera_id}")
        if camera_id in seen:
            raise ProtocolError(f"相机编号 {camera_id} 重复")
        if jpeg_size <= 0 or jpeg_size > max_jpeg:
            raise ProtocolError(f"相机 {camera_id} JPEG 长度非法: {jpeg_size}")
        if jpeg_size > len(payload) - offset:
            raise ProtocolError(
                f"相机 {camera_id} JPEG 不完整: 声明 {jpeg_size} 字节，剩余 {len(payload) - offset} 字节"
            )
        jpeg = payload[offset:offset + jpeg_size]
        offset += jpeg_size
        width, height = jpeg_dimensions(jpeg)
        records.append(
            ImageRecord(camera_id, sequence, capture_ns, jpeg_size, width, height, "JPEG", jpeg)
        )
        seen.add(camera_id)
    if offset != len(payload):
        raise ProtocolError(f"负载尾部存在 {len(payload) - offset} 个多余字节")
    if seen != set(range(1, EXPECTED_IMAGE_COUNT + 1)):
        raise ProtocolError(f"相机集合不完整: {sorted(seen)}")
    return tuple(records)


def read_group(conn: socket.socket, max_payload: int, max_jpeg: int) -> ImageGroup:
    raw_header = recv_exact(conn, PACKET_HEADER.size)
    magic, version, image_count, group_ns, payload_size = PACKET_HEADER.unpack(raw_header)
    if magic != MAGIC:
        raise ProtocolError(f"包标识错误: {magic!r}，期望 b'CIMG'")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"不支持协议版本 {version}，当前只接受版本 1")
    if image_count != EXPECTED_IMAGE_COUNT:
        raise ProtocolError(f"图片数量为 {image_count}，协议要求固定为 4")
    minimum = image_count * IMAGE_HEADER.size
    if payload_size < minimum or payload_size > max_payload:
        raise ProtocolError(f"payload_size 非法: {payload_size}（允许 {minimum}..{max_payload}）")
    payload = recv_exact(conn, payload_size)
    return ImageGroup(group_ns, payload_size, parse_payload(payload, image_count, max_jpeg))


def utc_text(timestamp_ns: int) -> str:
    try:
        value = dt.datetime.fromtimestamp(timestamp_ns / 1e9, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return f"invalid({timestamp_ns})"
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sequence_status(previous: int | None, current: int) -> tuple[str, int]:
    """Return status and missing count for uint32 sequence numbers."""
    if previous is None:
        return "首次", 0
    delta = (current - previous) & 0xFFFFFFFF
    if delta == 0:
        return "重复", 0
    if delta == 1:
        return "连续", 0
    if delta < 0x80000000:
        return f"丢帧(+{delta - 1})", delta - 1
    return "乱序", 0


class CimgValidationServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 30106, *,
                 max_payload: int = DEFAULT_MAX_PAYLOAD, max_jpeg: int = DEFAULT_MAX_JPEG,
                 timeout: float = 15.0, save_dir: str | None = None, save_groups: int = 1,
                 logger: Callable[[str], None] | None = None) -> None:
        self.host, self.port = host, port
        self.max_payload, self.max_jpeg, self.timeout = max_payload, max_jpeg, timeout
        self.save_dir = Path(save_dir).expanduser().resolve() if save_dir else None
        self.save_groups = save_groups
        self._logger = logger or (lambda message: print(message, flush=True))

    def log(self, message: str) -> None:
        now = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
        self._logger(f"[{now}] {message}")

    def save_group(self, group: ImageGroup, group_number: int) -> None:
        """Save a bounded number of received groups for visual inspection."""
        if self.save_dir is None or group_number > self.save_groups:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        for image in group.images:
            name = f"group{group_number:03d}_cam{image.camera_id}_seq{image.sequence}.jpg"
            path = self.save_dir / name
            path.write_bytes(image.jpeg_data)
            saved.append(str(path))
        self.log(f"组#{group_number} 样图已保存: {', '.join(saved)}")

    def handle_connection(self, conn: socket.socket, peer: tuple[str, int]) -> None:
        conn.settimeout(self.timeout)
        started = window_started = time.monotonic()
        window_groups = group_number = total_images = total_bytes = 0
        missing_total = out_of_order_total = 0
        previous_sequences: dict[int, int] = {}
        previous_group_ns: int | None = None
        self.log(f"连接已建立 peer={peer[0]}:{peer[1]}")
        try:
            while True:
                group = read_group(conn, self.max_payload, self.max_jpeg)
                received_ns, now_mono = time.time_ns(), time.monotonic()
                group_number += 1
                window_groups += 1
                total_images += len(group.images)
                total_bytes += PACKET_HEADER.size + group.payload_size
                elapsed = max(now_mono - started, 1e-9)
                window_elapsed = max(now_mono - window_started, 1e-9)
                avg_fps, recent_fps = group_number / elapsed, window_groups / window_elapsed
                if window_elapsed >= 5.0:
                    window_started, window_groups = now_mono, 0
                latency_ms = (received_ns - group.group_time_ns) / 1e6
                capture_times = [image.capture_time_ns for image in group.images]
                span_ms = (max(capture_times) - min(capture_times)) / 1e6
                group_order = "正常"
                if previous_group_ns is not None and group.group_time_ns < previous_group_ns:
                    group_order = "乱序"
                    out_of_order_total += 1
                previous_group_ns = max(previous_group_ns or 0, group.group_time_ns)
                self.log(
                    f"组#{group_number} 接收完成 recv={utc_text(received_ns)} "
                    f"group_ts={utc_text(group.group_time_ns)} latency={latency_ms:.1f}ms "
                    f"fps={recent_fps:.2f}(近窗)/{avg_fps:.2f}(平均) "
                    f"payload={group.payload_size}B capture_span={span_ms:.3f}ms group_order={group_order}"
                )
                self.save_group(group, group_number)
                for image in group.images:
                    status, missing = sequence_status(previous_sequences.get(image.camera_id), image.sequence)
                    out_of_order_total += int(status == "乱序")
                    missing_total += missing
                    if status not in {"乱序", "重复"}:
                        previous_sequences[image.camera_id] = image.sequence
                    image_latency_ms = (received_ns - image.capture_time_ns) / 1e6
                    self.log(
                        f"  cam={image.camera_id} frame={image.sequence} seq={status} "
                        f"capture_ts={utc_text(image.capture_time_ns)} latency={image_latency_ms:.1f}ms "
                        f"format={image.image_format} size={image.width}x{image.height} bytes={image.jpeg_size}"
                    )
        except TruncatedRead as exc:
            self.log(f"连接中断，丢弃未完成数据包: {exc}" if exc.received else "对端正常断开")
        except socket.timeout:
            self.log(f"接收超时（{self.timeout:g}s 无完整数据），关闭连接")
        except (ProtocolError, OSError) as exc:
            self.log(f"协议/网络错误，关闭连接: {exc}")
        finally:
            conn.close()
            elapsed = max(time.monotonic() - started, 1e-9)
            self.log(
                f"连接已关闭 peer={peer[0]}:{peer[1]} groups={group_number} images={total_images} "
                f"bytes={total_bytes} avg_fps={group_number / elapsed:.2f} "
                f"missing={missing_total} out_of_order={out_of_order_total}"
            )

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(8)
            self.log(f"CIMG 验证服务正在监听 {self.host}:{self.port} "
                     f"max_payload={self.max_payload} max_jpeg={self.max_jpeg}")
            if self.save_dir is not None:
                self.log(f"将保存前 {self.save_groups} 组样图到 {self.save_dir}")
            try:
                while True:
                    conn, peer = server.accept()
                    threading.Thread(target=self.handle_connection, args=(conn, peer), daemon=True,
                                     name=f"cimg-{peer[0]}:{peer[1]}").start()
            except KeyboardInterrupt:
                self.log("收到停止信号，服务退出")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="CIMG 四路 JPEG 接收与链路验证服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=30106, help="监听端口")
    parser.add_argument("--max-payload", type=positive_int, default=DEFAULT_MAX_PAYLOAD)
    parser.add_argument("--max-jpeg", type=positive_int, default=DEFAULT_MAX_JPEG)
    parser.add_argument("--timeout", type=float, default=15.0, help="连接读取超时（秒）")
    parser.add_argument("--save-dir", help="保存样图的目录；不指定则不保存")
    parser.add_argument("--save-groups", type=positive_int, default=1,
                        help="最多保存前几组图片（默认 1 组）")
    args = parser.parse_args()
    CimgValidationServer(args.host, args.port, max_payload=args.max_payload,
                         max_jpeg=args.max_jpeg, timeout=args.timeout,
                         save_dir=args.save_dir, save_groups=args.save_groups).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
