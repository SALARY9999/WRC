from __future__ import annotations

import importlib.util
import socket
import struct
import sys
import threading
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "direct_push_receiver.py"
SPEC = importlib.util.spec_from_file_location("direct_push_receiver", MODULE_PATH)
assert SPEC and SPEC.loader
cimg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cimg
SPEC.loader.exec_module(cimg)


def fake_jpeg(width: int = 640, height: int = 480) -> bytes:
    sof = b"\xff\xc0" + struct.pack("!H", 11) + bytes([8])
    sof += struct.pack("!HH", height, width) + b"\x01\x01\x11\x00"
    return b"\xff\xd8" + sof + b"\xff\xd9"


def packet(group_ns=1_700_000_000_000_000_000) -> bytes:
    jpeg = fake_jpeg()
    records = [cimg.IMAGE_HEADER.pack(cam, 0, 10, group_ns + cam, len(jpeg)) + jpeg
               for cam in range(1, 5)]
    payload = b"".join(records)
    return cimg.PACKET_HEADER.pack(b"CIMG", 1, 4, group_ns, len(payload)) + payload


class FragmentingSocket:
    def __init__(self, data: bytes, chunk_size: int) -> None:
        self.data, self.chunk_size = data, chunk_size

    def recv(self, size: int) -> bytes:
        count = min(size, self.chunk_size)
        result, self.data = self.data[:count], self.data[count:]
        return result


class CimgReceiverTests(unittest.TestCase):
    def test_fragmented_tcp_packet_is_parsed(self) -> None:
        group = cimg.read_group(FragmentingSocket(packet(), 3), 1_000_000, 100_000)
        self.assertEqual(4, len(group.images))
        self.assertEqual((640, 480), (group.images[0].width, group.images[0].height))

    def test_sticky_packets_are_read_one_at_a_time(self) -> None:
        stream = FragmentingSocket(packet() + packet(1_700_000_001_000_000_000), 10_000)
        first = cimg.read_group(stream, 1_000_000, 100_000)
        second = cimg.read_group(stream, 1_000_000, 100_000)
        self.assertLess(first.group_time_ns, second.group_time_ns)

    def test_rejects_oversize_before_payload_read(self) -> None:
        header = cimg.PACKET_HEADER.pack(b"CIMG", 1, 4, 1, 999_999)
        with self.assertRaisesRegex(cimg.ProtocolError, "payload_size"):
            cimg.read_group(FragmentingSocket(header, 100), 1000, 1000)

    def test_rejects_wrong_version_and_magic(self) -> None:
        for header in (cimg.PACKET_HEADER.pack(b"NOPE", 1, 4, 1, 80),
                       cimg.PACKET_HEADER.pack(b"CIMG", 2, 4, 1, 80)):
            with self.assertRaises(cimg.ProtocolError):
                cimg.read_group(FragmentingSocket(header, 100), 1000, 1000)

    def test_sequence_checks_and_wrap(self) -> None:
        self.assertEqual(("连续", 0), cimg.sequence_status(10, 11))
        self.assertEqual(("丢帧(+2)", 2), cimg.sequence_status(10, 13))
        self.assertEqual(("重复", 0), cimg.sequence_status(10, 10))
        self.assertEqual(("乱序", 0), cimg.sequence_status(10, 9))
        self.assertEqual(("连续", 0), cimg.sequence_status(0xFFFFFFFF, 0))

    def test_connection_logs_observation_and_summary(self) -> None:
        server_side, client_side = socket.socketpair()
        logs: list[str] = []
        server = cimg.CimgValidationServer(timeout=1, logger=logs.append)
        worker = threading.Thread(target=server.handle_connection,
                                  args=(server_side, ("local", 1234)))
        worker.start()
        client_side.sendall(packet())
        client_side.close()
        worker.join(2)
        joined = "\n".join(logs)
        for expected in ("连接已建立", "组#1", "format=JPEG", "size=640x480",
                         "missing=0", "连接已关闭"):
            self.assertIn(expected, joined)

    def test_save_group_writes_a_bounded_visual_sample(self) -> None:
        group = cimg.read_group(FragmentingSocket(packet(), 7), 1_000_000, 100_000)
        with tempfile.TemporaryDirectory() as directory:
            server = cimg.CimgValidationServer(save_dir=directory, save_groups=1)
            server.save_group(group, 1)
            files = sorted(Path(directory).glob("*.jpg"))
            self.assertEqual(4, len(files))
            self.assertEqual(fake_jpeg(), files[0].read_bytes())
            server.save_group(group, 2)
            self.assertEqual(4, len(list(Path(directory).glob("*.jpg"))))


if __name__ == "__main__":
    unittest.main()
