"""The offline-test HTTP client, over a real socket, against a real request.

``tools/request_model_test.py`` is what ``wrc/run_package_test.py`` runs on the
other side of the link, and its request shape is a contract with whatever serves
``/infer``: a JSON POST carrying each camera's images as ``images_base64``.  The
unit tests pin the URL default; this file pins the *request*, because the URL
being right while the body is wrong is the same failure - an offline test that
returns nothing useful.

The server here is the standard library's, on a loopback ephemeral port, and it
answers with a fixed document.  Nothing else is stubbed: the client runs as the
subprocess it runs as in production, reads a real sample file and real image
bytes off disk, and its answer is checked byte for byte as well as by value.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
CLIENT = BACKEND_ROOT / "tools" / "request_model_test.py"
ANSWER = {"trajectories": [[[31.2305, 121.4738], [31.2310, 121.4742]]], "timing_ms": 12}
IMAGE_BYTES = b"\xff\xd8\xff\xe0 not-really-a-jpeg \x00\xff\xd9"
SAMPLE = {
    "nav_text": "follow the planned route",
    "cameras": [{"camera_index": 0, "timestamps_us": [1, 2, 3, 4], "image_paths": ["front_t0.jpg"]}],
    "ego_history": [],
}


class _Recorder(BaseHTTPRequestHandler):
    """Answers every POST with ``ANSWER`` and keeps what it was sent."""

    def do_POST(self):  # noqa: N802 - the name is the protocol
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        self.server.requests.append(
            {
                "path": self.path,
                "content_type": self.headers.get("Content-Type"),
                "body": body,
            }
        )
        payload = json.dumps(ANSWER).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep the test output readable
        return


class _Server:
    def __init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
        self.httpd.requests = []
        self.httpd.timeout = 5
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/infer"

    @property
    def requests(self) -> list:
        return self.httpd.requests

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class OfflineClientHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _Server()
        self.addCleanup(self.server.close)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.package = Path(self.directory.name)
        (self.package / "front_t0.jpg").write_bytes(IMAGE_BYTES)
        self.sample = self.package / "model_test_sample.json"
        self.sample.write_text(json.dumps(SAMPLE), encoding="utf-8")
        self.output = self.package / "model_result.json"

    def _run(self, *extra: str, env: dict | None = None) -> subprocess.CompletedProcess:
        merged = dict(os.environ)
        merged.pop("ALPAMAYO_PACKAGE_TEST_URL", None)
        merged.update(env or {})
        return subprocess.run(
            [sys.executable, str(CLIENT), str(self.sample), "--output", str(self.output), *extra],
            capture_output=True,
            text=True,
            timeout=60,
            env=merged,
        )

    def test_the_client_posts_the_documented_request_to_the_given_url(self) -> None:
        result = self._run("--url", self.server.url)
        self.assertEqual(0, result.returncode, result.stderr)
        # The answer was written through, unchanged.
        self.assertEqual(ANSWER, json.loads(self.output.read_text(encoding="utf-8")))
        # The URL printed by the client resolved to the URL it was given.
        self.assertEqual(str(self.output.resolve()), result.stdout.strip())

        self.assertEqual(1, len(self.server.requests))
        request = self.server.requests[0]
        self.assertEqual("/infer", request["path"])
        self.assertEqual("application/json", request["content_type"])
        body = json.loads(request["body"].decode("utf-8"))
        self.assertEqual(SAMPLE["nav_text"], body["nav_text"])
        camera = body["cameras"][0]
        self.assertEqual(0, camera["camera_index"])
        # The image path was replaced by its bytes, base64 encoded.
        self.assertNotIn("image_paths", camera)
        self.assertEqual(
            [base64.b64encode(IMAGE_BYTES).decode("ascii")], camera["images_base64"]
        )

    def test_the_configured_url_is_what_a_bare_invocation_uses(self) -> None:
        """No ``--url``: the target has to come from the configuration, and it
        has to arrive on the wire at that address."""
        result = self._run(env={"ALPAMAYO_PACKAGE_TEST_URL": self.server.url})
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(1, len(self.server.requests), "the configured URL was not used")
        self.assertEqual("/infer", self.server.requests[0]["path"])

    def test_the_shipped_default_points_at_the_offline_service(self) -> None:
        """The default is a value, not a guess: loopback 8091, the offline test
        service's port, with the ``/infer`` path the routes use."""
        from config.loader import DEFAULT_PACKAGE_TEST_URL

        self.assertEqual("http://127.0.0.1:8091/infer", DEFAULT_PACKAGE_TEST_URL)
        with patch.dict(os.environ, {"ALPAMAYO_PACKAGE_TEST_URL": ""}, clear=False):
            with patch.dict(os.environ, {"ALPAMAYO_CONFIG": ""}, clear=False):
                from config.loader import load_network_config

                self.assertEqual(
                    DEFAULT_PACKAGE_TEST_URL, load_network_config().package_test_url
                )

    def test_the_documented_offline_test_path_matches_the_configured_url(self) -> None:
        """The service and the tunnel behind ``package_test_url`` are started by
        hand from ``docs/MODEL_TEST.md``.  A command line cannot read the config
        document, so those two numbers can only be *compared* with it - and
        without this comparison they are the one hop in the chain nothing
        checks.  A drift here is a URL that stops answering with nothing to
        point at.
        """
        from urllib.parse import urlparse

        from config.loader import load_network_config

        url = urlparse(load_network_config().package_test_url)
        doc = (BACKEND_ROOT / "docs" / "MODEL_TEST.md").read_text(encoding="utf-8")
        launch = re.search(
            r"uvicorn\s+test_backend:app\s+--host\s+(\S+)\s+--port\s+(\d+)", doc
        )
        self.assertIsNotNone(launch, "MODEL_TEST.md no longer documents the launch")
        self.assertEqual(url.hostname, launch.group(1))
        self.assertEqual(url.port, int(launch.group(2)))
        tunnel = re.search(r"-L\s+(\d+):(\S+):(\d+)", doc)
        self.assertIsNotNone(tunnel, "MODEL_TEST.md no longer documents the tunnel")
        self.assertEqual(url.port, int(tunnel.group(1)))  # PC side of the forward
        self.assertEqual(url.hostname, tunnel.group(2))  # the server's loopback
        self.assertEqual(url.port, int(tunnel.group(3)))  # server side of the forward

    def test_a_server_that_refuses_is_reported_by_the_client(self) -> None:
        """A bad URL is a client error with a non-zero exit, not a silent one."""
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", 0))
            closed_port = probe.getsockname()[1]
        finally:
            probe.close()
        result = self._run("--url", f"http://127.0.0.1:{closed_port}/infer")
        self.assertNotEqual(0, result.returncode)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
