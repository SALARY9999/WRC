"""HTTP client for one network inference worker.

Connect timeout and read timeout are separate: a machine that accepts the TCP
handshake and then never answers must hit the inference budget, not the
connect budget.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any

import http.client


class UrllibTransport:
    """Blocking ``GET /health`` and ``POST /infer`` against one worker URL."""

    def health(
        self,
        url: str,
        *,
        connect_timeout_s: float,
        read_timeout_s: float,
    ) -> dict[str, Any]:
        status, body = _request(
            "GET",
            _join(url, "/health"),
            None,
            connect_timeout_s,
            read_timeout_s,
        )
        if status != 200 or not isinstance(body, dict):
            raise ConnectionError(f"health HTTP {status}")
        return body

    def infer(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        connect_timeout_s: float,
        read_timeout_s: float,
    ) -> dict[str, Any]:
        status, body = _request(
            "POST",
            _join(url, "/infer"),
            payload,
            connect_timeout_s,
            read_timeout_s,
        )
        if status != 200 or not isinstance(body, dict):
            raise ConnectionError(f"infer HTTP {status}: {body}")
        return body


def _join(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _request(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    connect_timeout_s: float,
    read_timeout_s: float,
) -> tuple[int, Any]:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"unsupported worker URL {url}")
    host = parts.hostname
    if not host:
        raise ValueError(f"worker URL has no host: {url}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    connection_cls = (
        http.client.HTTPSConnection
        if parts.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_cls(host, port, timeout=float(connect_timeout_s))
    try:
        connection.connect()
        if connection.sock is not None:
            connection.sock.settimeout(float(read_timeout_s))
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        status = int(response.status)
    finally:
        connection.close()
    if not raw:
        return status, {}
    return status, json.loads(raw.decode("utf-8"))


__all__ = ["UrllibTransport"]
