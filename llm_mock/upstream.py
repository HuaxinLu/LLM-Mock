"""上游转发（纯 stdlib http.client，不引入任何依赖）。

两个要点：
  1. 强制 `Accept-Encoding: identity` —— 否则上游可能返回 gzip，录制时无法解析。
  2. 客户端 body **原样透传**（不改字段），只补/换 Authorization，
     这样任何 OpenAI 兼容参数（response_format、top_p、stream_options…）都不会丢。
"""

from __future__ import annotations

import json
from http.client import HTTPConnection, HTTPSConnection
from typing import Any, Iterator
from urllib.parse import urlsplit

HOP_BY_HOP = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "accept-encoding",
}


class Upstream:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 600.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        parts = urlsplit(self.base_url)
        if parts.scheme not in ("http", "https"):
            raise ValueError(f"upstream 必须是 http/https URL: {base_url}")
        self._scheme = parts.scheme
        self._host = parts.hostname or "127.0.0.1"
        self._port = parts.port or (443 if parts.scheme == "https" else 80)
        self._base_path = parts.path.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.calls = 0

    def _target(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self._base_path}{path}" or path

    def build_headers(self, client_headers: Any = None) -> dict[str, str]:
        headers: dict[str, str] = {"Accept-Encoding": "identity"}
        if client_headers is not None:
            for k, v in client_headers.items():
                if k.lower() in HOP_BY_HOP:
                    continue
                headers[k] = v
        headers.update(self.extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.setdefault("Content-Type", "application/json")
        return headers

    def open(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: Any = None,
        timeout: float | None = None,
    ) -> tuple[Any, Any]:
        """返回 (conn, resp)。调用方负责 conn.close()。

        刻意不用 contextmanager —— 流式响应要在生成器里跨 yield 持有连接，
        而 `__enter__` 失败后再调 `__exit__` 会踩 contextlib 的
        "generator didn't stop" 陷阱。
        """
        conn_cls = HTTPSConnection if self._scheme == "https" else HTTPConnection
        conn = conn_cls(self._host, self._port, timeout=timeout or self.timeout)
        try:
            conn.request(
                method,
                self._target(path),
                body=body,
                headers=self.build_headers(headers),
            )
            resp = conn.getresponse()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise
        self.calls += 1
        return conn, resp

    def post_json(
        self, path: str, body: bytes, headers: Any = None
    ) -> tuple[int, dict, bytes]:
        conn, resp = self.open("POST", path, body, headers)
        try:
            raw = resp.read()
            status = resp.status
        finally:
            conn.close()
        try:
            parsed = json.loads(raw) if raw else {}
        except Exception:
            parsed = {}
        return status, parsed, raw

    def iter_stream(self, resp: Any, chunk_size: int = 8192) -> Iterator[bytes]:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            yield chunk
