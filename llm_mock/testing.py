"""内置的"假上游" —— 离线演示与端到端测试用，不产生任何真实 API 调用。

它做真上游会做的事：说 OpenAI Chat Completions 协议、支持 stream=true 的 SSE、
支持 tool_calls，并且**记录自己被打了几次**（用来证明 replay 模式真的 0 外呼）。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .openai_api import sse_event


class FakeUpstreamState:
    def __init__(self, volatile: bool = False, latency_ms: int = 5) -> None:
        self.lock = threading.Lock()
        self.calls = 0
        self.requests: list[dict] = []
        self.volatile = volatile
        self.latency_ms = latency_ms
        self.started = time.time()

    def next_call(self, payload: dict) -> int:
        with self.lock:
            self.calls += 1
            self.requests.append(payload)
            return self.calls

    def make_reply(self, payload: dict, n: int) -> dict:
        messages = payload.get("messages") or []
        last_user = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                last_user = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                break
        tools = payload.get("tools") or []
        use_tool = "USE_TOOL" in last_user and tools
        stamp = f" t={time.time():.6f}" if self.volatile else ""
        if use_tool:
            name = (tools[0].get("function") or {}).get("name") or "bash"
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_fake_{n}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps({"command": "echo hi"}),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {
                "role": "assistant",
                "content": f"fake-reply#{n} to={last_user[:48]!r}{stamp}",
            }
            finish = "stop"
        return {
            "id": f"chatcmpl-fake-{n}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or "fake-model",
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish}
            ],
            "usage": {
                "prompt_tokens": 100 + n,
                "completion_tokens": 20,
                "total_tokens": 120 + n,
            },
        }

    def make_stream(self, payload: dict, n: int, body: dict) -> list[bytes]:
        msg = body["choices"][0]["message"]
        out: list[bytes] = []
        base = {
            "id": body["id"],
            "object": "chat.completion.chunk",
            "created": body["created"],
            "model": body["model"],
        }
        out.append(
            sse_event({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        )
        content = msg.get("content") or ""
        for i in range(0, len(content), 12):
            out.append(
                sse_event(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content[i : i + 12]},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            )
        for i, tc in enumerate(msg.get("tool_calls") or []):
            out.append(
                sse_event(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": i,
                                            "id": tc["id"],
                                            "type": "function",
                                            "function": {"name": tc["function"]["name"], "arguments": ""},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            )
            args = tc["function"]["arguments"]
            for j in range(0, len(args), 6):
                out.append(
                    sse_event(
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {"index": i, "function": {"arguments": args[j : j + 6]}}
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                )
        out.append(
            sse_event(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": body["choices"][0]["finish_reason"],
                        }
                    ],
                }
            )
        )
        out.append(sse_event({**base, "choices": [], "usage": body["usage"]}))
        out.append(b"data: [DONE]\n\n")
        return out


class _FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: FakeUpstreamState

    def _send(self, status: int, body: bytes, ctype="application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/v1/models"):
            self._send(
                200,
                json.dumps(
                    {
                        "object": "list",
                        "data": [{"id": "fake-model", "object": "model"}],
                    }
                ).encode(),
            )
            return
        if self.path.startswith("/__stats"):
            self._send(200, json.dumps({"calls": self.state.calls}).encode())
            return
        self._send(404, b'{"error":{"message":"not found"}}')

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}
        n = self.state.next_call(payload)
        if self.state.latency_ms:
            time.sleep(self.state.latency_ms / 1000.0)
        body = self.state.make_reply(payload, n)
        if payload.get("stream"):
            chunks = self.state.make_stream(payload, n, body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for c in chunks:
                self.wfile.write(c)
                self.wfile.flush()
            return
        self._send(200, json.dumps(body, ensure_ascii=False).encode())

    def log_message(self, fmt, *args):  # noqa: A003
        pass


class _FakeServer(ThreadingHTTPServer):
    """与真代理一样放大 backlog：默认 5 在并发测试下会导致连接被拒。"""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256


def start_fake_upstream(
    host: str = "127.0.0.1", port: int = 0, volatile: bool = False, latency_ms: int = 5
) -> tuple[ThreadingHTTPServer, FakeUpstreamState, str]:
    state = FakeUpstreamState(volatile=volatile, latency_ms=latency_ms)
    handler = type("FakeHandler", (_FakeHandler,), {"state": state})
    httpd = _FakeServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://{host}:{httpd.server_address[1]}"
    return httpd, state, url


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="假的上游 OpenAI 服务（离线演示用）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--volatile", action="store_true", help="在回复里插入时间戳（模拟不可控信息）")
    p.add_argument("--latency-ms", type=int, default=5)
    args = p.parse_args(argv)
    httpd, _, url = start_fake_upstream(
        args.host, args.port, volatile=args.volatile, latency_ms=args.latency_ms
    )
    print(f"fake upstream 已启动: {url}  (volatile={args.volatile})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
