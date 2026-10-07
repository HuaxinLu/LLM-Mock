"""OpenAI Chat Completions 协议的编解码（非流式 + SSE 流式）。

回放出来的响应必须是**合法 OpenAI 响应**，否则真实客户端（openai SDK /
LangChain / 自研 harness）不改一行代码就用不了。所以：
  * 存储里记的是归一化的 {content, tool_calls}
  * 对外回复时再组装成完整 chat.completion / chat.completion.chunk
  * 流式录制：一边把上游字节转发给客户端，一边用 SSECollector 拼回完整消息
  * 流式回放：用存储里的完整消息反向合成 SSE
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

SSE_DONE = b"data: [DONE]\n\n"


def sse_event(obj: Any) -> bytes:
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")


# --------------------------------------------------------------------------
# 上游响应 -> 归一化记录
# --------------------------------------------------------------------------
def normalize_completion(body: dict) -> tuple[dict, dict, str | None]:
    """OpenAI 非流式响应 -> (response, usage, model)"""
    choices = body.get("choices") or [{}]
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    tool_calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        tool_calls.append(
            {
                "id": tc.get("id") or "",
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "",
            }
        )
    resp = {
        "content": msg.get("content") or "",
        "tool_calls": tool_calls,
        "finish_reason": choice.get("finish_reason"),
        "role": msg.get("role") or "assistant",
    }
    return resp, (body.get("usage") or {}), body.get("model")


class SSECollector:
    """把上游 SSE 字节流拼回一条归一化消息。"""

    def __init__(self) -> None:
        self._buf = b""
        self._content: list[str] = []
        self._tool_calls: dict[int, dict] = {}
        self._finish: str | None = None
        self._usage: dict = {}
        self._model: str | None = None
        self._id: str | None = None
        self._created: int | None = None
        self._json_errors = 0

    # -- 解析 --------------------------------------------------------------
    def feed(self, data: bytes) -> None:
        self._buf += data
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._line(line)

    def close(self) -> None:
        if self._buf.strip():
            self._line(self._buf)
        self._buf = b""

    def _line(self, line: bytes) -> None:
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            chunk = json.loads(payload)
        except Exception:
            self._json_errors += 1
            return
        self._id = chunk.get("id") or self._id
        self._created = chunk.get("created") or self._created
        self._model = chunk.get("model") or self._model
        if isinstance(chunk.get("usage"), dict):
            self._usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str):
                self._content.append(piece)
            for tc in delta.get("tool_calls") or []:
                self._absorb_tool_call(tc)
            if choice.get("finish_reason"):
                self._finish = choice["finish_reason"]

    def _absorb_tool_call(self, tc: dict) -> None:
        idx = tc.get("index")
        if idx is None:
            idx = len(self._tool_calls)
        slot = self._tool_calls.setdefault(
            int(idx), {"id": "", "name": "", "arguments": ""}
        )
        if tc.get("id"):
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        if fn.get("arguments"):
            slot["arguments"] += fn["arguments"]

    # -- 输出 --------------------------------------------------------------
    def result(self) -> dict:
        ordered = [self._tool_calls[k] for k in sorted(self._tool_calls)]
        return {
            "content": "".join(self._content),
            "tool_calls": ordered,
            "finish_reason": self._finish,
            "role": "assistant",
        }

    @property
    def usage(self) -> dict:
        return self._usage

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def json_errors(self) -> int:
        return self._json_errors


# --------------------------------------------------------------------------
# 归一化记录 -> OpenAI 响应
# --------------------------------------------------------------------------
def build_message(response: dict) -> dict:
    tool_calls = []
    for i, tc in enumerate(response.get("tool_calls") or []):
        tool_calls.append(
            {
                "id": tc.get("id") or f"call_mock_{i}",
                "type": "function",
                "function": {
                    "name": tc.get("name") or "",
                    "arguments": tc.get("arguments") or "{}",
                },
            }
        )
    msg: dict[str, Any] = {
        "role": response.get("role") or "assistant",
        "content": response.get("content") or None,
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def build_completion(record: dict, model: str | None = None) -> dict:
    response = record.get("response") or {}
    finish = response.get("finish_reason")
    if not finish:
        finish = "tool_calls" if response.get("tool_calls") else "stop"
    created = int(record.get("ts") or time.time())
    return {
        "id": f"chatcmpl-mock-{(record.get('key') or 'nokey')[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": model or record.get("model") or "llm-mock",
        "mock": {
            "hit": True,
            "strategy": record.get("_strategy"),
            "key": record.get("key"),
            "ts": record.get("ts"),
        },
        "choices": [
            {
                "index": 0,
                "message": build_message(response),
                "finish_reason": finish,
            }
        ],
        "usage": record.get("usage") or {},
    }


def _chunks_for_text(text: str, pieces: int = 8) -> Iterator[str]:
    if not text:
        return
    size = max(1, len(text) // pieces)
    for i in range(0, len(text), size):
        yield text[i : i + size]


def synthesize_sse(record: dict, model: str | None = None) -> Iterator[bytes]:
    """用归一化记录合成 OpenAI 风格的 SSE 流。"""
    response = record.get("response") or {}
    cid = f"chatcmpl-mock-{(record.get('key') or 'nokey')[:24]}"
    created = int(record.get("ts") or time.time())
    mdl = model or record.get("model") or "llm-mock"

    def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> bytes:
        obj: dict[str, Any] = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": mdl,
            "choices": (
                [{"index": 0, "delta": delta, "finish_reason": finish}]
                if not usage
                else []
            ),
        }
        if usage:
            obj["usage"] = usage
        return sse_event(obj)

    yield chunk({"role": "assistant", "content": ""})
    for piece in _chunks_for_text(response.get("content") or ""):
        yield chunk({"content": piece})
    for i, tc in enumerate(response.get("tool_calls") or []):
        yield chunk(
            {
                "tool_calls": [
                    {
                        "index": i,
                        "id": tc.get("id") or f"call_mock_{i}",
                        "type": "function",
                        "function": {"name": tc.get("name") or "", "arguments": ""},
                    }
                ]
            }
        )
        for piece in _chunks_for_text(tc.get("arguments") or "", pieces=4):
            yield chunk(
                {
                    "tool_calls": [
                        {"index": i, "function": {"arguments": piece}}
                    ]
                }
            )
    finish = response.get("finish_reason")
    if not finish:
        finish = "tool_calls" if response.get("tool_calls") else "stop"
    yield chunk({}, finish_reason=finish)
    if record.get("usage"):
        yield chunk({}, usage=record["usage"])
    yield SSE_DONE


def error_body(kind: str, message: str, **extra: Any) -> dict:
    err: dict[str, Any] = {"message": message, "type": kind}
    err.update(extra)
    return {"error": err}
