import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_mock import Config, MockService, make_server  # noqa: E402
from llm_mock.testing import start_fake_upstream  # noqa: E402


@pytest.fixture
def fake_upstream():
    httpd, state, url = start_fake_upstream()
    yield state, url
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def make_proxy(tmp_path, fake_upstream):
    state, upstream_url = fake_upstream
    started = []

    def _make(mode="hybrid", name="store.jsonl", **kw):
        cfg = Config(
            mode=mode,
            store_path=str(tmp_path / name),
            upstream_url=upstream_url,
            replay_latency="none",
            host="127.0.0.1",
            port=0,
            **kw,
        )
        svc = MockService(cfg)
        httpd = make_server(cfg, svc)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        started.append(httpd)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        return base, svc, state

    yield _make
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


def post_json(base, path, payload, headers=None, timeout=30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def get_json(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read())


def user_msg(repo="astropy/astropy", commit="d16bfe05a744", extra=""):
    return {
        "role": "user",
        "content": f"仓库 {repo} @ {commit}\n\nissue：\n{extra or 'fix the thing'}",
    }


def base_payload(extra=""):
    return {
        "model": "fake-model",
        "messages": [
            {"role": "system", "content": "you are a fixer"},
            user_msg(extra=extra),
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }


def parse_sse(raw: bytes) -> dict:
    """把 SSE 字节流拼回 {content, tool_calls}。"""
    content = []
    tool_calls = {}
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["name"] = fn.get("name") or slot["name"]
                slot["arguments"] += fn.get("arguments") or ""
    return {"content": "".join(content), "tool_calls": [tool_calls[k] for k in sorted(tool_calls)]}
