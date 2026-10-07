"""键算法与模糊化单元测试（含与既有录制文件的兼容性验证）。"""

import json
import os
from pathlib import Path

import pytest

from llm_mock.canon import (
    canon,
    exact_key,
    fuzzy_key,
    key_candidates,
    normalize_messages,
    sha256_hex,
    strip_volatile,
)

EXISTING_CACHE = Path(os.environ.get("LLM_MOCK_STORE", ""))


def reference_key(messages, tools, model, temperature, max_tokens):
    """既有录制文件使用的键算法（照抄定义，用来交叉验证实现）。"""
    return sha256_hex(
        canon(
            {
                "m": messages,
                "t": tools,
                "model": model,
                "temp": temperature,
                "mx": max_tokens,
            }
        )
    )


def test_exact_key_matches_reference_formula():
    messages = [{"role": "user", "content": "你好"}]
    tools = [{"type": "function", "function": {"name": "bash"}}]
    assert exact_key(messages, tools, "deepseek-v4-pro", 0, 4096) == reference_key(
        messages, tools, "deepseek-v4-pro", 0, 4096
    )


@pytest.mark.skipif(
    not (os.environ.get("LLM_MOCK_STORE") and EXISTING_CACHE.exists()),
    reason="需要 LLM_MOCK_STORE 指向一份真实录制文件",
)
def test_exact_key_reproduces_existing_record():
    """用真实录制的 request 重算 key，必须与录制里的 key 完全一致。"""
    with EXISTING_CACHE.open(encoding="utf-8") as fh:
        checked = 0
        for line in fh:
            rec = json.loads(line)
            req = rec["request"]
            got = exact_key(
                req.get("messages"),
                req.get("tools"),
                rec.get("model"),
                req.get("temperature"),
                req.get("max_tokens"),
            )
            assert got == rec["key"], f"第 {checked} 条 key 不一致"
            checked += 1
            if checked >= 20:
                break
    assert checked == 20


def test_normalize_messages_masks_tool_output():
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "file1\nfile2  耗时 0.31s"},
    ]
    norm = normalize_messages(messages)
    assert norm[2]["content"] == "<TOOL_OUTPUT>"
    assert "tool_call_id" not in norm[2]
    assert "id" not in norm[1]["tool_calls"][0]
    # 未屏蔽时二者一致
    assert normalize_messages(messages, mask_tool_output=False) == messages


def test_fuzzy_key_ignores_tool_output_and_ids():
    msgs_a = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "Thu Oct 5 12:00:01 UTC 2026\nreal 0m1.234s"},
    ]
    msgs_b = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_z", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_z", "content": "Thu Oct 5 12:09:59 UTC 2026\nreal 0m9.876s"},
    ]
    assert exact_key(msgs_a, None, "m", 0, 4096) != exact_key(msgs_b, None, "m", 0, 4096)
    assert fuzzy_key(msgs_a, None, "m", 0, 4096) == fuzzy_key(msgs_b, None, "m", 0, 4096)


def test_fuzzy_key_distinguishes_different_dialogue_shape():
    a = [{"role": "tool", "content": "x"}]
    b = [{"role": "tool", "content": "y"}, {"role": "tool", "content": "z"}]
    assert fuzzy_key(a, None, "m", 0, 4096) != fuzzy_key(b, None, "m", 0, 4096)


def test_volatile_normalization():
    text = "ran in real 0m1.234s at 2026-10-05T12:00:01Z under /tmp/pytest-of-root/pytest-3"
    out = strip_volatile(text)
    assert "1.234" not in out and "2026-10-05T12:00:01Z" not in out
    a = [{"role": "user", "content": "耗时 0.31s"}]
    b = [{"role": "user", "content": "耗时 1.99s"}]
    assert fuzzy_key(a, None, "m", 0, 4096) != fuzzy_key(b, None, "m", 0, 4096)
    assert fuzzy_key(a, None, "m", 0, 4096, volatile=True) == fuzzy_key(
        b, None, "m", 0, 4096, volatile=True
    )


def test_key_candidates_fill_defaults():
    messages = [{"role": "user", "content": "x"}]
    keys = key_candidates(messages, None, "m", None, None, default_temperature=0, default_max_tokens=4096)
    assert exact_key(messages, None, "m", 0, 4096) in keys
    assert exact_key(messages, None, "m", None, None) in keys
    assert len(keys) == len(set(keys))
