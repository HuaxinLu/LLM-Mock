"""存储索引 / 会话解析 / 匹配策略单元测试。"""

import json

import pytest

from llm_mock.canon import exact_key, fuzzy_key
from llm_mock.matcher import Matcher
from llm_mock.session import SessionResolver, TurnTracker
from llm_mock.store import RecordStore


def msg_pair(task="task", tool_output="out"):
    messages = [
        {"role": "user", "content": f"仓库 astropy/astropy @ d16bfe05a744\n\n{task}"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": tool_output},
    ]
    return messages


def make_rec(messages, ts, response=None, model="m", temperature=0, max_tokens=4096, session="s"):
    return {
        "key": exact_key(messages, None, model, temperature, max_tokens),
        "fuzzy_key": fuzzy_key(messages, None, model, temperature, max_tokens),
        "ts": ts,
        "model": model,
        "session": session,
        "request": {"messages": messages, "tools": None,
                    "temperature": temperature, "max_tokens": max_tokens},
        "response": response or {"content": f"r{ts}", "tool_calls": []},
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "latency_ms": 1,
    }


def test_session_resolver_default_regex():
    r = SessionResolver()
    sid, source = r.resolve(None, msg_pair())
    assert sid == "astropy/astropy@d16bfe05a744"
    assert source == "regex"


def test_session_resolver_header_and_implicit():
    r = SessionResolver(implicit="fallback")
    assert r.resolve({"x-llm-mock-session": "abc"}, []) == ("abc", "header")
    assert r.resolve(None, [{"role": "user", "content": "无身份信息"}]) == (
        "fallback",
        "implicit",
    )
    assert r.resolve(None, [{"role": "user", "content": "无身份信息"}])[0] == "fallback"


def test_turn_tracker_increments_and_resets():
    t = TurnTracker()
    assert [t.next("s") for _ in range(3)] == [0, 1, 2]
    assert t.next("other") == 0
    t.reset("s")
    assert t.next("s") == 0
    # 显式请求头覆盖，并把计数器对齐到其后
    assert t.next("s", {"x-llm-mock-turn": "7"}) == 7
    assert t.next("s") == 8


def test_store_indexes_and_positional_order(tmp_path):
    p = tmp_path / "s.jsonl"
    store = RecordStore(p, load=False, session_of=SessionResolver().from_messages)
    r2 = make_rec(msg_pair("task-2"), ts=200, session=None)
    r1 = make_rec(msg_pair("task-1"), ts=100, session=None)
    store.add(r1)
    store.add(r2)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    seq = store.by_session["astropy/astropy@d16bfe05a744"]
    assert [r["ts"] for r in seq] == [100, 200]          # 按 ts 排序
    assert store.lookup_position("astropy/astropy@d16bfe05a744", 0)["ts"] == 100
    assert store.lookup_position("astropy/astropy@d16bfe05a744", 1)["ts"] == 200
    assert store.lookup_position("astropy/astropy@d16bfe05a744", 2) is None
    assert [r["turn"] for r in seq] == [0, 1]
    # 重新加载后索引一致
    store2 = RecordStore(p, session_of=SessionResolver().from_messages)
    assert len(store2.by_key) == 2
    seq2 = store2.by_session["astropy/astropy@d16bfe05a744"]
    assert len(seq2) == 2 and [r["turn"] for r in seq2] == [0, 1]


def test_store_computes_fuzzy_for_older_records(tmp_path):
    """旧记录没有 fuzzy_key/session/turn，加载时应就地补算。"""
    p = tmp_path / "older.jsonl"
    rec = {
        "key": exact_key(msg_pair("task"), None, "m", 0, 4096),
        "ts": 1,
        "model": "m",
        "request": {"messages": msg_pair("task"), "tools": None,
                    "temperature": 0, "max_tokens": 4096},
        "response": {"content": "hi", "tool_calls": []},
        "usage": {},
        "latency_ms": 3,
    }
    p.write_text(json.dumps(rec, ensure_ascii=False) + "\n")
    store = RecordStore(p, session_of=SessionResolver().from_messages)
    loaded = store.by_key[rec["key"]]
    assert loaded["fuzzy_key"] == fuzzy_key(msg_pair("task"), None, "m", 0, 4096)
    assert loaded["session"] == "astropy/astropy@d16bfe05a744"
    assert loaded["turn"] == 0


def test_matcher_order_prefers_exact(tmp_path):
    store = RecordStore(tmp_path / "s.jsonl", load=False)
    msgs = msg_pair()
    rec = make_rec(msgs, ts=1)
    store.index(rec)
    m = Matcher(store, order=("exact", "fuzzy", "positional"))
    hit = m.match(keys=[rec["key"]], fuzzy=rec["fuzzy_key"], session="s", turn=0)
    assert hit.strategy == "exact"


def test_matcher_fuzzy_and_positional(tmp_path):
    store = RecordStore(tmp_path / "s.jsonl", load=False)
    msgs = msg_pair(tool_output="first run out")
    rec = make_rec(msgs, ts=1)
    store.index(rec)
    m = Matcher(store, order=("exact", "fuzzy", "positional"))

    drifted = msg_pair(tool_output="second run out (different)")
    hit = m.match(
        keys=[exact_key(drifted, None, "m", 0, 4096)],
        fuzzy=fuzzy_key(drifted, None, "m", 0, 4096),
        session="s",
        turn=0,
    )
    assert hit.strategy == "fuzzy"

    other = msg_pair(task="完全不同的任务")
    hit = m.match(
        keys=[exact_key(other, None, "m", 0, 4096)],
        fuzzy=fuzzy_key(other, None, "m", 0, 4096),
        session="s",
        turn=0,
    )
    assert hit.strategy == "positional"
    assert hit.detail["turn"] == 0


def test_positional_verify_rejects_mismatch(tmp_path):
    store = RecordStore(tmp_path / "s.jsonl", load=False)
    rec = make_rec(msg_pair(), ts=1)
    store.index(rec)
    m = Matcher(store, order=("positional",), positional_verify="fuzzy")
    other = msg_pair(task="别的任务")
    assert m.match(
        keys=[exact_key(other, None, "m", 0, 4096)],
        fuzzy=fuzzy_key(other, None, "m", 0, 4096),
        session="s",
        turn=0,
    ) is None
    assert m.rejected_positional == 1
    # verify=none 时恢复纯轨迹回放
    m2 = Matcher(store, order=("positional",), positional_verify="none")
    hit = m2.match(
        keys=[exact_key(other, None, "m", 0, 4096)],
        fuzzy=fuzzy_key(other, None, "m", 0, 4096),
        session="s",
        turn=0,
    )
    assert hit is not None and hit.strategy == "positional"


def test_session_header_utf8_is_recovered():
    """HTTP 头的 latin-1 解码不该把中文会话名弄成乱码。"""
    r = SessionResolver()
    mangled = "会话-A".encode("utf-8").decode("latin-1")
    assert r.resolve({"x-llm-mock-session": mangled}, []) == ("会话-A", "header")
    assert r.resolve({"x-llm-mock-session": "plain-ascii"}, []) == ("plain-ascii", "header")
