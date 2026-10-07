"""端到端测试：真的起 HTTP 代理 + 假上游，覆盖四种模式与三种匹配策略。"""

import json

from conftest import base_payload, get_json, parse_sse, post_json, user_msg


def test_record_then_exact_replay(make_proxy):
    base, svc, state = make_proxy(mode="record")
    payload = base_payload()
    st, _, body = post_json(base, "/v1/chat/completions", payload)
    assert st == 200
    first = json.loads(body)
    assert first["choices"][0]["message"]["content"].startswith("fake-reply#1")
    assert state.calls == 1
    assert len(svc.store.by_key) == 1

    # 同一个代理切到 replay：内容一字不改 -> 精准命中
    base2, svc2, state2 = make_proxy(mode="replay", name="store.jsonl")
    st, headers, body = post_json(base2, "/v1/chat/completions", payload)
    assert st == 200, body
    assert headers["X-LLM-Mock-Hit"] == "exact"
    assert json.loads(body)["choices"][0]["message"]["content"] == first["choices"][0]["message"]["content"]
    assert state.calls == 1, "replay 不得访问真实 API"


def test_fuzzy_replay_ignores_tool_output(make_proxy):
    base, svc, state = make_proxy(mode="record")
    messages = [
        {"role": "system", "content": "sys"},
        user_msg(),
        {"role": "user", "content": "USE_TOOL 请执行"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_a", "type": "function",
                         "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
        {"role": "tool", "tool_call_id": "call_a",
         "content": "Thu Oct 5 12:00:01 UTC 2026\nreal 0m1.234s\nfile1"},
    ]
    payload = {"model": "fake-model", "messages": messages, "temperature": 0, "max_tokens": 4096,
               "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}]}
    st, _, body = post_json(base, "/v1/chat/completions", payload)
    assert st == 200
    recorded = json.loads(body)["choices"][0]["message"]

    drifted = [dict(m) for m in messages]
    drifted[3] = {"role": "assistant", "content": None,
                  "tool_calls": [{"id": "call_zzz", "type": "function",
                                  "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]}
    drifted[4] = {"role": "tool", "tool_call_id": "call_zzz",
                  "content": "Thu Oct 5 12:09:59 UTC 2026\nreal 0m7.891s\nfile2\nfile3"}
    payload2 = dict(payload, messages=drifted)

    base2, svc2, state2 = make_proxy(mode="replay", name="store.jsonl")
    st, headers, body = post_json(base2, "/v1/chat/completions", payload2)
    assert st == 200, body
    assert headers["X-LLM-Mock-Hit"] == "fuzzy"
    assert json.loads(body)["choices"][0]["message"]["content"] == recorded["content"]
    assert state.calls == 1, "模糊命中同样不得访问真实 API"


def test_replay_miss_returns_409_and_zero_upstream(make_proxy):
    base, svc, state = make_proxy(mode="record")
    post_json(base, "/v1/chat/completions", base_payload())
    assert state.calls == 1

    base2, svc2, state2 = make_proxy(mode="replay", name="store.jsonl")
    st, _, body = post_json(base2, "/v1/chat/completions", base_payload("一个全新的任务"))
    assert st == 409
    err = json.loads(body)["error"]
    assert err["type"] == "replay_miss"
    assert err["turn"] == 0 and err["session"] == "astropy/astropy@d16bfe05a744"
    assert state.calls == 1, "replay 未命中绝不允许外呼"  # 只有录制阶段那一次
    stats = get_json(base2, "/health")[1]
    assert stats["misses"] == 1 and stats["upstream_calls"] == 0


def test_positional_replay_follows_turns(make_proxy):
    base, svc, state = make_proxy(mode="record")
    # 位置回放的正确用法：客户端**显式声明**自己在重放哪条轨迹
    traj = {"X-LLM-Mock-Session": "traj-A"}
    contents = []
    for i in range(3):
        st, _, body = post_json(base, "/v1/chat/completions", base_payload(f"任务{i}"), traj)
        assert st == 200
        contents.append(json.loads(body)["choices"][0]["message"]["content"])
    assert state.calls == 3

    base2, svc2, state2 = make_proxy(
        mode="replay", name="store.jsonl", replay_order=("positional",)
    )
    # 请求内容全都换掉（精准/模糊都不可能命中），位置回放照样按顺序给答案
    for i in range(3):
        st, headers, body = post_json(
            base2, "/v1/chat/completions", base_payload(f"全新内容{i}"), traj
        )
        assert st == 200, body
        assert headers["X-LLM-Mock-Hit"] == "positional"
        assert json.loads(body)["choices"][0]["message"]["content"] == contents[i]
    st, _, body = post_json(base2, "/v1/chat/completions", base_payload("第 4 轮"), traj)
    assert st == 409 and json.loads(body)["error"]["type"] == "replay_miss"
    assert state.calls == 3, "replay 全程 0 外呼（只有录制阶段的 3 次）"


def test_positional_auto_verify_rejects_other_task(make_proxy):
    """会话是"猜"出来的（regex）时，位置命中要过一次模糊键校验 ——
    同一 repo@commit 下的另一个任务不应被误回放。"""
    base, svc, state = make_proxy(mode="record")
    post_json(base, "/v1/chat/completions", base_payload("任务A"))
    base2, svc2, state2 = make_proxy(
        mode="replay", name="store.jsonl", replay_order=("positional",)
    )
    st, _, body = post_json(base2, "/v1/chat/completions", base_payload("任务B"))
    assert st == 409 and json.loads(body)["error"]["type"] == "replay_miss"
    assert svc2.matcher.rejected_positional == 1
    assert state.calls == 1

    # 显式指定 --positional-verify none，就恢复成"纯轨迹回放"
    base3, svc3, state3 = make_proxy(
        mode="replay", name="store.jsonl", replay_order=("positional",),
        positional_verify="none",
    )
    st, headers, _ = post_json(base3, "/v1/chat/completions", base_payload("任务B"))
    assert st == 200 and headers["X-LLM-Mock-Hit"] == "positional"


def test_positional_turn_header_and_reset(make_proxy):
    base, svc, state = make_proxy(mode="record")
    traj = {"X-LLM-Mock-Session": "traj-B"}
    for i in range(2):
        post_json(base, "/v1/chat/completions", base_payload(f"任务{i}"), traj)
    base2, svc2, state2 = make_proxy(
        mode="replay", name="store.jsonl", replay_order=("positional",)
    )
    hdr = {**traj, "X-LLM-Mock-Turn": "1"}
    st, headers, _ = post_json(base2, "/v1/chat/completions", base_payload("任意"), hdr)
    assert st == 200 and headers["X-LLM-Mock-Turn"] == "1"
    post_json(base2, "/admin/reset", {})
    st, headers, _ = post_json(base2, "/v1/chat/completions", base_payload("任意"), traj)
    assert st == 200 and headers["X-LLM-Mock-Turn"] == "0"


def test_hybrid_hit_and_forward(make_proxy):
    base, svc, state = make_proxy(mode="record")
    post_json(base, "/v1/chat/completions", base_payload())
    recorded = json.loads(
        post_json(base, "/v1/chat/completions", base_payload()) [2]
    )["choices"][0]["message"]["content"]

    base2, svc2, state2 = make_proxy(mode="hybrid", name="store.jsonl")
    st, headers, _ = post_json(base2, "/v1/chat/completions", base_payload())
    assert st == 200 and headers["X-LLM-Mock-Hit"] == "exact"
    assert state.calls == 2, "hybrid 命中不得外呼"

    # 新会话：匹配不上 -> 转发真实 API 并录制
    new_task = base_payload()
    new_task["messages"][1] = user_msg(repo="django/django", commit="abcdef123456")
    st, headers, body = post_json(base2, "/v1/chat/completions", new_task)
    assert st == 200
    assert "X-LLM-Mock-Hit" not in headers
    assert state.calls == 3
    assert len(svc2.store.by_key) == 2
    assert json.loads(body)["choices"][0]["message"]["content"].startswith("fake-reply#3")
    assert recorded  # 上一次录制的回复确实被回放了


def test_stream_record_and_replay(make_proxy):
    base, svc, state = make_proxy(mode="record")
    payload = dict(base_payload(), stream=True)
    st, _, raw = post_json(base, "/v1/chat/completions", payload)
    assert st == 200
    live = parse_sse(raw)
    assert live["content"].startswith("fake-reply#1")
    assert len(svc.store.by_key) == 1, "流式响应也必须被拼回完整消息并录制"

    base2, svc2, state2 = make_proxy(mode="replay", name="store.jsonl")
    st, headers, raw2 = post_json(base2, "/v1/chat/completions", payload)
    assert st == 200 and headers["X-LLM-Mock-Hit"] == "exact"
    replayed = parse_sse(raw2)
    assert replayed["content"] == live["content"]
    assert state.calls == 1


def test_stream_tool_call_roundtrip(make_proxy):
    base, svc, state = make_proxy(mode="record")
    messages = [
        user_msg(),
        {"role": "user", "content": "USE_TOOL 请执行"},
    ]
    payload = {
        "model": "fake-model",
        "messages": messages,
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }
    st, _, raw = post_json(base, "/v1/chat/completions", payload)
    assert st == 200
    live = parse_sse(raw)
    assert live["tool_calls"] and live["tool_calls"][0]["name"] == "bash"

    base2, svc2, state2 = make_proxy(mode="replay", name="store.jsonl")
    st, _, raw2 = post_json(base2, "/v1/chat/completions", payload)
    replayed = parse_sse(raw2)
    assert replayed["tool_calls"][0]["name"] == "bash"
    assert json.loads(replayed["tool_calls"][0]["arguments"]) == {"command": "echo hi"}
    assert state.calls == 1


def test_live_mode_never_records(make_proxy):
    base, svc, state = make_proxy(mode="live")
    st, _, _ = post_json(base, "/v1/chat/completions", base_payload())
    assert st == 200 and state.calls == 1
    assert len(svc.store.by_key) == 0
    assert not (svc.store.path.exists() and svc.store.path.stat().st_size)


def test_replay_mode_blocks_other_endpoints(make_proxy):
    base, svc, state = make_proxy(mode="replay", name="empty.jsonl")
    st, _, body = post_json(base, "/v1/embeddings", {"input": "x"})
    assert st == 501 and json.loads(body)["error"]["type"] == "replay_unsupported"
    assert state.calls == 0


def test_admin_endpoints(make_proxy):
    base, svc, state = make_proxy(mode="record")
    post_json(base, "/v1/chat/completions", base_payload())
    st, health = get_json(base, "/health")
    assert st == 200
    assert health["mode"] == "record"
    assert health["upstream_calls"] == 1
    assert health["records"] == 1
    st, sessions = get_json(base, "/admin/sessions")
    assert sessions["sessions"][0]["session"] == "astropy/astropy@d16bfe05a744"
    st, records = get_json(base, "/admin/records?limit=10")
    assert records["total"] == 1 and records["records"][0]["turn"] == 0
    post_json(base, "/admin/mode", {"mode": "live"})
    assert get_json(base, "/health")[1]["mode"] == "live"


def test_fuzzy_volatile_normalization_e2e(make_proxy):
    base, svc, state = make_proxy(mode="record")
    messages = [user_msg(extra="跑测试"), {"role": "user", "content": "耗时 0.31s 完成"}]
    payload = {"model": "fake-model", "messages": messages, "temperature": 0, "max_tokens": 4096}
    post_json(base, "/v1/chat/completions", payload)
    drifted = [user_msg(extra="跑测试"), {"role": "user", "content": "耗时 9.99s 完成"}]
    payload2 = dict(payload, messages=drifted)

    base_no, _, _ = make_proxy(mode="replay", name="store.jsonl")
    st, _, _ = post_json(base_no, "/v1/chat/completions", payload2)
    assert st == 409, "默认不洗噪声时应当未命中"

    base_vol, _, state_vol = make_proxy(
        mode="replay", name="store.jsonl", volatile=True
    )
    st, headers, _ = post_json(base_vol, "/v1/chat/completions", payload2)
    assert st == 200 and headers["X-LLM-Mock-Hit"] == "fuzzy"
    assert state.calls == 1, "模糊命中后仍只有录制阶段那一次外呼"


def test_concurrent_record_and_replay(make_proxy):
    """并发安全：400 个并发请求（模拟 500 并发评测场景的规模）不应丢记录或报错。"""
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    def brief(results, n=3):
        bad = [(st, h.get("X-LLM-Mock-Hit"), b[:160]) for st, h, b in results if st != 200]
        return f"状态分布={dict(Counter(st for st, _, _ in results))} 非200样本={bad[:n]}"

    base, svc, state = make_proxy(mode="record")
    payloads = [base_payload(f"并发任务-{i}") for i in range(40)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(lambda p: post_json(base, "/v1/chat/completions", p), payloads * 10)
        )
    assert all(st == 200 for st, _, _ in results), f"并发下有请求失败: {brief(results)}"
    assert len(svc.store.by_key) == 40, "同一请求并发重放应去重为 40 条"
    assert svc.store.path.read_text().count("\n") == 400, "落盘计数应与请求数一致"
    assert state.calls == 400, f"上游被调用 {state.calls} 次，应为 400"

    # 并发回放：全部必须命中，且一次外呼都没有
    base2, svc2, state2 = make_proxy(mode="replay", name="store.jsonl")
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(lambda p: post_json(base2, "/v1/chat/completions", p), payloads * 5)
        )
    hits = Counter(h.get("X-LLM-Mock-Hit") for _, h, _ in results)
    assert all(st == 200 and h.get("X-LLM-Mock-Hit") == "exact" for st, h, _ in results), (
        f"并发回放有非精准命中: 命中分布={dict(hits)} {brief(results)}"
    )
    assert get_json(base2, "/health")[1]["upstream_calls"] == 0
