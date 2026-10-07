"""miss 诊断测试：离线 Diagnoser、/admin/why 接口、why 命令。"""

import json

import pytest

from conftest import base_payload, get_json, post_json, user_msg
from llm_mock.canon import exact_key, fuzzy_key
from llm_mock.diagnose import DiagOptions, Diagnoser, format_report
from llm_mock.session import SessionResolver
from llm_mock.store import RecordStore
from llm_mock.why_cli import main as why_main


def tool_msgs(tool_output, tool_id="call_a", task="task"):
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"仓库 astropy/astropy @ d16bfe05a744\n\n{task}"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": tool_id, "type": "function",
                         "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
        {"role": "tool", "tool_call_id": tool_id, "content": tool_output},
    ]


def build_store(tmp_path, messages, model="m", temperature=0, max_tokens=4096):
    store = RecordStore(
        tmp_path / "s.jsonl", load=False, session_of=SessionResolver().from_messages
    )
    rec = store.make_record(
        key=exact_key(messages, None, model, temperature, max_tokens),
        fuzzy=fuzzy_key(messages, None, model, temperature, max_tokens),
        request={"messages": messages, "tools": None,
                 "temperature": temperature, "max_tokens": max_tokens},
        response={"content": "ok", "tool_calls": []},
        usage={}, latency_ms=1, model=model,
    )
    store.add(rec)
    return store, rec


def diag(store, **kw):
    opts = DiagOptions(top=kw.pop("top", 3), max_scan=kw.pop("max_scan", 0), **kw)
    return Diagnoser(store, opts)


def test_fuzzy_drift_is_reported_as_hit_expected(tmp_path):
    store, rec = build_store(tmp_path, tool_msgs("Thu Oct 5 12:00:01 UTC 2026\nreal 0m1.234s"))
    drifted = tool_msgs("Fri Oct 6 03:44:59 UTC 2026\nreal 0m9.876s", tool_id="call_zzz")
    report = diag(store).explain(
        {"model": "m", "messages": drifted, "temperature": 0, "max_tokens": 4096},
        session="astropy/astropy@d16bfe05a744",
        session_source="regex",
        turn=0,
    )
    assert report["verdict"] == "fuzzy-hit"
    assert report["strategies"]["exact"]["hit"] is False
    assert report["strategies"]["fuzzy"]["hit"] is True
    best = report["nearest"][0]
    kinds = {d["kind"] for d in best["tolerated"]}
    assert "tool_output" in kinds and "tool_call_id" in kinds
    assert best["fatal"] == []


def test_content_drift_is_fatal_and_suggests_positional(tmp_path):
    store, rec = build_store(tmp_path, tool_msgs("out-1", task="任务A"))
    changed = tool_msgs("out-1", task="任务B")
    report = diag(store).explain(
        {"model": "m", "messages": changed, "temperature": 0, "max_tokens": 4096},
        session="astropy/astropy@d16bfe05a744",
        session_source="regex",
        turn=0,
    )
    assert report["verdict"] == "content-drift"
    best = report["nearest"][0]
    assert any(d["kind"] == "content" for d in best["fatal"])
    assert report["strategies"]["positional"]["state"] == "rejected-by-verify"
    assert any("位置回放" in a for a in report["advice"])


def test_volatile_only_diff_suggests_flag(tmp_path):
    a = [{"role": "user", "content": "仓库 astropy/astropy @ d16bfe05a744 耗时 0.31s"}]
    b = [{"role": "user", "content": "仓库 astropy/astropy @ d16bfe05a744 耗时 9.99s"}]
    path = tmp_path / "v.jsonl"
    store = RecordStore(path, load=False, session_of=SessionResolver().from_messages)
    rec = store.make_record(
        key=exact_key(a, None, "m", 0, 4096), fuzzy=fuzzy_key(a, None, "m", 0, 4096),
        request={"messages": a, "tools": None, "temperature": 0, "max_tokens": 4096},
        response={"content": "x", "tool_calls": []}, usage={}, latency_ms=1, model="m",
    )
    store.add(rec)
    payload = {"model": "m", "messages": b, "temperature": 0, "max_tokens": 4096}

    # ① 都没开噪声归一化 -> 差异"只差噪声"，属致命项，但建议里给出开关
    report = diag(store, volatile=False).explain(
        payload, session="astropy/astropy@d16bfe05a744", session_source="regex", turn=0,
    )
    assert report["verdict"] == "content-drift"
    assert any(d["kind"] == "volatile_only" for d in report["nearest"][0]["fatal"])
    assert any("--fuzzy-normalize-volatile" in a for a in report["advice"])

    # ② 只在诊断侧开、存储侧没开 -> 报"只差可容忍项但没命中" + 明确指出两边参数不一致
    report_mixed = diag(store, volatile=True).explain(
        payload, session="astropy/astropy@d16bfe05a744", session_source="regex", turn=0,
    )
    assert report_mixed["verdict"] == "tolerable-only"
    assert any("参数不一致" in a for a in report_mixed["advice"])
    assert any("--fuzzy-* 参数" in a for a in report_mixed["advice"])

    # ③ 存储按 volatile 重建（等价于服务带 --fuzzy-normalize-volatile 重启）-> 模糊命中
    store_vol = RecordStore(
        path, mask_tool_output=True, volatile=True,
        session_of=SessionResolver().from_messages,
    )
    report2 = diag(store_vol, volatile=True).explain(
        payload, session="astropy/astropy@d16bfe05a744", session_source="regex", turn=0,
    )
    assert report2["verdict"] == "fuzzy-hit"


def test_missing_session_and_out_of_range(tmp_path):
    store, _ = build_store(tmp_path, tool_msgs("out"))
    report = diag(store).explain(
        {"model": "m", "messages": [{"role": "user", "content": "没有身份信息的请求"}],
         "temperature": 0, "max_tokens": 4096},
        session=None, session_source="none", turn=0,
    )
    assert report["strategies"]["positional"]["state"] == "no-session"
    assert any("X-LLM-Mock-Session" in a for a in report["advice"])

    report_unknown = diag(store).explain(
        {"model": "m", "messages": tool_msgs("out"), "temperature": 0, "max_tokens": 4096},
        session="从来没录过的会话", session_source="header", turn=0,
    )
    assert report_unknown["strategies"]["positional"]["state"] == "unknown-session"
    assert any("不在录制库里" in a for a in report_unknown["advice"])

    report2 = diag(store).explain(
        {"model": "m", "messages": tool_msgs("out"), "temperature": 0, "max_tokens": 4096},
        session="astropy/astropy@d16bfe05a744", session_source="regex", turn=5,
    )
    assert report2["strategies"]["positional"]["state"] == "out-of-range"
    assert any("超出" in a for a in report2["advice"])


def test_structure_diff_and_empty_store(tmp_path):
    store, _ = build_store(tmp_path, tool_msgs("out"))
    longer = tool_msgs("out") + [{"role": "user", "content": "extra"}]
    report = diag(store).explain(
        {"model": "m", "messages": longer, "temperature": 0, "max_tokens": 4096},
        session="astropy/astropy@d16bfe05a744", session_source="regex", turn=0,
    )
    assert any(d["kind"] == "structure" for d in report["nearest"][0]["fatal"])

    empty = RecordStore(tmp_path / "empty.jsonl", load=False)
    report2 = diag(empty).explain(
        {"model": "m", "messages": tool_msgs("out"), "temperature": 0, "max_tokens": 4096},
        session="s", session_source="regex", turn=0,
    )
    assert report2["verdict"] == "no-candidate"
    assert "没有结构相近" in format_report(report2)


def test_text_report_is_readable(tmp_path):
    store, _ = build_store(tmp_path, tool_msgs("out-1", task="任务A"))
    report = diag(store).explain(
        {"model": "m", "messages": tool_msgs("out-2", task="任务B"),
         "temperature": 0, "max_tokens": 4096},
        session="astropy/astropy@d16bfe05a744", session_source="regex", turn=0,
    )
    text = format_report(report)
    assert "miss 诊断" in text and "结论" in text and "最接近的" in text


# --------------------------------------------------------------------------
# 服务内
# --------------------------------------------------------------------------
def test_admin_why_after_replay_miss(make_proxy):
    base, svc, state = make_proxy(mode="record")
    post_json(base, "/v1/chat/completions", base_payload("任务A"))

    base2, svc2, state2 = make_proxy(mode="replay", name="store.jsonl")
    st, _, body = post_json(base2, "/v1/chat/completions", base_payload("任务B"))
    assert st == 409
    err = json.loads(body)["error"]
    assert "miss_id" in err and "hint" in err
    assert state.calls == 1, "诊断路径不得外呼（只有录制阶段那一次）"

    st, listing = get_json(base2, "/admin/misses")
    assert listing["count"] == 1
    assert listing["misses"][0]["id"] == err["miss_id"]

    st, why = get_json(base2, f"/admin/why?id={err['miss_id']}")
    assert st == 200
    report = why["report"]
    assert report["verdict"] == "content-drift"
    assert report["request"]["session"] == "astropy/astropy@d16bfe05a744"
    assert "miss 诊断" in why["text"]

    # 文本格式
    import urllib.request

    with urllib.request.urlopen(base2 + "/admin/why?format=text", timeout=30) as resp:
        assert "=== llm-mock miss 诊断 ===" in resp.read().decode()


def test_admin_why_post_arbitrary_request(make_proxy):
    import urllib.request

    base, svc, state = make_proxy(mode="record")
    post_json(base, "/v1/chat/completions", base_payload("任务A"))
    req = urllib.request.Request(
        base + "/admin/why",
        data=json.dumps(base_payload("任务A")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read())
    assert out["report"]["verdict"] == "exact-hit"
    assert state.calls == 1  # 诊断本身不外呼


def test_hybrid_miss_defers_deep_report(make_proxy):
    base, svc, state = make_proxy(mode="hybrid", name="empty.jsonl")
    st, headers, _ = post_json(base, "/v1/chat/completions", base_payload("新任务"))
    assert st == 200 and "X-LLM-Mock-Hit" not in headers
    assert state.calls == 1
    # hybrid 的 miss 只留档不算深度报告（避免首次录制时每轮全库扫描）
    item = svc.miss_log.items[-1]
    assert item["report"] is None and item["payload"] is not None
    st, why = get_json(base, "/admin/why")
    # 深度报告按需计算，并排除"本次 miss 之后才录进去的那条"，否则会自证命中
    assert why["report"]["verdict"] in ("no-candidate", "content-drift")
    assert why["report"]["strategies"]["exact"]["hit"] is False
    assert any("已忽略" in a for a in why["report"]["advice"])
    assert svc.miss_log.items[-1]["report"] is not None  # 按需算完后缓存


def test_no_miss_returns_404(make_proxy):
    base, svc, state = make_proxy(mode="replay", name="empty.jsonl")
    import urllib.error

    try:
        get_json(base, "/admin/why")
        raise AssertionError("应当 404")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------
def test_why_cli_offline(tmp_path, capsys):
    store = RecordStore(
        tmp_path / "cli.jsonl", load=False, session_of=SessionResolver().from_messages
    )
    msgs = tool_msgs("out-1", task="任务A")
    rec = store.make_record(
        key=exact_key(msgs, None, "m", 0, 4096), fuzzy=fuzzy_key(msgs, None, "m", 0, 4096),
        request={"messages": msgs, "tools": None, "temperature": 0, "max_tokens": 4096},
        response={"content": "ok", "tool_calls": []}, usage={}, latency_ms=1, model="m",
    )
    store.add(rec)
    req = tmp_path / "req.json"
    req.write_text(json.dumps(
        {"model": "m", "messages": tool_msgs("out-2", task="任务B"),
         "temperature": 0, "max_tokens": 4096}, ensure_ascii=False))

    code = why_main(["--store", str(store.path), "--request", str(req), "--turn", "0"])
    out = capsys.readouterr().out
    assert code == 2, "未命中应返回退出码 2"
    assert "miss 诊断" in out and "content-drift" in out

    req2 = tmp_path / "req2.json"
    req2.write_text(json.dumps(
        {"model": "m", "messages": msgs, "temperature": 0, "max_tokens": 4096},
        ensure_ascii=False))
    code2 = why_main(["--store", str(store.path), "--request", str(req2), "--json"])
    out2 = capsys.readouterr().out
    assert code2 == 0 and json.loads(out2)["verdict"] == "exact-hit"
