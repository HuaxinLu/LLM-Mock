"""脱敏测试：模式覆盖、确定性、端到端不落盘、与回放键的一致性。"""

import json

from conftest import base_payload, get_json, post_json, user_msg
from llm_mock.canon import exact_key, fuzzy_key
from llm_mock.scrub import Scrubber, build_scrubber
from llm_mock.session import SessionResolver
from llm_mock.store import RecordStore

# 这些"假密钥"在**运行时拼接**，文件里不出现任何完整的密钥形态字面量 ——
# 否则 GitHub 推送保护 / secret scanner 会把仓库当成泄露了真密钥而拦下。
def _fake(*parts: str) -> str:
    return "".join(parts)


FAKE_OPENAI = _fake("sk-", "abcdefghijkl", "mnopqrstuvwx", "yz0123456789")
FAKE_GH = _fake("ghp_", "ABCDEFGHIJKL", "MNOPQRSTUVWX", "YZ0123456789")
FAKE_AWS = _fake("AKIA", "IOSFODNN7EXAMPLE")
FAKE_BEARER = _fake("Bearer ", "abcdefghijklmnopqrstuvwxyz")
FAKE_JWT = _fake(
    "eyJhbGciOiJIUzI1NiJ9", ".", "eyJzdWIiOiIxMjM0NTY3ODkwIn0", ".",
    "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
)
# 连 PEM 的 -----BEGIN/END 标记也拆开拼，避免被 gitleaks / GitHub 推送保护的
# "private key" 规则在仓库里扫出一个假阳性
FAKE_PRIVATE_KEY = _fake(
    "-----BEGIN ", "RSA ", "PRIVATE ", "KEY-----\n",
    "MIIEowIBAAKCAQEA",
    "\n-----END ", "RSA ", "PRIVATE ", "KEY-----",
)
FAKE_PASSWORD = _fake("Super", "Secret", "123")
FAKE_CONN = _fake("postgres://admin:", FAKE_PASSWORD, "@db.example:5432/app")


# --------------------------------------------------------------------------
# 单元
# --------------------------------------------------------------------------
def test_known_shapes_are_redacted():
    s = Scrubber()
    text = (
        f"key={FAKE_OPENAI} token={FAKE_GH} aws={FAKE_AWS} "
        f"auth={FAKE_BEARER} conn={FAKE_CONN}"
    )
    out = s.scrub_text(text)
    for raw in (FAKE_OPENAI, FAKE_GH, FAKE_AWS, FAKE_BEARER, FAKE_PASSWORD):
        assert raw not in out
    assert out.count("[REDACTED:") >= 5
    assert s.counts["openai_key"] == 1 and s.counts["github_token"] == 1
    assert "aws_access_key" in s.counts and "bearer_token" in s.counts
    assert "connection_string" in s.counts


def test_jwt_private_key_and_env_assignment():
    s = Scrubber()
    token_value = _fake("abcdefghijkl", "mnop1234")
    out = s.scrub_text(f"jwt={FAKE_JWT}\n{FAKE_PRIVATE_KEY}\nMY_APP_TOKEN={token_value}")
    assert "eyJhbGciOiJIUzI1NiJ9" not in out
    assert "MIIEowIBAAKCAQEA" not in out
    assert token_value not in out
    assert "MY_APP_TOKEN=[REDACTED:" in out, "= 左边的名字应当保留，便于人工核对"
    assert {"jwt", "private_key_block", "env_assignment"} <= set(s.counts)


def test_deterministic_and_distinct():
    a = Scrubber()
    b = Scrubber()
    assert a.scrub_text(FAKE_OPENAI) == b.scrub_text(FAKE_OPENAI), "确定性：同值同占位"
    other = _fake("sk-", "z" * 32)
    assert a.scrub_text(FAKE_OPENAI) != a.scrub_text(other), "不同密钥不应撞成一个占位"
    red = Scrubber(mode="redact")
    assert red.scrub_text(FAKE_OPENAI) == red.scrub_text(other) == "[REDACTED:openai_key]"


def test_by_name_redaction_and_protocol_fields_pass_through():
    s = Scrubber()
    payload = {
        "model": "m",
        "max_tokens": "4096",          # 名字含 token，但不能被打码
        "temperature": 0,
        "messages": [{"role": "user", "content": "hello"}],
        "api_key": _fake("totally-", "unpatterned-", "value"),
        "tool_call_id": "call_abc",
        "nested": {"authorization": _fake("bearer ", "whatever-value-here")},
    }
    out = s.scrub_obj(payload)
    assert out["max_tokens"] == "4096"
    assert out["tool_call_id"] == "call_abc"
    assert out["api_key"] == "[REDACTED:by_name:" + out["api_key"].split(":")[-1]
    assert "unpatterned" not in json.dumps(out)
    assert "whatever-value-here" not in json.dumps(out)
    assert payload["api_key"] == _fake("totally-", "unpatterned-", "value"), "不得修改入参"
    assert s.counts["by_name"] == 2


def test_dry_run_counts_without_replacing():
    s = Scrubber(dry_run=True)
    assert s.scrub_text(FAKE_OPENAI) == FAKE_OPENAI
    assert s.counts["openai_key"] == 1


def test_disabled_and_custom_pattern():
    off = Scrubber(enabled=False)
    assert off.scrub_text(FAKE_OPENAI) == FAKE_OPENAI
    custom_token = _fake("MYCO-", "ABCDEFGHIJ")
    custom = Scrubber(extra_patterns=[r"MYCO-[A-Z0-9]{10}"])
    out = custom.scrub_text(f"id {custom_token} done")
    assert custom_token not in out and custom.stats()["extra_patterns"] == 1
    assert custom.counts["custom"] == 1


def test_build_scrubber_key_resolution(tmp_path):
    keyfile = tmp_path / "scrub.key"
    keyfile.write_text("project-specific-key\n")
    s = build_scrubber(key_file=str(keyfile))
    assert s.key == b"project-specific-key"
    assert build_scrubber().key  # 没给 key 时用默认常量，不崩


# --------------------------------------------------------------------------
# 索引兼容：老录制（落盘时未脱敏）在开脱敏后仍能精准命中
# --------------------------------------------------------------------------
def test_existing_record_still_exact_matches_with_scrub_on(tmp_path):
    msgs = [{"role": "user", "content": f"仓库 astropy/astropy @ d16bfe05a744 密钥 {FAKE_OPENAI}"}]
    path = tmp_path / "existing.jsonl"
    path.write_text(json.dumps({
        "key": exact_key(msgs, None, "m", 0, 4096),      # 落盘时未脱敏：按原文算的键
        "ts": 1, "model": "m",
        "request": {"messages": msgs, "tools": None, "temperature": 0, "max_tokens": 4096},
        "response": {"content": "ok", "tool_calls": []}, "usage": {}, "latency_ms": 1,
    }, ensure_ascii=False) + "\n")

    store = RecordStore(
        path, session_of=SessionResolver().from_messages, scrubber=Scrubber()
    )
    assert store.stats()["compat_equivalent_keys"] == 1
    # 客户端发来的还是原文 -> 脱敏后应命中那条既有记录
    scrubbed = Scrubber().scrub_obj({"messages": msgs, "tools": None,
                                     "temperature": 0, "max_tokens": 4096})
    key = exact_key(scrubbed["messages"], None, "m", 0, 4096)
    assert store.lookup_exact([key]) is not None, "开脱敏后既有记录的精准命中不该丢"
    assert store.lookup_fuzzy(fuzzy_key(scrubbed["messages"], None, "m", 0, 4096)) is not None


# --------------------------------------------------------------------------
# 端到端
# --------------------------------------------------------------------------
def secret_payload(task="带密钥的任务"):
    return {
        "model": "fake-model",
        "messages": [
            {"role": "system", "content": "you are a fixer"},
            {"role": "user", "content": f"仓库 astropy/astropy @ d16bfe05a744\n{task}\nOPENAI_API_KEY={FAKE_OPENAI}"},
            {"role": "user", "content": f"我的数据库是 {FAKE_CONN}"},
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }


def test_record_never_writes_secret_to_disk(make_proxy):
    base, svc, state = make_proxy(mode="record")
    st, _, body = post_json(base, "/v1/chat/completions", secret_payload())
    assert st == 200

    raw_file = svc.store.path.read_text(encoding="utf-8")
    assert FAKE_OPENAI not in raw_file, "密钥写进了 JSONL！"
    assert FAKE_PASSWORD not in raw_file, "连接串密码写进了 JSONL！"
    assert "[REDACTED:" in raw_file

    # 上游必须收到**原文**（脱敏只针对落盘/匹配，不影响业务）
    upstream_body = json.dumps(state.requests[-1], ensure_ascii=False)
    assert FAKE_OPENAI in upstream_body, "转发给上游的请求被误改"

    # 落盘记录里的 request 与 response 都脱敏了
    rec = next(iter(svc.store.by_key.values()))
    assert FAKE_OPENAI not in json.dumps(rec, ensure_ascii=False)
    assert "[REDACTED:" in json.dumps(rec["response"], ensure_ascii=False)

    health = get_json(base, "/health")[1]
    assert health["scrub"]["enabled"] is True
    assert health["scrub"]["redactions_total"] > 0
    # OPENAI_API_KEY=sk-… 会先被 env_assignment 命中（保留名字前缀），两种标签都算对
    assert {"env_assignment", "openai_key"} & set(health["scrub"]["redactions"])


def test_scrub_keeps_replay_matching(make_proxy):
    base, svc, state = make_proxy(mode="record")
    payload = secret_payload()
    st, _, body = post_json(base, "/v1/chat/completions", payload)
    first = json.loads(body)["choices"][0]["message"]["content"]
    assert FAKE_PASSWORD in first, "record 阶段客户端拿到的仍是上游原始响应（脱敏只作用于落盘）"
    assert state.calls == 1

    base2, svc2, state2 = make_proxy(mode="replay", name="store.jsonl")
    st, headers, body = post_json(base2, "/v1/chat/completions", payload)
    assert st == 200 and headers["X-LLM-Mock-Hit"] == "exact", "脱敏必须是确定性的，否则回放会失配"
    replayed = json.loads(body)["choices"][0]["message"]["content"]
    # 回放只能给出落盘的那份（已脱敏）：模型原样吐回的密钥在回放里变成占位符
    assert FAKE_PASSWORD not in replayed and "[REDACTED:" in replayed
    assert replayed.startswith("fake-reply#1")
    assert state.calls == 1, "replay 阶段 0 外呼"


def test_no_scrub_keeps_raw(make_proxy):
    base, svc, state = make_proxy(mode="record", scrub=False)
    post_json(base, "/v1/chat/completions", secret_payload())
    raw_file = svc.store.path.read_text(encoding="utf-8")
    assert FAKE_OPENAI in raw_file, "--no-scrub 时应保持原始行为"


def test_miss_log_and_why_do_not_leak(make_proxy):
    base, svc, state = make_proxy(mode="record")
    post_json(base, "/v1/chat/completions", secret_payload())

    base2, svc2, state2 = make_proxy(mode="replay", name="store.jsonl")
    st, _, body = post_json(base2, "/v1/chat/completions", secret_payload("完全不同的问题"))
    assert st == 409
    miss_id = json.loads(body)["error"]["miss_id"]

    blob = json.dumps(get_json(base2, f"/admin/why?id={miss_id}")[1], ensure_ascii=False)
    assert FAKE_OPENAI not in blob and FAKE_PASSWORD not in blob
    blob2 = json.dumps(get_json(base2, "/admin/misses")[1], ensure_ascii=False)
    assert FAKE_OPENAI not in blob2

    # POST /admin/why 传进来的请求体也要先脱敏再诊断
    import urllib.request

    req = urllib.request.Request(
        base2 + "/admin/why",
        data=json.dumps(secret_payload()).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = resp.read().decode()
    assert FAKE_OPENAI not in out and FAKE_PASSWORD not in out
