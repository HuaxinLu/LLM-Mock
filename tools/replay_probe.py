#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据回放探针：拿一份既有 JSONL 录制跑三种回放，全程 0 token。

做法：
  1. 用 --store 指定的录制文件当存储加载进代理（replay 模式）
  2. 上游地址指向一个**死端口**（127.0.0.1:9）—— 一旦发生真实外呼立刻暴露
  3. 用真实 HTTP 请求打代理，验证 exact / fuzzy / positional 三种命中
  4. 另外做一次"工具输出扰动"实验：把每条录制的工具输出换成随机文本，
     看模糊键能否把第 N 轮重新对上

用法: python tools/replay_probe.py --store records.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_mock.canon import exact_key, fuzzy_key, normalize_messages  # noqa: E402
from llm_mock.proxy import Config, MockService, make_server  # noqa: E402

DEFAULT_STORE = os.environ.get("LLM_MOCK_STORE", "./llm_mock_records.jsonl")
DEAD_UPSTREAM = "http://127.0.0.1:9"


def post(base: str, payload: dict, headers: dict | None = None):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), json.loads(exc.read())


def perturb_tool_outputs(messages, rnd: random.Random):
    """把 role=tool/function 的内容换成不可控的运行时噪声。"""
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") in ("tool", "function"):
            m = dict(m)
            m["content"] = (
                f"Thu Oct 8 {rnd.randint(0,23):02d}:{rnd.randint(0,59):02d}:{rnd.randint(0,59):02d} UTC 2026\n"
                f"real 0m{rnd.randint(1,9)}.{rnd.randint(100,999)}s\n"
                f"/tmp/pytest-of-root/pytest-{rnd.randint(0,9)}/out.txt\n"
                f"line-{rnd.randint(1000,9999)}"
            )
        out.append(m)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--probe-turns", type=int, default=400, help="扰动实验抽样条数")
    ap.add_argument("--port", type=int, default=8891)
    args = ap.parse_args(argv)

    print(f"== 加载录制存储: {args.store}")
    t0 = time.time()
    cfg = Config(
        mode="replay",
        store_path=args.store,
        upstream_url=DEAD_UPSTREAM,   # 死端口：任何外呼都会失败
        replay_latency="none",
        host="127.0.0.1",
        port=args.port,
        replay_order=("exact", "fuzzy", "positional"),
    )
    svc = MockService(cfg)
    load_sec = time.time() - t0
    rss = 0
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    print(
        f"   载入 {len(svc.store.by_key)} 条 / {len(svc.store.by_session)} 会话 "
        f"耗时 {load_sec:.1f}s, RSS {rss:.2f} GiB"
    )

    httpd = make_server(cfg, svc)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    # ---------- 扰动实验（纯函数，先跑，快） ----------
    rnd = random.Random(20261006)
    checked = recovered = 0
    with open(args.store, encoding="utf-8") as fh:
        for line in fh:
            if checked >= args.probe_turns:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            req = rec.get("request") or {}
            messages = req.get("messages") or []
            if not any(isinstance(m, dict) and m.get("role") == "tool" for m in messages):
                continue
            checked += 1
            drifted = perturb_tool_outputs(messages, rnd)
            if fuzzy_key(
                drifted, req.get("tools"), req.get("model"),
                req.get("temperature"), req.get("max_tokens"),
            ) == fuzzy_key(
                messages, req.get("tools"), req.get("model"),
                req.get("temperature"), req.get("max_tokens"),
            ):
                recovered += 1
    print(
        f"\n== 工具输出扰动实验: {recovered}/{checked} 条在扰动后模糊键不变 "
        f"({100.0*recovered/max(1,checked):.1f}%)"
    )

    # ---------- 三种回放（真实 HTTP） ----------
    session = max(svc.store.by_session.items(), key=lambda kv: len(kv[1]))[0]
    seq = svc.store.by_session[session]
    print(f"\n== 用会话 {session}（{len(seq)} 轮）做真实 HTTP 回放")

    rows = []

    def show(label, status, headers, extra=""):
        hit = headers.get("X-LLM-Mock-Hit") or "-"
        rows.append((label, status, hit, extra))
        print(f"   {label:<28} HTTP {status}  hit={hit}  {extra}")

    # 1) 精准：原样重放第 0 轮请求（记录层的 model 不在 request 子对象里）
    def payload_of(rec, messages=None):
        p = dict(rec["request"])
        p["messages"] = messages if messages is not None else rec["request"]["messages"]
        p["model"] = rec.get("model")
        return p

    rec0 = seq[0]
    st, h, _ = post(base, payload_of(rec0), {"X-LLM-Mock-Session": session, "X-LLM-Mock-Turn": "0"})
    show("exact（原样重放 turn0）", st, h)

    # 2) 模糊：第 1 轮请求 + 工具输出全扰动
    if len(seq) > 1:
        rec1 = seq[1]
        drifted = perturb_tool_outputs(rec1["request"]["messages"], rnd)
        st, h, _ = post(
            base,
            payload_of(rec1, drifted),
            {"X-LLM-Mock-Session": session, "X-LLM-Mock-Turn": "1"},
        )
        show("fuzzy（工具输出全扰动）", st, h)

    # 3) 位置：内容完全换掉，只声明会话与轮次
    payload3 = payload_of(rec0, [{"role": "user", "content": "一个与录制完全无关的请求"}])
    st, h, _ = post(base, payload3, {"X-LLM-Mock-Session": session, "X-LLM-Mock-Turn": "2"})
    show("positional（内容无关）", st, h)

    # 5) 未命中：未知会话 -> 拿服务内的诊断报告
    st, h, body = post(base, payload3, {"X-LLM-Mock-Session": "no-such-session"})
    miss_id = body["error"].get("miss_id")
    show("replay miss（未知会话）", st, h, json.loads(json.dumps(body))["error"]["type"])
    if miss_id:
        with urllib.request.urlopen(
            base + f"/admin/why?id={miss_id}&format=text", timeout=120
        ) as resp:
            print("\n== GET /admin/why?id=%s&format=text（服务内诊断）" % miss_id)
            print(resp.read().decode())

    # 6) 离线诊断：把"工具输出全扰动 + 正文改一句"的真实请求丢给 Diagnoser
    from llm_mock.diagnose import DiagOptions, Diagnoser, format_report

    if len(seq) > 5:
        rec5 = seq[5]
        drifted5 = perturb_tool_outputs(rec5["request"]["messages"], rnd)
        for i, m in enumerate(drifted5):
            if isinstance(m, dict) and m.get("role") == "user":
                m = dict(m)
                m["content"] = str(m.get("content")) + "\n（这次还额外改了一句正文）"
                drifted5[i] = m
                break
        diag = Diagnoser(
            svc.store,
            DiagOptions(
                mask_tool_output=svc.cfg.mask_tool_output,
                volatile=svc.cfg.volatile,
                replay_order=tuple(svc.cfg.replay_order),
                positional_verify=svc.cfg.positional_verify,
                top=2,
                max_scan=0,
            ),
        )
        t0 = time.time()
        rep = diag.explain(
            payload_of(rec5, drifted5),
            session=session,
            session_source="regex",
            turn=5,
            mode="replay",
        )
        print("== 离线诊断（全库 %.0f 条扫描，耗时 %.1fs）" % (len(svc.store.by_key), time.time() - t0))
        print(format_report(rep))

    with urllib.request.urlopen(base + "/health", timeout=30) as resp:
        health = json.loads(resp.read())
    print("\n== /health")
    print(json.dumps({k: health[k] for k in
                      ("mode", "chat_requests", "hits", "misses", "replayed",
                       "forwarded", "upstream_calls", "records", "sessions")},
                     ensure_ascii=False, indent=2))
    assert health["upstream_calls"] == 0, "replay 模式发生了真实外呼！"
    print("\n上游地址=%s（死端口），upstream_calls = 0 -> 0 token 得到直接证明" % DEAD_UPSTREAM)

    httpd.shutdown()
    httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
