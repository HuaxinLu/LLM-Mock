#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""录制文件体检：确认一份既有 JSONL 录制能被代理直接使用，并量出各策略的收益/风险。

流式扫描，不把全量对象读进内存。输出：
  * key 复算一致率        —— 文件里的 key 是否等于"按记录内容重算"的结果
  * 会话（位置索引）覆盖率 —— 有多少条能落到 (repo, commit) 轨迹上
  * 模糊键收敛度          —— 多少条不同请求会被 fuzzy 归并
  * turn 分布             —— 每条轨迹多少轮
  * --scrub 时的脱敏影响面与等价键自检

用法:
  python tools/store_report.py --store records.jsonl [--limit N] [--scrub]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_mock.canon import exact_key, fuzzy_key  # noqa: E402
from llm_mock.scrub import Scrubber  # noqa: E402
from llm_mock.session import SessionResolver  # noqa: E402

DEFAULT_STORE = os.environ.get("LLM_MOCK_STORE", "./llm_mock_records.jsonl")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=DEFAULT_STORE, help="要体检的 JSONL 录制文件")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全量")
    ap.add_argument("--volatile", action="store_true", help="模糊键启用噪声归一化")
    ap.add_argument("--scrub", action="store_true", help="额外量一遍 API 脱敏的影响面与等价键自检")
    ap.add_argument("--scrub-sample", type=int, default=200, help="脱敏等价键抽样自检条数")
    args = ap.parse_args(argv)

    path = Path(args.store)
    if not path.exists():
        print(f"找不到存储: {path}（用 --store 指定，或设 LLM_MOCK_STORE）", file=sys.stderr)
        return 2

    resolver = SessionResolver()
    n = 0
    key_ok = key_bad = key_missing = 0
    sessions: Counter = Counter()
    no_session = 0
    exact_to_fuzzy: dict[str, set] = defaultdict(set)
    fuzzy_to_exact: dict[str, set] = defaultdict(set)
    ts_seen: list[int] = []
    t0 = time.time()

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if args.limit and n >= args.limit:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            req = rec.get("request") or {}
            messages = req.get("messages")
            model = rec.get("model")
            temp = req.get("temperature")
            mx = req.get("max_tokens")
            ek = exact_key(messages, req.get("tools"), model, temp, mx)
            fk = fuzzy_key(
                messages, req.get("tools"), model, temp, mx, volatile=args.volatile
            )
            stored = rec.get("key")
            if stored is None:
                key_missing += 1
            elif stored == ek:
                key_ok += 1
            else:
                key_bad += 1
            sid = resolver.from_messages(messages)
            if sid:
                sessions[sid] += 1
            else:
                no_session += 1
            exact_to_fuzzy[ek].add(fk)
            fuzzy_to_exact[fk].add(ek)
            ts_seen.append(rec.get("ts") or 0)
            n += 1

    elapsed = time.time() - t0
    merged = sum(1 for fk, eks in fuzzy_to_exact.items() if len(eks) > 1)
    collapsed = sum(len(eks) for fk, eks in fuzzy_to_exact.items() if len(eks) > 1)
    per_session = Counter(sessions.values())
    turns = sorted(sessions.values())

    def pct(x, total):
        return f"{100.0 * x / total:.2f}%" if total else "n/a"

    def quantile(vals, q):
        if not vals:
            return 0
        idx = min(len(vals) - 1, int(q * (len(vals) - 1)))
        return sorted(vals)[idx]

    report = {
        "store": str(path),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
        "records": n,
        "scan_sec": round(elapsed, 1),
        "key_recompute_match": key_ok,
        "key_recompute_match_pct": pct(key_ok, n),
        "key_mismatch": key_bad,
        "key_missing": key_missing,
        "sessions": len(sessions),
        "records_in_sessions": n - no_session,
        "records_in_sessions_pct": pct(n - no_session, n),
        "records_without_session": no_session,
        "turns_p50": quantile(turns, 0.5),
        "turns_p90": quantile(turns, 0.9),
        "turns_max": max(turns) if turns else 0,
        "distinct_exact_keys": len(exact_to_fuzzy),
        "distinct_fuzzy_keys": len(fuzzy_to_exact),
        "fuzzy_collision_groups": merged,
        "records_collapsed_by_fuzzy": collapsed,
        "collapsed_pct": pct(collapsed, n),
        "ts_min": min(ts_seen) if ts_seen else 0,
        "ts_max": max(ts_seen) if ts_seen else 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.scrub:
        # 脱敏影响面 + 等价键自检：老录制按原文落盘，开脱敏后必须仍能精准命中
        from llm_mock.store import RecordStore

        sc = Scrubber()
        ts = time.time()
        store = RecordStore(path, session_of=resolver.from_messages, scrubber=sc,
                            max_records=args.limit or None)
        load_sec = time.time() - ts
        # 统计只算"加载存储"这一遍；下面的自检循环会再次脱敏，不能重复计数
        load_stats = sc.stats()
        changed = 0
        checked = ok = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if args.limit and changed + checked >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                req = rec.get("request") or {}
                eff = sc.scrub_obj(req)
                if eff is req:
                    continue
                changed += 1
                if checked >= args.scrub_sample:
                    continue
                checked += 1
                k = exact_key(eff.get("messages"), eff.get("tools"), rec.get("model"),
                              eff.get("temperature"), eff.get("max_tokens"))
                hit = store.lookup_exact([k])
                if hit is not None and hit.get("key") == rec.get("key"):
                    ok += 1
        report["scrub"] = {
            "enabled": True,
            "mode": sc.mode,
            "records_canonical_form_changed": changed,
            "records_changed_pct": pct(changed, n),
            "redactions": load_stats["redactions"],
            "redactions_total": load_stats["redactions_total"],
            "compat_equiv_keys": store.stats()["compat_equivalent_keys"],
            "equiv_key_sample_checked": checked,
            "equiv_key_sample_ok": ok,
            "equiv_key_sample_ok_pct": pct(ok, checked),
            "load_sec_with_scrub": round(load_sec, 1),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\ntop 5 会话（轮数）:")
    for sid, cnt in sessions.most_common(5):
        print(f"  {sid}: {cnt}")
    print(f"\n每条轨迹轮数分布（top 5 轮数 -> 轨迹数）: {per_session.most_common(5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
