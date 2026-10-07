"""miss 诊断：回答"为什么没命中 / 差在哪几个字段 / 怎么才能命中"。

两条使用路径：
  * 离线：`python -m llm_mock why --store records.jsonl --request req.json`
  * 服务内：replay/hybrid 未命中时自动留档，`GET /admin/why[?id=N]` 取报告

诊断结论分三层：
  1. 三种策略各自为什么没命中（exact / fuzzy / positional）
  2. 与最接近的若干条录制逐字段比对，差异分成 **可被模糊匹配容忍** 与 **致命** 两类
  3. 给出可执行建议（开哪个开关 / 换哪种策略 / 只能重新录制）
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .canon import (
    TOOL_ROLES,
    canon,
    exact_key,
    fuzzy_key,
    key_candidates,
    sha256_hex,
    strip_volatile,
)

STRATEGY_LABEL = {"exact": "精准", "fuzzy": "模糊", "positional": "位置"}


@dataclass
class DiagOptions:
    """诊断时使用的"当前进程配置"（必须与代理一致，否则结论会跑偏）。"""

    mask_tool_output: bool = True
    volatile: bool = False
    replay_order: Sequence[str] = ("exact", "fuzzy", "positional")
    positional_verify: str = "auto"
    key_default_temperature: Any = None
    key_default_max_tokens: Any = None
    top: int = 3
    max_scan: int = 5000


# ---------------------------------------------------------------------------
# 请求指纹与相似度
# ---------------------------------------------------------------------------
def _msg_raw(msg: Any) -> str:
    return sha256_hex(canon(msg))


def _msg_semantic(msg: Any, mask_tool_output: bool) -> str:
    """语义指纹：工具输出按配置屏蔽后再哈希。"""
    if isinstance(msg, dict):
        m = dict(msg)
        if mask_tool_output and m.get("role") in TOOL_ROLES:
            m["content"] = "<TOOL_OUTPUT>"
            m.pop("tool_call_id", None)
            m.pop("name", None)
        if isinstance(m.get("tool_calls"), list):
            tcs = []
            for tc in m["tool_calls"]:
                tc2 = dict(tc) if isinstance(tc, dict) else tc
                if mask_tool_output and isinstance(tc2, dict):
                    tc2.pop("id", None)
                tcs.append(tc2)
            m["tool_calls"] = tcs
        return sha256_hex(canon(m))
    return sha256_hex(canon(msg))


def signature(messages: Sequence[Any] | None, mask_tool_output: bool) -> tuple[list[str], list[str]]:
    raw = [_msg_raw(m) for m in (messages or [])]
    sem = [_msg_semantic(m, mask_tool_output) for m in (messages or [])]
    return raw, sem


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


# ---------------------------------------------------------------------------
# 逐字段比对
# ---------------------------------------------------------------------------
def _content_differs_volatile_only(a: Any, b: Any) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return a != b and strip_volatile(a) == strip_volatile(b)


def diff_messages(
    cur: Sequence[Any] | None,
    cand: Sequence[Any] | None,
    opts: DiagOptions,
) -> tuple[list[dict], list[dict]]:
    """返回 (tolerated, fatal)。tolerated = 模糊匹配能吸收的差异。"""
    tolerated: list[dict] = []
    fatal: list[dict] = []
    cur = list(cur or [])
    cand = list(cand or [])
    if len(cur) != len(cand):
        fatal.append(
            {
                "path": "messages",
                "kind": "structure",
                "detail": f"消息条数不同：本次 {len(cur)} 条，录制 {len(cand)} 条",
            }
        )
    for i in range(min(len(cur), len(cand))):
        a, b = cur[i], cand[i]
        path = f"messages[{i}]"
        if not isinstance(a, dict) or not isinstance(b, dict):
            if a != b:
                fatal.append({"path": path, "kind": "content", "detail": "消息体不同"})
            continue
        if a.get("role") != b.get("role"):
            fatal.append(
                {
                    "path": path,
                    "kind": "structure",
                    "detail": f"role 不同：本次 {a.get('role')!r}，录制 {b.get('role')!r}",
                }
            )
            continue
        role = a.get("role")
        if role in TOOL_ROLES:
            if a.get("content") != b.get("content"):
                item = {
                    "path": f"{path}.content",
                    "kind": "tool_output",
                    "detail": "工具输出内容不同",
                }
                (tolerated if opts.mask_tool_output else fatal).append(item)
            if a.get("tool_call_id") != b.get("tool_call_id"):
                item = {
                    "path": f"{path}.tool_call_id",
                    "kind": "tool_call_id",
                    "detail": "tool_call_id 不同（上游生成，每次都不一样）",
                }
                (tolerated if opts.mask_tool_output else fatal).append(item)
            continue
        ac, bc = a.get("content"), b.get("content")
        if ac != bc:
            if _content_differs_volatile_only(ac, bc):
                item = {
                    "path": f"{path}.content",
                    "kind": "volatile_only",
                    "detail": "内容只差耗时/时间戳/临时路径等噪声",
                }
                (tolerated if opts.volatile else fatal).append(item)
            else:
                fatal.append(
                    {
                        "path": f"{path}.content",
                        "kind": "content",
                        "detail": "正文内容不同（模糊匹配只屏蔽工具输出，不吸收这个）",
                    }
                )
        atcs = a.get("tool_calls") or []
        btcs = b.get("tool_calls") or []
        if len(atcs) != len(btcs):
            fatal.append(
                {
                    "path": f"{path}.tool_calls",
                    "kind": "structure",
                    "detail": f"tool_calls 数量不同：{len(atcs)} vs {len(btcs)}",
                }
            )
        for j in range(min(len(atcs), len(btcs))):
            atc, btc = atcs[j], btcs[j]
            if not isinstance(atc, dict) or not isinstance(btc, dict):
                continue
            if atc.get("id") != btc.get("id"):
                item = {
                    "path": f"{path}.tool_calls[{j}].id",
                    "kind": "tool_call_id",
                    "detail": "tool_call id 不同（上游生成）",
                }
                (tolerated if opts.mask_tool_output else fatal).append(item)
            afn = (atc.get("function") or {}).get("arguments")
            bfn = (btc.get("function") or {}).get("arguments")
            if afn != bfn:
                if _content_differs_volatile_only(afn, bfn):
                    item = {
                        "path": f"{path}.tool_calls[{j}].function.arguments",
                        "kind": "volatile_only",
                        "detail": "工具参数只差噪声",
                    }
                    (tolerated if opts.volatile else fatal).append(item)
                else:
                    fatal.append(
                        {
                            "path": f"{path}.tool_calls[{j}].function.arguments",
                            "kind": "content",
                            "detail": "工具参数不同",
                        }
                    )
    return tolerated, fatal


# ---------------------------------------------------------------------------
# 主诊断器
# ---------------------------------------------------------------------------
class Diagnoser:
    def __init__(self, store: Any, opts: DiagOptions | None = None) -> None:
        self.store = store
        self.opts = opts or DiagOptions()

    # -- 候选择优 ----------------------------------------------------------
    def nearest(
        self,
        messages: Sequence[Any] | None,
        model: str | None,
        exclude_keys: set[str] | None = None,
    ) -> list[dict]:
        opts = self.opts
        exclude_keys = exclude_keys or set()
        cur_raw, cur_sem = signature(messages, opts.mask_tool_output)
        n_cur = len(cur_raw)
        scored: list[tuple[float, float, dict]] = []
        scanned = 0
        records = self.store.iter_records()
        for rec in records:
            if opts.max_scan and scanned >= opts.max_scan:
                break
            if rec.get("key") in exclude_keys:
                continue
            r_model = rec.get("model")
            if model and r_model and r_model != model:
                continue
            req = rec.get("request") or {}
            r_msgs = req.get("messages") or []
            if abs(len(r_msgs) - n_cur) > 4:
                continue
            scanned += 1
            r_raw, r_sem = signature(r_msgs, opts.mask_tool_output)
            s_sem = _jaccard(cur_sem, r_sem)
            s_raw = _jaccard(cur_raw, r_raw)
            scored.append((s_sem, s_raw, rec))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out = []
        for s_sem, s_raw, rec in scored[: opts.top]:
            req = rec.get("request") or {}
            tolerated, fatal = diff_messages(messages, req.get("messages"), opts)
            out.append(
                {
                    "session": rec.get("session"),
                    "turn": rec.get("turn"),
                    "ts": rec.get("ts"),
                    "key": rec.get("key"),
                    "fuzzy_key": rec.get("fuzzy_key"),
                    "latency_ms": rec.get("latency_ms"),
                    "usage": rec.get("usage"),
                    "similarity_semantic": round(s_sem, 4),
                    "similarity_raw": round(s_raw, 4),
                    "tolerated": tolerated,
                    "fatal": fatal,
                }
            )
        return out

    # -- 主入口 ------------------------------------------------------------
    def explain(
        self,
        payload: dict,
        *,
        session: str | None = None,
        session_source: str = "none",
        turn: int = 0,
        mode: str = "replay",
        exclude_keys: Iterable[str] | None = None,
    ) -> dict:
        opts = self.opts
        exclude = set(exclude_keys or ())
        messages = payload.get("messages") or []
        tools = payload.get("tools")
        model = payload.get("model")
        temperature = payload.get("temperature")
        max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")

        keys = key_candidates(
            messages,
            tools,
            model,
            temperature,
            max_tokens,
            default_temperature=opts.key_default_temperature,
            default_max_tokens=opts.key_default_max_tokens,
        )
        fuzzy = fuzzy_key(
            messages,
            tools,
            model,
            temperature,
            max_tokens,
            mask_tool_output=opts.mask_tool_output,
            volatile=opts.volatile,
        )

        exact_hit = self.store.lookup_exact(keys)
        if exact_hit is not None and exact_hit.get("key") in exclude:
            exact_hit = None
        # 存储索引里的模糊键是按**存储自己的参数**算的；诊断参数可能不同（离线 CLI 常见）。
        # 两种都试，并把不一致如实报出来，避免给出"参数不对导致"的错误结论。
        store_mask = getattr(self.store, "mask_tool_output", opts.mask_tool_output)
        store_vol = getattr(self.store, "volatile", opts.volatile)
        fuzzy_store = fuzzy
        if (store_mask, store_vol) != (opts.mask_tool_output, opts.volatile):
            fuzzy_store = fuzzy_key(
                messages,
                tools,
                model,
                temperature,
                max_tokens,
                mask_tool_output=store_mask,
                volatile=store_vol,
            )
        fuzzy_hit = self.store.lookup_fuzzy(fuzzy)
        if fuzzy_hit is not None and fuzzy_hit.get("key") in exclude:
            fuzzy_hit = None
        if fuzzy_hit is None and fuzzy_store != fuzzy:
            fuzzy_hit = self.store.lookup_fuzzy(fuzzy_store)
            if fuzzy_hit is not None and fuzzy_hit.get("key") in exclude:
                fuzzy_hit = None
        seq = self.store.by_session.get(session) if session else None
        pos_rec = None
        pos_state = "unknown-session" if session else "no-session"
        if session and seq is not None:
            if 0 <= turn < len(seq):
                pos_rec = seq[turn]
                pos_state = "available"
            else:
                pos_state = "out-of-range"

        verify = opts.positional_verify
        if verify == "auto":
            verify = "none" if session_source in ("header", "implicit") else "fuzzy"
        pos_rejected = False
        if pos_rec is not None and verify != "none":
            if verify == "exact":
                pos_rejected = pos_rec.get("key") not in set(keys)
            elif verify == "fuzzy":
                pos_rejected = bool(fuzzy) and pos_rec.get("fuzzy_key") != fuzzy
            if pos_rejected:
                pos_state = "rejected-by-verify"

        strategies = {
            "exact": {
                "hit": exact_hit is not None,
                "tried_keys": len(keys),
                "reason": None if exact_hit else "完整请求（含工具输出、tool_call id）与任何录制都不完全一致",
            },
            "fuzzy": {
                "hit": fuzzy_hit is not None,
                "reason": None
                if fuzzy_hit
                else (
                    "屏蔽工具输出后仍与任何录制不一致"
                    + ("（已开噪声归一化）" if opts.volatile else "（未开噪声归一化）")
                ),
            },
            "positional": {
                "hit": pos_rec is not None and not pos_rejected,
                "state": pos_state,
                "session_turns": (len(seq) if seq else 0),
                "requested_turn": turn,
                "reason": {
                    "no-session": "请求里没有可识别的会话身份（也没有隐式会话）",
                    "unknown-session": (
                        f"会话 {session} 不在录制库里"
                        f"（库里共 {len(self.store.by_session)} 条会话）"
                    ),
                    "out-of-range": f"该会话只录到 {len(seq) if seq else 0} 轮，本次是第 {turn} 轮",
                    "rejected-by-verify": f"第 {turn} 轮存在，但被 positional_verify={verify} 拒绝（内容差异过大）",
                    "available": None,
                }.get(pos_state, None),
            },
        }

        candidates = self.nearest(messages, model, exclude_keys=exclude)
        best = candidates[0] if candidates else None

        # -- 结论与建议 ----------------------------------------------------
        tolerated = list(best["tolerated"]) if best else []
        fatal = list(best["fatal"]) if best else []
        kinds = {d["kind"] for d in fatal}
        advice: list[str] = []
        if exact_hit is not None:
            verdict = "exact-hit"
            advice.append("精准命中：无需处理（诊断只用于解释 miss）")
        elif fuzzy_hit is not None:
            verdict = "fuzzy-hit"
            advice.append("模糊命中：精准 miss 的原因就是工具输出/id 变化，属预期")
        elif not best:
            verdict = "no-candidate"
            advice.append("存储里没有任何结构相近的录制 —— 这是新轨迹，只能转发真实 API 录制（hybrid）")
        elif not fatal:
            verdict = "tolerable-only"
            advice.append(
                "与最接近的录制只差可容忍项，模糊键本应相同 → 检查回放进程的 --fuzzy-* 参数"
                "是否与录制时一致（模糊键按当前参数重算）"
            )
        else:
            verdict = "content-drift"
            if "content" in kinds:
                advice.append(
                    "差异落在 system/user 正文或工具参数上，模糊匹配不吸收 → "
                    "要么重新录制该轮（hybrid），要么改用位置回放"
                )
            if "volatile_only" in kinds and not opts.volatile:
                advice.append("有差异只是耗时/时间戳/临时路径 → 加 --fuzzy-normalize-volatile")
            if "tool_output" in kinds or "tool_call_id" in kinds:
                advice.append("差异在工具输出/id 上，属可容忍项 → 确认没关掉 --fuzzy-mask-tool-output")
            if "structure" in kinds:
                advice.append("消息条数/role/tool_calls 结构不同 → 模糊与精准都救不了，改位置回放或重录")
        if pos_state == "no-session":
            advice.append(
                "想用位置回放：带上 X-LLM-Mock-Session: <会话 id> 与 X-LLM-Mock-Turn: <轮次>，"
                "或配 --session-regex / --implicit-session"
            )
        elif pos_state == "unknown-session":
            advice.append(
                f"会话 {session} 不在录制库里 → 确认拼写，或先录一遍这条轨迹"
                "（GET /admin/sessions 可看库里已有会话）"
            )
        elif pos_state == "out-of-range":
            advice.append(f"位置回放不可用：第 {turn} 轮超出了该会话已录制的 {len(seq) if seq else 0} 轮")
        elif pos_state == "rejected-by-verify":
            advice.append(
                "位置回放本来命中，被内容校验拦下 → 若确实只想复现轨迹，用 --positional-verify none"
            )
        elif pos_state == "available":
            advice.append(
                f"位置回放可用：第 {turn} 轮已录制（带 X-LLM-Mock-Session: {session} 即可命中）"
            )
        if fuzzy_store != fuzzy:
            advice.append(
                "诊断参数与存储索引参数不一致：存储是 "
                f"mask_tool_output={store_mask} / normalize_volatile={store_vol}，"
                f"本次诊断用的是 mask_tool_output={opts.mask_tool_output} / "
                f"normalize_volatile={opts.volatile} → 结论可能偏移，建议两边保持一致"
            )
        if exclude:
            advice.append(
                f"已忽略 {len(exclude)} 条本次 miss 之后才录进去的记录"
                "（hybrid 转发+录制所致）—— 同一个请求下次会精准命中"
            )

        return {
            "type": "llm_mock_miss_report",
            "ts": int(time.time()),
            "mode": mode,
            "request": {
                "session": session,
                "session_source": session_source,
                "turn": turn,
                "model": model,
                "stream": bool(payload.get("stream")),
                "messages": len(messages),
                "tools": len(tools or []),
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "keys": {
                "exact_candidates": keys,
                "exact": keys[0] if keys else None,
                "fuzzy": fuzzy,
                "fuzzy_as_stored": fuzzy_store,
            },
            "config_used": {
                "replay_order": list(opts.replay_order),
                "fuzzy_mask_tool_output": opts.mask_tool_output,
                "fuzzy_normalize_volatile": opts.volatile,
                "positional_verify": verify,
                "max_scan": opts.max_scan,
            },
            "strategies": strategies,
            "nearest": candidates,
            "verdict": verdict,
            "advice": advice,
        }


# ---------------------------------------------------------------------------
# 文本渲染
# ---------------------------------------------------------------------------
def format_report(rep: dict) -> str:
    req = rep["request"]
    cfg = rep["config_used"]
    out: list[str] = []
    out.append("=== llm-mock miss 诊断 ===")
    out.append(
        f"模式={rep['mode']}  会话={req['session']}（来源={req['session_source']}）  "
        f"轮次={req['turn']}  model={req['model']}  stream={req['stream']}"
    )
    out.append(f"消息 {req['messages']} 条 / 工具 {req['tools']} 个  回放顺序={','.join(cfg['replay_order'])}")
    out.append(
        f"模糊参数: mask_tool_output={cfg['fuzzy_mask_tool_output']} "
        f"normalize_volatile={cfg['fuzzy_normalize_volatile']} "
        f"positional_verify={cfg['positional_verify']}"
    )
    out.append(f"键: exact={str(rep['keys']['exact'])[:16]}…  fuzzy={str(rep['keys']['fuzzy'])[:16]}…")
    out.append("")
    out.append("--- 三种策略 ---")
    for name in cfg["replay_order"]:
        st = rep["strategies"].get(name)
        if st is None:
            continue
        mark = "命中" if st["hit"] else "未命中"
        line = f"  [{STRATEGY_LABEL.get(name, name)}] {mark}"
        if name == "exact" and not st["hit"]:
            line += f"  候选键 {st['tried_keys']} 个"
        if name == "positional":
            line += f"  状态={st['state']}  该会话 {st['session_turns']} 轮"
        out.append(line)
        if st.get("reason"):
            out.append(f"        原因: {st['reason']}")
    out.append("")
    if not rep["nearest"]:
        out.append("--- 没有结构相近的录制 ---")
    else:
        out.append(f"--- 最接近的 {len(rep['nearest'])} 条录制 ---")
        for i, c in enumerate(rep["nearest"], 1):
            out.append(
                f"  #{i} 相似度(语义)={c['similarity_semantic']} 原始={c['similarity_raw']}  "
                f"session={c['session']} turn={c['turn']} key={str(c['key'])[:12]}…"
            )
            for d in c["tolerated"]:
                out.append(f"       ~ 可容忍 {d['path']}: {d['detail']}")
            for d in c["fatal"]:
                out.append(f"       ! 致命   {d['path']}: {d['detail']}")
            if not c["tolerated"] and not c["fatal"]:
                out.append("       (逐字段无差异)")
    out.append("")
    out.append(f"--- 结论: {rep['verdict']} ---")
    for a in rep["advice"]:
        out.append(f"  * {a}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 服务内留档
# ---------------------------------------------------------------------------
class MissLog:
    """最近 N 次未命中的请求留档；深度报告按需计算（避免 hybrid 每次 miss 都全库扫描）。"""

    def __init__(self, maxlen: int = 20, max_payload_bytes: int = 2_000_000) -> None:
        self.items: deque[dict] = deque(maxlen=maxlen)
        self.max_payload_bytes = max_payload_bytes
        self._seq = 0

    def add(
        self,
        payload: dict,
        *,
        session: str | None,
        session_source: str,
        turn: int,
        mode: str,
        keys: Sequence[str],
        fuzzy: str,
        model: str | None,
        report: dict | None = None,
    ) -> dict:
        self._seq += 1
        try:
            size = len(json.dumps(payload, ensure_ascii=False))
        except Exception:
            size = 0
        item = {
            "id": self._seq,
            "ts": int(time.time()),
            "mode": mode,
            "session": session,
            "session_source": session_source,
            "turn": turn,
            "model": model,
            "key": keys[0] if keys else None,
            "fuzzy_key": fuzzy,
            "payload_bytes": size,
            "payload": payload if size <= self.max_payload_bytes else None,
            "report": report,
        }
        self.items.append(item)
        return item

    def get(self, miss_id: int | None = None) -> dict | None:
        if not self.items:
            return None
        if miss_id is None:
            return self.items[-1]
        for it in self.items:
            if it["id"] == miss_id:
                return it
        return None

    def listing(self, limit: int = 20) -> list[dict]:
        items = list(self.items)[-limit:]
        return [{k: v for k, v in it.items() if k not in ("payload", "report")} for it in items]
