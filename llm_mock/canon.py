"""请求规范化 / 缓存键计算 / 模糊化。

精准键算法与既有 `llm_cache.jsonl` 录制格式保持 **字节级兼容**：

    sha256(canon({"m": messages, "t": tools, "model": model, "temp": temperature, "mx": max_tokens}))

沿用短字段名（m/t/model/temp/mx）就是为了让历史录制文件**无需任何转换**即可命中。

两种键：
  * exact_key —— 完整请求的哈希（"精准匹配回放"）
  * fuzzy_key —— 把**工具输出**（role=tool / role=function 的消息内容）替换为占位符后
                 的哈希（"模糊匹配回放"）。工具输出里有耗时、临时路径、遍历顺序等
                 不可控信息，把它们从键里洗掉，链条才不会在第 N 轮断掉。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Sequence

TOOL_ROLES = ("tool", "function")
TOOL_OUTPUT_PLACEHOLDER = "<TOOL_OUTPUT>"

# 仅用于 fuzzy 模式（且需要显式 --fuzzy-normalize-volatile 打开）。
# 这些是"同一条轨迹重跑时必然变化"的噪声，不是语义内容。
DEFAULT_VOLATILE_PATTERNS: tuple[str, ...] = (
    r"\breal\s+\d+m\d+(?:\.\d+)?s\b",              # bash `time` 输出
    r"\b\d+m\d+\.\d+s\b",                          # pytest / 计时器
    r"\b\d+\.\d{2,}s\b",                           # 0.31s 之类
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",          # ISO 时间戳
    r"/tmp/[^\s\"']*pytest-of-[^\s\"'/]+",         # pytest 临时目录
    r"\bpytest-\d+\b",
    r"\x1b\[[0-9;]*[A-Za-z]",                      # ANSI 转义码
)
VOLATILE_REPLACEMENTS: tuple[str, ...] = (
    "real <TIME>", "<TIME>", "<TIME>", "<TS>", "/tmp/<TMP>", "pytest-<N>", "",
)

_compiled_volatile = tuple(re.compile(p) for p in DEFAULT_VOLATILE_PATTERNS)


def canon(obj: Any) -> str:
    """规范化 JSON 序列化：键排序 + 紧凑分隔符 + 非 ASCII 原样保留。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_volatile(text: str) -> str:
    """把可变的运行时噪声替换为固定占位（仅 fuzzy 模式使用）。"""
    for rx, repl in zip(_compiled_volatile, VOLATILE_REPLACEMENTS):
        text = rx.sub(repl, text)
    return text


def _norm_content(value: Any, volatile: bool) -> Any:
    if isinstance(value, str):
        return strip_volatile(value) if volatile else value
    if isinstance(value, list):  # 多模态 parts
        out = []
        for part in value:
            if isinstance(part, dict):
                p = dict(part)
                if isinstance(p.get("text"), str):
                    p["text"] = _norm_content(p["text"], volatile)
                out.append(p)
            else:
                out.append(part)
        return out
    return value


def normalize_messages(
    messages: Sequence[Any] | None,
    *,
    mask_tool_output: bool = True,
    volatile: bool = False,
) -> list[Any]:
    """生成用于 fuzzy 键的消息列表。

    mask_tool_output=True（模糊匹配的核心）：role 为 tool/function 的消息内容
    统一替换为占位符，并丢弃 tool_call_id（上游生成的 id 同样不可控）。
    assistant 消息里的 tool_calls 保留（它们体现模型决策，是轨迹语义的一部分），
    但其中的 id 同样丢弃。
    """
    out: list[Any] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        role = msg.get("role")
        nm = dict(msg)
        if mask_tool_output and role in TOOL_ROLES:
            nm["content"] = TOOL_OUTPUT_PLACEHOLDER
            nm.pop("tool_call_id", None)
            nm.pop("name", None)
            out.append(nm)
            continue
        if "content" in nm:
            nm["content"] = _norm_content(nm["content"], volatile)
        tcs = nm.get("tool_calls")
        if isinstance(tcs, list):
            new_tcs = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    new_tcs.append(tc)
                    continue
                tc2 = dict(tc)
                if mask_tool_output:
                    tc2.pop("id", None)
                fn = tc2.get("function")
                if isinstance(fn, dict):
                    fn2 = dict(fn)
                    if volatile and isinstance(fn2.get("arguments"), str):
                        fn2["arguments"] = strip_volatile(fn2["arguments"])
                    tc2["function"] = fn2
                new_tcs.append(tc2)
            nm["tool_calls"] = new_tcs
        out.append(nm)
    return out


def exact_key(
    messages: Any,
    tools: Any = None,
    model: str | None = None,
    temperature: Any = None,
    max_tokens: Any = None,
) -> str:
    """精准键。"""
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


def fuzzy_key(
    messages: Any,
    tools: Any = None,
    model: str | None = None,
    temperature: Any = None,
    max_tokens: Any = None,
    *,
    mask_tool_output: bool = True,
    volatile: bool = False,
) -> str:
    """模糊键：不带工具输出（可选再洗掉运行时噪声）。"""
    return exact_key(
        normalize_messages(
            messages, mask_tool_output=mask_tool_output, volatile=volatile
        ),
        tools,
        model,
        temperature,
        max_tokens,
    )


def key_candidates(
    messages: Any,
    tools: Any = None,
    model: str | None = None,
    temperature: Any = None,
    max_tokens: Any = None,
    *,
    default_temperature: Any = None,
    default_max_tokens: Any = None,
) -> list[str]:
    """返回若干等价候选键（去重、保序）。

    客户端是否显式发送 temperature / max_tokens 会影响精准键，而历史录制里
    这两个字段常被显式填了默认值（0 / 4096）。这里把"补默认值"与"原样"两种
    变体都算出来，任一命中即算精准命中。
    """
    variants: list[tuple[Any, Any]] = [(temperature, max_tokens)]
    filled = (
        default_temperature if temperature is None else temperature,
        default_max_tokens if max_tokens is None else max_tokens,
    )
    if filled != variants[0]:
        variants.append(filled)
    variants.append((None, max_tokens))
    variants.append((temperature, None))

    keys: list[str] = []
    for temp, mx in variants:
        k = exact_key(messages, tools, model, temp, mx)
        if k not in keys:
            keys.append(k)
    return keys


def iter_store_lines(path: Any) -> Iterable[dict]:
    """流式读取 JSONL 存储；坏行直接跳过。"""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
