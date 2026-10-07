"""会话（session）与轮次（turn）解析 —— 位置回放的键。

位置回放 = 不看请求内容，只看"这是哪个实例的第几轮"。
所以需要两个东西：
  * session：一条轨迹的身份。优先级：请求头 > 从消息里正则抽取 > 隐式默认会话
  * turn   ：该轨迹内的第几次 LLM 调用（从 0 开始）。优先级：请求头 > 计数器

默认规则从消息里认 `仓库 <repo> @ <commit>`、`repo=<x> @ <y>`、
`<owner>/<name>@<sha>` 这类写法，commit 截断到 12 位；
也可用请求头 / `--implicit-session` 直接指定。
"""

from __future__ import annotations

import re
import threading
from typing import Any, Sequence

DEFAULT_SESSION_PATTERNS: tuple[str, ...] = (
    r"仓库\s+(\S+)\s+@\s+(\S+)",          # 形如 "仓库 astropy/astropy @ d16bfe05a744"
    r"repo(?:sitory)?\s*[:=]\s*(\S+)\s*[@=]\s*(\S+)",
    r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([0-9a-f]{7,40})\b",
)
DEFAULT_SESSION_TEMPLATE = "{0}@{1}"
COMMIT_TRUNCATE = 12


def _fix_latin1(value: str) -> str:
    """HTTP 头在 Python 里按 latin-1 解码，客户端发 UTF-8 会变乱码。
    这里尝试还原一次（纯 ASCII 值不受影响）。"""
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


class SessionResolver:
    """把一次请求映射到一条轨迹的 id。"""

    def __init__(
        self,
        patterns: Sequence[str] | None = None,
        header: str = "x-llm-mock-session",
        template: str = DEFAULT_SESSION_TEMPLATE,
        implicit: str | None = None,
        scan_messages: int = 3,
    ) -> None:
        pats = tuple(patterns) if patterns else DEFAULT_SESSION_PATTERNS
        self._patterns = tuple(re.compile(p) for p in pats)
        self.header = header.lower()
        self.template = template
        self.implicit = implicit
        self.scan_messages = scan_messages
        self.last_source = "none"

    def resolve(self, headers: Any, messages: Sequence[Any] | None) -> tuple[str | None, str]:
        """返回 (session_id, source)。source ∈ header|regex|implicit|none。

        source 很重要：**显式声明**的会话（header / implicit）可以走内容无关的
        纯位置回放；**从消息里猜出来**的会话（regex）只应作为弱身份，位置命中
        前要再校验一次内容。
        """
        raw = None
        if headers is not None:
            try:
                raw = headers.get(self.header)
            except Exception:
                raw = None
        if raw and str(raw).strip():
            return _fix_latin1(str(raw).strip()), "header"
        sid = self.from_messages(messages)
        if sid:
            return sid, "regex"
        if self.implicit:
            return self.implicit, "implicit"
        return None, "none"

    def from_messages(self, messages: Sequence[Any] | None) -> str | None:
        if not messages:
            return None
        blobs: list[str] = []
        for msg in list(messages)[: self.scan_messages]:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    blobs.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            blobs.append(part["text"])
            if len(blobs) >= 2:
                break
        blob = "\n".join(blobs)
        for rx in self._patterns:
            m = rx.search(blob)
            if not m:
                continue
            groups = [g for g in m.groups() if g]
            if not groups:
                continue
            if len(groups) >= 2:
                groups = [groups[0], groups[1][:COMMIT_TRUNCATE]]
                try:
                    return self.template.format(*groups)
                except Exception:
                    return f"{groups[0]}@{groups[1]}"
            return groups[0]
        return None


class TurnTracker:
    """每个 session 一个请求计数器；位置回放的 turn 就是它。"""

    def __init__(self, header: str = "x-llm-mock-turn", start: int = 0) -> None:
        self.header = header.lower()
        self.start = start
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}

    def next(self, session: str | None, headers: Any = None) -> int:
        explicit = None
        if headers is not None:
            try:
                explicit = headers.get(self.header)
            except Exception:
                explicit = None
        key = session or "__default__"
        with self._lock:
            if explicit is not None and str(explicit).strip() != "":
                try:
                    turn = int(str(explicit).strip())
                except ValueError:
                    turn = self._counters.get(key, self.start)
                self._counters[key] = turn + 1
                return turn
            turn = self._counters.get(key, self.start)
            self._counters[key] = turn + 1
            return turn

    def peek(self, session: str | None) -> int:
        with self._lock:
            return self._counters.get(session or "__default__", self.start)

    def reset(self, session: str | None = None) -> None:
        with self._lock:
            if session is None:
                self._counters.clear()
            else:
                self._counters.pop(session, None)
                self._counters.pop("__default__" if session is None else session, None)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)
