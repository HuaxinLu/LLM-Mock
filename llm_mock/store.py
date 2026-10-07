"""录制存储（JSONL，append-only）+ 三种索引。

记录 schema（与既有的 `llm_cache.jsonl` 格式兼容 —— 前 7 个字段同名同义）：

    {"key": "<exact sha256>", "ts": 1791205658, "model": "deepseek-v4-pro",
     "request": {"messages": [...], "tools": [...], "temperature": 0,
                 "max_tokens": 4096, "stream": false},
     "response": {"content": "...", "tool_calls": [{"id","name","arguments"}],
                  "finish_reason": "tool_calls"},
     "usage": {...}, "latency_ms": 3842,
     "fuzzy_key": "<sha256>", "session": "astropy/astropy@d16bfe05a744", "turn": 0,
     "source": "record|import", "mock": {"stream": false, "upstream_model": "..."}}

旧记录若缺 `fuzzy_key`/`session`/`turn`，加载时**就地补算**，
因此既有录制文件不用转换即可三种回放全用。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .canon import exact_key as compute_exact_key
from .canon import fuzzy_key as compute_fuzzy_key
from .canon import iter_store_lines, normalize_messages


class RecordStore:
    """线程安全的 JSONL 录制库，内存中维护三份索引。"""

    def __init__(
        self,
        path: str | Path,
        *,
        load: bool = True,
        mask_tool_output: bool = True,
        volatile: bool = False,
        session_of: Callable[[Sequence[Any] | None], str | None] | None = None,
        max_records: int | None = None,
        scrubber: Any = None,
    ) -> None:
        self.path = Path(path)
        self.mask_tool_output = mask_tool_output
        self.volatile = volatile
        self.session_of = session_of
        self.max_records = max_records
        self.scrubber = scrubber
        self.by_key: dict[str, dict] = {}
        #: 落盘时未脱敏的历史记录，在脱敏视角下的等价键 -> 记录。开了脱敏也能精准命中。
        self.by_key_alt: dict[str, dict] = {}
        self.by_fuzzy: dict[str, dict] = {}
        self.by_session: dict[str, list[dict]] = {}
        self._lock = threading.RLock()
        self.loaded_lines = 0
        self.bad_lines = 0
        if load:
            self.load()

    # ------------------------------------------------------------------ load
    def load(self) -> int:
        if not self.path.exists():
            return 0
        n = 0
        for rec in iter_store_lines(self.path):
            if self.max_records and n >= self.max_records:
                break
            try:
                self.index(rec)
            except Exception:
                self.bad_lines += 1
                continue
            n += 1
        self.loaded_lines = n
        for seq in self.by_session.values():
            seq.sort(key=lambda r: r.get("ts") or 0)
        return n

    def index(self, rec: dict) -> dict:
        """把一个记录放进索引（不做落盘）。

        脱敏视角：`req_eff` 是"当前进程脱敏规则下的请求"。历史记录落盘时可能没有脱敏，
        因此这里额外给它算一个等价精准键放进 `by_key_alt`，开了脱敏也不会丢精准命中。
        """
        req = rec.get("request") or {}
        messages = req.get("messages")
        req_eff = req
        if self.scrubber is not None and getattr(self.scrubber, "enabled", False):
            req_eff = self.scrubber.scrub_obj(req)
            messages = req_eff.get("messages")
        rec.setdefault("key", "")
        # 模糊键一律按**当前进程的模糊化参数**重算，而不是信 JSONL 里存的那个 ——
        # 否则"录制时没开噪声归一化、回放时开了"这类配置变化会静默失配。
        rec["fuzzy_key"] = compute_fuzzy_key(
            messages,
            req_eff.get("tools"),
            rec.get("model"),
            req_eff.get("temperature"),
            req_eff.get("max_tokens"),
            mask_tool_output=self.mask_tool_output,
            volatile=self.volatile,
        )
        session = rec.get("session")
        if session is None and self.session_of is not None:
            session = self.session_of(messages)
            rec["session"] = session
        alt_key = None
        if req_eff is not req and rec.get("key"):
            alt_key = compute_exact_key(
                messages,
                req_eff.get("tools"),
                rec.get("model"),
                req_eff.get("temperature"),
                req_eff.get("max_tokens"),
            )
        with self._lock:
            if rec.get("key"):
                self.by_key[rec["key"]] = rec
                if alt_key and alt_key != rec["key"]:
                    self.by_key_alt[alt_key] = rec
            self.by_fuzzy[rec["fuzzy_key"]] = rec
            if session:
                seq = self.by_session.setdefault(session, [])
                if rec not in seq:
                    seq.append(rec)
                seq.sort(key=lambda r: r.get("ts") or 0)
                for i, r in enumerate(seq):
                    if r.get("turn") is None:
                        r["turn"] = i
        return rec

    def rebuild_fuzzy(self, mask_tool_output: bool, volatile: bool) -> int:
        """模糊化参数变化时重建模糊索引。"""
        self.mask_tool_output = mask_tool_output
        self.volatile = volatile
        with self._lock:
            self.by_fuzzy.clear()
            for rec in self.by_key.values():
                req = rec.get("request") or {}
                rec["fuzzy_key"] = compute_fuzzy_key(
                    req.get("messages"),
                    req.get("tools"),
                    rec.get("model"),
                    req.get("temperature"),
                    req.get("max_tokens"),
                    mask_tool_output=mask_tool_output,
                    volatile=volatile,
                )
                self.by_fuzzy[rec["fuzzy_key"]] = rec
        return len(self.by_fuzzy)

    # ------------------------------------------------------------------- add
    def make_record(
        self,
        *,
        key: str,
        fuzzy: str,
        request: dict,
        response: dict,
        usage: dict | None,
        latency_ms: int,
        model: str | None,
        session: str | None = None,
        turn: int | None = None,
        source: str = "record",
        extra: dict | None = None,
    ) -> dict:
        rec = {
            "key": key,
            "ts": int(time.time()),
            "model": model,
            "request": request,
            "response": response,
            "usage": usage or {},
            "latency_ms": int(latency_ms),
            "fuzzy_key": fuzzy,
            "session": session,
            "turn": turn,
            "source": source,
        }
        if extra:
            rec.update(extra)
        return rec

    def add(self, rec: dict, persist: bool = True) -> dict:
        """索引 + 追加落盘（同 key 后写覆盖，重载后 last-wins）。"""
        req = rec.get("request") or {}
        if rec.get("session") is None and self.session_of is not None:
            rec["session"] = self.session_of(req.get("messages"))
        session = rec.get("session")
        if rec.get("turn") is None and session:
            with self._lock:
                rec["turn"] = len(self.by_session.get(session, []))
        self.index(rec)
        if persist:
            self.append(rec)
        return rec

    def append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

    # ---------------------------------------------------------------- lookup
    def lookup_exact(self, keys: Iterable[str]) -> dict | None:
        with self._lock:
            for k in keys:
                rec = self.by_key.get(k) or self.by_key_alt.get(k)
                if rec is not None:
                    return rec
        return None

    def lookup_fuzzy(self, key: str) -> dict | None:
        with self._lock:
            return self.by_fuzzy.get(key)

    def lookup_position(self, session: str | None, turn: int) -> dict | None:
        if not session:
            return None
        with self._lock:
            seq = self.by_session.get(session)
            if not seq or turn < 0 or turn >= len(seq):
                return None
            return seq[turn]

    # ----------------------------------------------------------------- stats
    def stats(self) -> dict:
        with self._lock:
            return {
                "store": str(self.path),
                "records": len(self.by_key),
                "fuzzy_index": len(self.by_fuzzy),
                "sessions": len(self.by_session),
                "loaded_lines": self.loaded_lines,
                "bad_lines": self.bad_lines,
                "mask_tool_output": self.mask_tool_output,
                "volatile_normalize": self.volatile,
                "compat_equivalent_keys": len(self.by_key_alt),
            }

    def session_summary(self, limit: int = 50) -> list[dict]:
        with self._lock:
            items = [
                {"session": s, "turns": len(v)}
                for s, v in self.by_session.items()
            ]
        items.sort(key=lambda x: -x["turns"])
        return items[:limit]

    def iter_records(self, session: str | None = None) -> list[dict]:
        with self._lock:
            if session:
                return list(self.by_session.get(session, []))
            return list(self.by_key.values())

    def import_jsonl(self, path: str | Path, limit: int | None = None) -> int:
        """把另一份同格式的 JSONL 原样并入（补 fuzzy/session，不改写既有字段）。"""
        n = 0
        for rec in iter_store_lines(path):
            if limit and n >= limit:
                break
            rec.setdefault("source", "import")
            self.index(rec)
            rec.pop("_strategy", None)
            self.append(rec)
            n += 1
        return n
