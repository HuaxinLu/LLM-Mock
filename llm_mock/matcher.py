"""匹配策略：精准 / 模糊 / 位置。

一次请求的匹配顺序由 `--replay-order` 决定，默认 `exact,fuzzy,positional`：
  1. exact     —— 完整请求哈希命中（环境完全确定）
  2. fuzzy     —— 洗掉工具输出后的哈希命中（环境有微小漂移，但轨迹语义相同）
  3. positional—— 不看内容，取该 session 第 N 条录制（只复现轨迹）

`--positional-verify` 可以在位置命中后再校验一次内容（none / fuzzy / exact），
默认 none —— 位置回放的价值恰恰在于"内容会漂移也能走完"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .store import RecordStore

STRATEGIES = ("exact", "fuzzy", "positional")


@dataclass
class MatchResult:
    record: dict
    strategy: str
    detail: dict = field(default_factory=dict)


class Matcher:
    def __init__(
        self,
        store: RecordStore,
        *,
        order: Sequence[str] = ("exact", "fuzzy", "positional"),
        positional_verify: str = "none",
    ) -> None:
        bad = [s for s in order if s not in STRATEGIES]
        if bad:
            raise ValueError(f"未知的回放策略: {bad}（可选 {STRATEGIES}）")
        self.store = store
        self.order = tuple(order)
        self.positional_verify = positional_verify
        self.rejected_positional = 0

    def match(
        self,
        *,
        keys: Sequence[str],
        fuzzy: str,
        session: str | None,
        turn: int,
        verify: str | None = None,
    ) -> MatchResult | None:
        mode = verify or self.positional_verify
        for strategy in self.order:
            if strategy == "exact":
                rec = self.store.lookup_exact(keys)
                if rec:
                    return MatchResult(
                        rec, "exact", {"key": rec.get("key"), "candidates": len(keys)}
                    )
            elif strategy == "fuzzy":
                if not fuzzy:
                    continue
                rec = self.store.lookup_fuzzy(fuzzy)
                if rec:
                    return MatchResult(
                        rec, "fuzzy", {"fuzzy_key": fuzzy}
                    )
            elif strategy == "positional":
                rec = self.store.lookup_position(session, turn)
                if not rec:
                    continue
                if self._reject(rec, keys, fuzzy, mode):
                    self.rejected_positional += 1
                    continue
                return MatchResult(
                    rec,
                    "positional",
                    {"session": session, "turn": turn, "verified": mode},
                )
        return None

    def _reject(self, rec: dict, keys: Sequence[str], fuzzy: str, mode: str) -> bool:
        """返回 True 表示校验失败（应拒绝该位置命中）。"""
        if mode == "none":
            return False
        if mode == "exact":
            return rec.get("key") not in set(keys)
        if mode == "fuzzy":
            return bool(fuzzy) and rec.get("fuzzy_key") != fuzzy
        return False
