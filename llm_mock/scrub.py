"""请求/响应脱敏 —— 保证密钥永远不落盘。

脱敏在**请求进入代理时**做一次（`serve` 路径上的唯一咽喉），因此在它下游的一切
（精确/模糊键、JSONL 存储、miss 留档、诊断报告、/admin/* 输出）看到的都是脱敏后的数据；
而**转发给真实 API 的仍是原始请求体**（不受影响）。

两类规则并用：
  1. **按名字**：字典键名命中 `api_key` / `token` / `secret` / `password` / `authorization` /
     `credential` … 时，其字符串值直接脱敏（哪怕它长得不像密钥）。
  2. **按形状**：已知厂商/协议的密钥形态（`sk-…`、`ghp_…`、`AKIA…`、`Bearer …`、JWT、
     私钥块、`scheme://user:pass@host` 连接串…）。

替换文本：
  * `--scrub-mode hmac`（默认）：`[REDACTED:<kind>:<8位hmac>]` —— 同一个密钥每次替换结果相同，
    因此**不影响录制与回放之间的键一致性**；不同密钥不会撞成一个值。
  * `--scrub-mode redact`：`[REDACTED:<kind>]` —— 更宽松（值变了也还能对上），但失去区分度。

诚实边界：**没有规则能覆盖"任意形状的随机串"**。若密钥没有厂商前缀、也没落在敏感键名下，
需要 `--scrub-extra-regex` 显式指定，或用 `--scrub-dry-run` 先量一遍命中情况。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

log = logging.getLogger("llm_mock")

DEFAULT_SCRUB_KEY = b"llm-mock-default-scrub-key"

#: 字典键名命中即脱敏其字符串值
SECRET_NAME_RE = re.compile(
    r"(?i)(api[-_]?key|access[-_]?key|secret[-_]?key|client[-_]?secret|secret|"
    r"token|password|passwd|credential|authorization|cookie|private[-_]?key|"
    r"session[-_]?key|bearer)"
)

#: 已知密钥形态：(kind, 正则)
KNOWN_PATTERNS: tuple[tuple[str, str], ...] = (
    ("anthropic_key", r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"),
    ("openai_key", r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    ("github_token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    ("github_pat", r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    ("gitlab_token", r"\bglpat-[A-Za-z0-9_\-]{16,}\b"),
    ("slack_token", r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    ("google_api_key", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    ("aws_access_key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("huggingface_token", r"\bhf_[A-Za-z0-9]{20,}\b"),
    ("npm_token", r"\bnpm_[A-Za-z0-9]{30,}\b"),
    ("bearer_token", r"(?i)\b(?:bearer|token)\s+[A-Za-z0-9._\-]{12,}"),
    # 真 Basic 认证是 base64（常含数字/+/=），不能把 "basic authentication" 这种词组打掉
    ("basic_auth", r"(?i)\bbasic\s+(?=[A-Za-z0-9+/=]*[0-9+/=])[A-Za-z0-9+/]{16,}={0,2}"),
    ("jwt", r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
    ("private_key_block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    (
        "connection_string",
        r"\b[a-z][a-z0-9+.\-]{1,20}://[^\s:/@\"'`]+:[^\s:/@\"'`]{6,}@",
    ),
    # 只认“大写环境变量风格”的赋值，且值要有 16 位以上——否则 settings.py 里的
    # `SECRET_KEY = "..."` 这类源码常量会被大面积误伤（实测占 22% 的录制）
    (
        "env_assignment",
        r"\b([A-Z][A-Z0-9_]{2,}(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD))\s*[:=]\s*[\"']?(?P<env_value>[A-Za-z0-9_\-./+]{16,})[\"']?",
    ),
)

_COMPILED: tuple[tuple[str, re.Pattern], ...] = tuple(
    (kind, re.compile(rx)) for kind, rx in KNOWN_PATTERNS
)


def _build_combined(patterns: list[tuple[str, str]]) -> tuple[re.Pattern, list[tuple[str, str]]]:
    """把 N 条正则合成一条带命名组的正则：只扫一遍文本，而不是 N 遍。

    实测这一步把 611 MB 老缓存的全量脱敏从 4 分钟级压到十几秒。
    """
    parts, names = [], []
    for i, (kind, rx) in enumerate(patterns):
        group = kind if kind != "custom" else f"custom_{i}"
        names.append((group, kind))
        # 合并后 inline flag 必须就地作用域化，否则 re 报 "global flags not at the start"
        if rx.startswith("(?i)"):
            rx = "(?i:" + rx[4:] + ")"
        parts.append(f"(?P<{group}>{rx})")
    return re.compile("|".join(parts)), names


#: 名字里含 token 但**不是**密钥的协议字段，必须放行（否则 max_tokens 会被打码）
NON_SECRET_NAMES = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "tool_call_id",
        "token_type",
        "tokens",
        "n_tokens",
    }
)


#: 廉价粗筛：文本里没有这些字面串就整体跳过（纯字面 alternation，很快）
HINT_RE = re.compile(
    "|".join(
        re.escape(h)
        for h in (
            "sk-", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_", "glpat-",
            "xox", "AIza", "AKIA", "ASIA", "hf_", "npm_", "eyJ", "eyj", "-----BEGIN",
            "://", "earer", "asic", "oken",
            "KEY=", "KEY =", "KEY:", "TOKEN=", "TOKEN =", "TOKEN:", "SECRET=",
            "SECRET =", "SECRET:", "PASSWORD=", "PASSWORD =", "PASSWORD:",
            "PASSWD=", "PASSWD =", "PASSWD:",
        )
    )
)

#: 每条正则自己的触发串：粗筛通过后再逐条判断，避免 16 条正则全跑一遍
PATTERN_HINTS: dict[str, tuple[str, ...]] = {
    "anthropic_key": ("sk-ant-",),
    "openai_key": ("sk-",),
    "github_token": ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
    "github_pat": ("github_pat_",),
    "gitlab_token": ("glpat-",),
    "slack_token": ("xox",),
    "google_api_key": ("AIza",),
    "aws_access_key": ("AKIA", "ASIA"),
    "huggingface_token": ("hf_",),
    "npm_token": ("npm_",),
    "bearer_token": ("earer", "oken"),
    "basic_auth": ("asic ",),
    "jwt": ("eyJ", "eyj"),
    "private_key_block": ("-----BEGIN",),
    "connection_string": ("://",),
    "env_assignment": (
        "KEY=", "KEY =", "KEY:", "TOKEN=", "TOKEN =", "TOKEN:", "SECRET=",
        "SECRET =", "SECRET:", "PASSWORD=", "PASSWORD =", "PASSWORD:",
        "PASSWD=", "PASSWD =", "PASSWD:",
    ),
}


def _literal_prefix(pattern: str) -> str:
    """取正则开头连续的字面字符，用作预筛关键字。"""
    out: list[str] = []
    for ch in pattern:
        if ch.isalnum() or ch in "_-":
            out.append(ch)
        else:
            break
    return "".join(out)


class Scrubber:
    def __init__(
        self,
        *,
        enabled: bool = True,
        mode: str = "hmac",
        key: bytes | str = DEFAULT_SCRUB_KEY,
        extra_patterns: Sequence[str] = (),
        dry_run: bool = False,
        cache_max_entries: int = 20000,
        cache_max_string: int = 4096,
    ) -> None:
        if mode not in ("hmac", "redact"):
            raise ValueError("scrub mode 必须是 hmac 或 redact")
        self.enabled = enabled
        self.mode = mode
        self.dry_run = dry_run
        self.cache_max_entries = cache_max_entries
        self.cache_max_string = cache_max_string
        self._cache: dict[str, str] = {}
        self.key = key.encode("utf-8") if isinstance(key, str) else key
        self.counts: dict[str, int] = {}
        sources = list(KNOWN_PATTERNS)
        self._custom_hints: list[str] = []
        self._gate_ok = True
        for rx in extra_patterns:
            try:
                re.compile(rx)
            except re.error as exc:
                raise ValueError(f"--scrub-extra-regex 不是合法正则 {rx!r}: {exc}") from exc
            sources.append(("custom", rx))
            prefix = _literal_prefix(rx)
            if len(prefix) >= 3:
                self._custom_hints.append(prefix)
            else:
                self._gate_ok = False     # 前缀太短，只能每条串都跑
        active: list[tuple[str, re.Pattern]] = []
        for kind, rx in sources:
            if rx.startswith("(?i)"):
                rx = "(?i:" + rx[4:] + ")"
            active.append((kind, re.compile(rx)))
        self._active = active
        self._sources = sources
        self._n_builtin = len(KNOWN_PATTERNS)

    # ------------------------------------------------------------- 替换文本
    def _tag(self, kind: str, value: str) -> str:
        if self.mode == "redact":
            return f"[REDACTED:{kind}]"
        digest = hmac.new(self.key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:8]
        return f"[REDACTED:{kind}:{digest}]"

    def _count(self, kind: str) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def _is_interesting(self, text: str) -> bool:
        if HINT_RE.search(text):
            return True
        if not self._gate_ok:
            return True
        return any(h in text for h in self._custom_hints)

    def _scrub_by_shape(self, text: str) -> str:
        if not self._is_interesting(text):
            return text
        for kind, rx in self._active:
            hints = PATTERN_HINTS.get(kind)
            if hints is not None and not any(h in text for h in hints):
                continue

            def repl(m: re.Match, kind: str = kind) -> str:
                self._count(kind)
                if self.dry_run:
                    return m.group(0)
                secret = m.group(0)
                if kind == "env_assignment" and m.lastgroup == "env_value":
                    # 只替换值，保留 "OPENAI_API_KEY=" 前缀，便于人工核对
                    prefix = secret[: m.start("env_value") - m.start(0)]
                    return prefix + self._tag(kind, m.group("env_value"))
                return self._tag(kind, secret)

            text = rx.sub(repl, text)
        return text

    def scrub_text(self, text: str) -> str:
        if not self.enabled or not isinstance(text, str) or not text:
            return text
        if not self._is_interesting(text):
            return text                      # 绝大多数消息在这里就短路了
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        out = self._scrub_by_shape(text)
        if out != text and len(text) <= self.cache_max_string:
            if len(self._cache) >= self.cache_max_entries:
                self._cache.clear()
            self._cache[text] = out
        return out

    # --------------------------------------------------------------- 结构化
    def scrub_obj(self, obj: Any) -> Any:
        """返回脱敏后的副本；**内容没变时返回原对象**（便于上游判断是否需要重算键）。"""
        if not self.enabled:
            return obj
        out, _ = self._scrub(obj)
        return out

    def _scrub(self, obj: Any) -> tuple[Any, bool]:
        if isinstance(obj, str):
            new = self.scrub_text(obj)
            return new, new is not obj and new != obj
        if isinstance(obj, list):
            changed = False
            items = []
            for v in obj:
                nv, ch = self._scrub(v)
                items.append(nv)
                changed = changed or ch
            return (items, True) if changed else (obj, False)
        if isinstance(obj, dict):
            changed = False
            out: dict[Any, Any] = {}
            for k, v in obj.items():
                by_name = (
                    isinstance(k, str)
                    and k.lower() not in NON_SECRET_NAMES
                    and SECRET_NAME_RE.search(k)
                    and isinstance(v, (str, list))
                )
                if by_name:
                    if isinstance(v, str) and v:
                        self._count("by_name")
                        out[k] = self._tag("by_name", v)
                        changed = True
                    elif isinstance(v, list):
                        self._count("by_name")
                        out[k] = [
                            self._tag("by_name", x) if isinstance(x, str) else self.scrub_obj(x)
                            for x in v
                        ]
                        changed = True
                    else:
                        out[k] = v
                else:
                    nv, ch = self._scrub(v)
                    out[k] = nv
                    changed = changed or ch
            return (out, True) if changed else (obj, False)
        return obj, False

    # ------------------------------------------------------------------ 统计
    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "dry_run": self.dry_run,
            "extra_patterns": max(0, len(self._sources) - self._n_builtin),
            "redactions": dict(self.counts),
            "redactions_total": sum(self.counts.values()),
        }

    def reset_stats(self) -> None:
        self.counts.clear()


def build_scrubber(
    *,
    enabled: bool = True,
    mode: str = "hmac",
    key: str | None = None,
    key_file: str | None = None,
    extra_patterns: Iterable[str] = (),
    dry_run: bool = False,
) -> Scrubber:
    """从 CLI/环境变量构造；key 为空时用默认常量并告警（占位符不具密码学保护）。"""
    resolved = None
    if key_file:
        resolved = Path(key_file).read_text(encoding="utf-8").strip()
    elif key:
        resolved = key
    if enabled and not resolved:
        log.warning(
            "[llm-mock] 未提供 --scrub-key/--scrub-key-file/LLM_MOCK_SCRUB_KEY，"
            "使用内置默认 HMAC key：占位符是确定性的，但不抵抗针对短密钥的暴力枚举"
        )
    return Scrubber(
        enabled=enabled,
        mode=mode,
        key=resolved or DEFAULT_SCRUB_KEY,
        extra_patterns=extra_patterns,
        dry_run=dry_run,
    )


def dumps_scrubbed(scrubber: Scrubber, obj: Any) -> str:
    """调试/测试用：脱敏后序列化，便于断言"密钥没进文件"。"""
    return json.dumps(scrubber.scrub_obj(obj), ensure_ascii=False)
