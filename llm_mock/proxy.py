"""LLM Mock 代理服务本体。

对外是一个**本地监听的 HTTP 服务**，说 OpenAI Chat Completions 协议：
    客户端 base_url 从 https://api.deepseek.com 改成 http://127.0.0.1:8800/v1
    业务代码一行不用改。

四种模式：
    live     只转发，不录制
    record   只转发 + 录制（建缓存）
    replay   只回放；未命中**直接报错**，绝不偷偷调 API（保证 0 token）
    hybrid   命中回放，未命中转发 + 录制
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Sequence
from urllib.parse import parse_qs, urlsplit

from .canon import fuzzy_key as compute_fuzzy_key
from .canon import key_candidates
from .diagnose import Diagnoser, DiagOptions, MissLog, format_report
from .matcher import Matcher
from .openai_api import (
    SSECollector,
    build_completion,
    error_body,
    normalize_completion,
    synthesize_sse,
)
from .session import SessionResolver, TurnTracker
from .scrub import Scrubber, build_scrubber
from .store import RecordStore
from .upstream import Upstream

log = logging.getLogger("llm_mock")

MODES = ("live", "record", "replay", "hybrid")


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8800
    mode: str = "hybrid"
    store_path: str = "./llm_mock_records.jsonl"
    upstream_url: str = "https://api.deepseek.com"
    upstream_key: str | None = None
    upstream_headers: dict[str, str] = field(default_factory=dict)
    replay_order: Sequence[str] = ("exact", "fuzzy", "positional")
    replay_latency: str = "recorded"          # recorded | none
    replay_latency_cap: float = 60.0
    mask_tool_output: bool = True
    volatile: bool = False
    session_patterns: Sequence[str] | None = None
    session_header: str = "x-llm-mock-session"
    turn_header: str = "x-llm-mock-turn"
    implicit_session: str | None = None
    positional_verify: str = "auto"           # auto | none | fuzzy | exact
    key_default_temperature: Any = None
    key_default_max_tokens: Any = None
    timeout: float = 600.0
    load_store: bool = True
    max_records: int | None = None
    # ---- miss 诊断 ----
    why_on_miss: str = "auto"                 # auto（仅 replay 深诊）| always | never
    why_top: int = 3
    why_max_scan: int = 5000
    miss_log: int = 20
    # ---- 脱敏 ----
    scrub: bool = True
    scrub_mode: str = "hmac"                  # hmac | redact
    scrub_key: str | None = None
    scrub_key_file: str | None = None
    scrub_extra_patterns: Sequence[str] = ()
    scrub_dry_run: bool = False

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"--mode 必须是 {MODES}，收到 {self.mode!r}")
        if self.replay_latency not in ("recorded", "none"):
            raise ValueError("--replay-latency 必须是 recorded 或 none")
        if self.positional_verify not in ("auto", "none", "fuzzy", "exact"):
            raise ValueError("--positional-verify 必须是 auto/none/fuzzy/exact")
        if self.why_on_miss not in ("auto", "always", "never"):
            raise ValueError("--why-on-miss 必须是 auto/always/never")
        if self.scrub_mode not in ("hmac", "redact"):
            raise ValueError("--scrub-mode 必须是 hmac 或 redact")


@dataclass
class Resp:
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    stream: Iterator[bytes] | None = None

    def close(self) -> None:
        if self.stream is not None and hasattr(self.stream, "close"):
            try:
                self.stream.close()  # type: ignore[union-attr]
            except Exception:
                pass


class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests = 0
        self.chat_requests = 0
        self.hits = {"exact": 0, "fuzzy": 0, "positional": 0}
        self.misses = 0
        self.replayed = 0
        self.recorded = 0
        self.forwarded = 0
        self.upstream_errors = 0
        self.replay_sleep_ms = 0
        self.replayed_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.miss_examples: list[dict] = []
        self.auth_forwarded = 0

    def add_usage(self, usage: dict) -> None:
        with self._lock:
            for k in ("prompt_tokens", "completion_tokens"):
                try:
                    self.replayed_usage[k] += int(usage.get(k) or 0)
                except Exception:
                    pass

    def add_miss(self, info: dict) -> None:
        with self._lock:
            self.misses += 1
            if len(self.miss_examples) < 20:
                self.miss_examples.append(info)

    def snapshot(self, upstream: Upstream, store: RecordStore, cfg: Config) -> dict:
        with self._lock:
            data = {
                "uptime_sec": round(time.time() - self.started, 3),
                "mode": cfg.mode,
                "requests": self.requests,
                "chat_requests": self.chat_requests,
                "hits": dict(self.hits),
                "misses": self.misses,
                "replayed": self.replayed,
                "recorded": self.recorded,
                "forwarded": self.forwarded,
                "upstream_calls": upstream.calls,
                "upstream_errors": self.upstream_errors,
                "replay_sleep_ms": self.replay_sleep_ms,
                "replayed_usage": dict(self.replayed_usage),
                "miss_examples": list(self.miss_examples),
            }
        data["config"] = {
            "upstream": upstream.base_url,
            "replay_order": list(cfg.replay_order),
            "replay_latency": cfg.replay_latency,
            "fuzzy_mask_tool_output": cfg.mask_tool_output,
            "fuzzy_normalize_volatile": cfg.volatile,
            "positional_verify": cfg.positional_verify,
            "session_header": cfg.session_header,
            "turn_header": cfg.turn_header,
            "implicit_session": cfg.implicit_session,
        }
        data.update(store.stats())
        scrubber = getattr(store, "scrubber", None)
        if scrubber is not None:
            data["scrub"] = scrubber.stats()
        return data


class MockService:
    def __init__(self, cfg: Config) -> None:
        cfg.validate()
        self.cfg = cfg
        self.session_resolver = SessionResolver(
            patterns=cfg.session_patterns,
            header=cfg.session_header,
            implicit=cfg.implicit_session,
        )
        self.turn_tracker = TurnTracker(header=cfg.turn_header)
        self.scrubber = build_scrubber(
            enabled=cfg.scrub,
            mode=cfg.scrub_mode,
            key=cfg.scrub_key,
            key_file=cfg.scrub_key_file,
            extra_patterns=cfg.scrub_extra_patterns,
            dry_run=cfg.scrub_dry_run,
        )
        self.store = RecordStore(
            cfg.store_path,
            load=cfg.load_store,
            mask_tool_output=cfg.mask_tool_output,
            volatile=cfg.volatile,
            session_of=self.session_resolver.from_messages,
            max_records=cfg.max_records,
            scrubber=self.scrubber,
        )
        self.matcher = Matcher(
            self.store,
            order=tuple(cfg.replay_order),
            positional_verify=cfg.positional_verify,
        )
        self.upstream = Upstream(
            cfg.upstream_url,
            cfg.upstream_key,
            timeout=cfg.timeout,
            extra_headers=cfg.upstream_headers,
        )
        self.diagnoser = Diagnoser(
            self.store,
            DiagOptions(
                mask_tool_output=cfg.mask_tool_output,
                volatile=cfg.volatile,
                replay_order=tuple(cfg.replay_order),
                positional_verify=cfg.positional_verify,
                key_default_temperature=cfg.key_default_temperature,
                key_default_max_tokens=cfg.key_default_max_tokens,
                top=cfg.why_top,
                max_scan=cfg.why_max_scan,
            ),
        )
        self.miss_log = MissLog(maxlen=cfg.miss_log)
        self.stats = Stats()
        log.info(
            "[llm-mock] mode=%s store=%s records=%d sessions=%d order=%s upstream=%s",
            cfg.mode,
            cfg.store_path,
            len(self.store.by_key),
            len(self.store.by_session),
            ",".join(cfg.replay_order),
            cfg.upstream_url,
        )

    # ------------------------------------------------------------------ chat
    def handle_chat(
        self, raw_body: bytes, headers: Any, path: str = "/v1/chat/completions"
    ) -> Resp:
        stat = self.stats
        with stat._lock:
            stat.chat_requests += 1
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception as exc:
            return self._json(
                400, error_body("invalid_request", f"请求体不是合法 JSON: {exc}")
            )
        if not isinstance(payload, dict):
            return self._json(400, error_body("invalid_request", "请求体必须是 JSON 对象"))

        cfg = self.cfg
        # ⭐ 脱敏只在此处做一次：下游（键、存储、留档、诊断、admin 输出）看到的都是脱敏数据；
        #    真正转发给上游的仍是未改动的 raw_body。
        payload = self.scrubber.scrub_obj(payload)
        messages = payload.get("messages") or []
        tools = payload.get("tools")
        model = payload.get("model")
        temperature = payload.get("temperature")
        max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
        stream = bool(payload.get("stream"))

        session, session_source = self.session_resolver.resolve(headers, messages)
        turn = self.turn_tracker.next(session, headers)

        keys = key_candidates(
            messages,
            tools,
            model,
            temperature,
            max_tokens,
            default_temperature=cfg.key_default_temperature,
            default_max_tokens=cfg.key_default_max_tokens,
        )
        fuzzy = compute_fuzzy_key(
            messages,
            tools,
            model,
            temperature,
            max_tokens,
            mask_tool_output=cfg.mask_tool_output,
            volatile=cfg.volatile,
        )

        ctx = {
            "session": session,
            "session_source": session_source,
            "turn": turn,
            "keys": keys,
            "fuzzy": fuzzy,
            "stream": stream,
            "model": model,
            "path": path,
            "request": self._request_record(
                payload, messages, tools, temperature, max_tokens, stream
            ),
        }

        if cfg.mode in ("replay", "hybrid"):
            verify = cfg.positional_verify
            if verify == "auto":
                # 显式声明的轨迹（header / implicit）允许内容无关的纯位置回放；
                # 从消息里猜出来的会话只当弱身份，位置命中前再校验一次模糊键。
                verify = "none" if session_source in ("header", "implicit") else "fuzzy"
            match = self.matcher.match(
                keys=keys, fuzzy=fuzzy, session=session, turn=turn, verify=verify
            )
            if match is not None:
                return self._replay(match, ctx)
            miss = self._log_miss(payload, ctx)
            if cfg.mode == "replay":
                self.stats.add_miss(
                    {
                        "session": session,
                        "turn": turn,
                        "key": keys[0],
                        "fuzzy_key": fuzzy,
                        "model": model,
                        "miss_id": miss["id"],
                    }
                )
                return self._json(
                    409,
                    error_body(
                        "replay_miss",
                        "replay 模式未命中任何录制，且本模式禁止访问真实 API（保证 0 token）",
                        session=session,
                        session_source=session_source,
                        turn=turn,
                        key=keys[0],
                        fuzzy_key=fuzzy,
                        replay_order=list(cfg.replay_order),
                        miss_id=miss["id"],
                        hint=f"诊断: GET /admin/why?id={miss['id']}（或 python -m llm_mock why --store ... --request req.json）",
                    ),
                )

        # ---------------- 转发 ----------------
        return self._forward(raw_body, headers, ctx)

    # --------------------------------------------------------------- 诊断
    def diagnose(
        self,
        payload: dict,
        *,
        session: str | None,
        session_source: str,
        turn: int,
        exclude_keys: Sequence[str] | None = None,
    ) -> dict:
        return self.diagnoser.explain(
            payload,
            session=session,
            session_source=session_source,
            turn=turn,
            mode=self.cfg.mode,
            exclude_keys=exclude_keys,
        )

    def _log_miss(self, payload: dict, ctx: dict) -> dict:
        """留档一次未命中。深度报告只在 replay 模式（或 --why-on-miss always）算，
        避免 hybrid 首次录制时每次 miss 都全库扫描。"""
        cfg = self.cfg
        deep = cfg.why_on_miss == "always" or (
            cfg.why_on_miss == "auto" and cfg.mode == "replay"
        )
        report = None
        if deep:
            report = self.diagnose(
                payload,
                session=ctx.get("session"),
                session_source=ctx.get("session_source", "none"),
                turn=ctx.get("turn", 0),
            )
        item = self.miss_log.add(
            payload,
            session=ctx.get("session"),
            session_source=ctx.get("session_source", "none"),
            turn=ctx.get("turn", 0),
            mode=cfg.mode,
            keys=ctx.get("keys") or [],
            fuzzy=ctx.get("fuzzy") or "",
            model=ctx.get("model"),
            report=report,
        )
        log.info(
            "[llm-mock] MISS session=%s turn=%s miss_id=%s deep=%s verdict=%s",
            ctx.get("session"),
            ctx.get("turn"),
            item["id"],
            deep,
            (report or {}).get("verdict"),
        )
        return item

    def _why_get(self, query: dict) -> Resp:
        raw_id = (query.get("id") or [None])[0]
        if raw_id in (None, "", "latest"):
            item = self.miss_log.get(None)
        else:
            try:
                item = self.miss_log.get(int(raw_id))
            except ValueError:
                return self._json(
                    400, error_body("invalid_request", f"id 必须是整数，收到 {raw_id!r}")
                )
        if item is None:
            return self._json(
                404,
                error_body(
                    "no_miss",
                    "还没有任何未命中记录",
                    hint="用 POST /admin/why 直接把一个请求体丢进来诊断",
                ),
            )
        if item.get("report") is None:
            if item.get("payload") is None:
                return self._json(
                    413,
                    error_body(
                        "payload_dropped",
                        "这次 miss 的请求体超过留档上限，已丢弃；请改用 POST /admin/why 传请求体",
                        miss_id=item["id"],
                    ),
                )
            item["report"] = self.diagnose(
                item["payload"],
                session=item.get("session"),
                session_source=item.get("session_source", "none"),
                turn=item.get("turn", 0),
                exclude_keys=[item["key"]] if item.get("key") else None,
            )
        text = format_report(item["report"])
        if (query.get("format") or [""])[0] == "text":
            return Resp(
                status=200,
                headers={"Content-Type": "text/plain; charset=utf-8"},
                body=text.encode("utf-8"),
            )
        return self._json(
            200,
            {
                "miss": {k: v for k, v in item.items() if k not in ("payload", "report")},
                "report": item["report"],
                "text": text,
            },
        )

    def _why_post(self, raw_body: bytes) -> Resp:
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception as exc:
            return self._json(
                400, error_body("invalid_request", f"请求体不是合法 JSON: {exc}")
            )
        if not isinstance(payload, dict) or "messages" not in payload:
            return self._json(
                400,
                error_body(
                    "invalid_request",
                    "需要一个 OpenAI Chat Completions 请求体（至少含 messages）",
                ),
            )
        payload = self.scrubber.scrub_obj(payload)
        messages = payload.get("messages") or []
        session, source = self.session_resolver.resolve(None, messages)
        turn = self.turn_tracker.peek(session)
        report = self.diagnose(
            payload, session=session, session_source=source, turn=turn
        )
        return self._json(
            200, {"report": report, "text": format_report(report)}
        )

    # ------------------------------------------------------------- responses
    def _request_record(
        self,
        payload: dict,
        messages: Any,
        tools: Any,
        temperature: Any,
        max_tokens: Any,
        stream: bool,
    ) -> dict:
        return {
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "model": payload.get("model"),
        }

    def _replay(self, match, ctx: dict) -> Resp:
        cfg = self.cfg
        record = match.record
        record = dict(record)
        record["_strategy"] = match.strategy
        self.stats.replayed += 1
        with self.stats._lock:
            self.stats.hits[match.strategy] += 1
        self.stats.add_usage(record.get("usage") or {})

        delay = 0.0
        if cfg.replay_latency == "recorded":
            latency_ms = record.get("latency_ms") or 0
            delay = min(latency_ms / 1000.0, cfg.replay_latency_cap)
        if delay > 0:
            self.stats.replay_sleep_ms += int(delay * 1000)
            time.sleep(delay)

        headers = {
            "X-LLM-Mock-Hit": match.strategy,
            "X-LLM-Mock-Key": str(record.get("key") or ""),
            "X-LLM-Mock-Session": str(ctx.get("session") or ""),
            "X-LLM-Mock-Turn": str(ctx.get("turn")),
        }
        log.info(
            "[llm-mock] HIT %s session=%s turn=%s key=%s delay=%.3fs",
            match.strategy,
            ctx.get("session"),
            ctx.get("turn"),
            str(record.get("key"))[:12],
            delay,
        )
        if ctx.get("stream"):
            body = synthesize_sse(record, ctx.get("model"))
            return Resp(
                status=200,
                headers={**headers, "Content-Type": "text/event-stream; charset=utf-8"},
                stream=body,
            )
        return self._json(200, build_completion(record, ctx.get("model")), headers)

    def _forward(self, raw_body: bytes, headers: Any, ctx: dict) -> Resp:
        cfg = self.cfg
        stat = self.stats
        started = time.time()
        path = self._upstream_path(ctx)
        try:
            conn, resp = self.upstream.open("POST", path, raw_body, headers)
        except Exception as exc:
            with stat._lock:
                stat.upstream_errors += 1
            log.warning("[llm-mock] 上游连接失败: %s", exc)
            return self._json(
                502, error_body("upstream_error", f"连接上游失败: {exc}")
            )
        if ctx.get("stream"):
            return self._forward_stream(conn, resp, ctx, started)

        try:
            raw = resp.read()
            status = resp.status
            ctype = resp.getheader("Content-Type") or "application/json"
        finally:
            conn.close()
        with stat._lock:
            stat.forwarded += 1
        if status != 200:
            with stat._lock:
                stat.upstream_errors += 1
            return Resp(
                status=status,
                headers={"Content-Type": ctype},
                body=raw,
            )
        if cfg.mode in ("record", "hybrid"):
            self._record_from_body(raw, ctx, started)
        return Resp(status=200, headers={"Content-Type": ctype}, body=raw)

    def _forward_stream(self, conn: Any, resp: Any, ctx: dict, started: float):

        def gen() -> Iterator[bytes]:
            collector = SSECollector()
            try:
                for chunk in self.upstream.iter_stream(resp):
                    collector.feed(chunk)
                    yield chunk
            finally:
                try:
                    collector.close()
                except Exception:
                    pass
                conn.close()
                with self.stats._lock:
                    self.stats.forwarded += 1
                if resp.status == 200 and self.cfg.mode in ("record", "hybrid"):
                    self._record_from_stream(collector, ctx, started)

        return Resp(
            status=resp.status,
            headers={
                "Content-Type": resp.getheader("Content-Type")
                or "text/event-stream; charset=utf-8",
            },
            stream=gen(),
        )

    def _record_from_body(self, raw: bytes, ctx: dict, started: float) -> None:
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            return
        response, usage, upstream_model = normalize_completion(body)
        self._store_record(response, usage, ctx, started, upstream_model, stream=False)

    def _record_from_stream(
        self, collector: SSECollector, ctx: dict, started: float
    ) -> None:
        response = collector.result()
        if not response.get("content") and not response.get("tool_calls"):
            return
        self._store_record(
            response, collector.usage, ctx, started, collector.model, stream=True
        )

    def _store_record(
        self,
        response: dict,
        usage: dict,
        ctx: dict,
        started: float,
        upstream_model: str | None,
        *,
        stream: bool,
    ) -> None:
        latency_ms = int((time.time() - started) * 1000)
        rec = self.store.make_record(
            key=ctx["keys"][0],
            fuzzy=ctx["fuzzy"],
            request=ctx["request"],
            # 模型可能把提示里的密钥原样吐回来，所以响应落盘前也过一遍脱敏
            response=self.scrubber.scrub_obj(response),
            usage=usage,
            latency_ms=latency_ms,
            model=upstream_model or ctx.get("model"),
            session=ctx.get("session"),
            turn=ctx.get("turn"),
            source="record",
            extra={"mock": {"stream": stream}},
        )
        self.store.add(rec)
        with self.stats._lock:
            self.stats.recorded += 1
        log.info(
            "[llm-mock] RECORD session=%s turn=%s key=%s latency=%dms tokens=%s",
            ctx.get("session"),
            ctx.get("turn"),
            str(rec.get("key"))[:12],
            latency_ms,
            usage.get("total_tokens"),
        )

    def _upstream_path(self, ctx: dict) -> str:
        return ctx.get("path") or "/v1/chat/completions"

    # ---------------------------------------------------------------- helpers
    def _json(self, status: int, obj: Any, headers: dict | None = None) -> Resp:
        hdrs = {"Content-Type": "application/json; charset=utf-8"}
        if headers:
            hdrs.update(headers)
        return Resp(
            status=status,
            headers=hdrs,
            body=json.dumps(obj, ensure_ascii=False).encode("utf-8"),
        )

    # --------------------------------------------------------------- generic
    def handle_get(self, path: str, query: dict, headers: Any = None) -> Resp:
        if path in ("/health", "/healthz"):
            return self._json(
                200, self.stats.snapshot(self.upstream, self.store, self.cfg)
            )
        if path == "/admin/stats":
            return self._json(
                200, self.stats.snapshot(self.upstream, self.store, self.cfg)
            )
        if path == "/admin/sessions":
            return self._json(
                200,
                {
                    "count": len(self.store.by_session),
                    "sessions": self.store.session_summary(
                        int((query.get("limit") or ["50"])[0])
                    ),
                    "turns_inflight": self.turn_tracker.snapshot(),
                },
            )
        if path == "/admin/records":
            session = (query.get("session") or [None])[0]
            limit = int((query.get("limit") or ["50"])[0])
            offset = int((query.get("offset") or ["0"])[0])
            recs = self.store.iter_records(session)
            page = recs[offset : offset + limit]
            return self._json(
                200,
                {
                    "total": len(recs),
                    "offset": offset,
                    "limit": limit,
                    "records": [
                        {
                            "key": r.get("key"),
                            "fuzzy_key": r.get("fuzzy_key"),
                            "session": r.get("session"),
                            "turn": r.get("turn"),
                            "ts": r.get("ts"),
                            "model": r.get("model"),
                            "latency_ms": r.get("latency_ms"),
                            "usage": r.get("usage"),
                            "source": r.get("source"),
                        }
                        for r in page
                    ],
                },
            )
        if path == "/admin/misses":
            limit = int((query.get("limit") or ["20"])[0])
            return self._json(
                200,
                {
                    "count": len(self.miss_log.items),
                    "why_on_miss": self.cfg.why_on_miss,
                    "misses": self.miss_log.listing(limit),
                },
            )
        if path == "/admin/why":
            return self._why_get(query)
        if path == "/v1/models":
            models = sorted(
                {r.get("model") for r in self.store.by_key.values() if r.get("model")}
            )
            return self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": m, "object": "model", "owned_by": "llm-mock"}
                        for m in models
                    ],
                },
            )
        if path == "/":
            return Resp(
                status=200,
                headers={"Content-Type": "text/plain; charset=utf-8"},
                body=(
                    "llm-mock: LLM 录制/回放代理\n"
                    "  POST /v1/chat/completions   OpenAI 兼容入口\n"
                    "  GET  /health                状态与统计\n"
                    "  GET  /admin/sessions        会话/轮次概览\n"
                    "  GET  /admin/records         录制条目\n"
                    "  GET  /admin/misses          最近未命中列表\n"
                    "  GET  /admin/why[?id=N]      未命中诊断（?format=text 出文本）\n"
                    "  POST /admin/why             对任意请求体做诊断\n"
                    "  POST /admin/reset           重置轮次计数\n"
                    "  POST /admin/reload          重新加载存储\n"
                ).encode("utf-8"),
            )
        if path.startswith("/v1/") and self.cfg.mode != "replay":
            return self._forward_generic("GET", path, b"", headers)
        return self._json(404, error_body("not_found", f"未知路径 {path}"))

    def handle_post(self, path: str, raw_body: bytes, headers: Any) -> Resp:
        if path == "/admin/reset":
            try:
                body = json.loads(raw_body or b"{}")
            except Exception:
                body = {}
            session = body.get("session") if isinstance(body, dict) else None
            self.turn_tracker.reset(session)
            return self._json(200, {"ok": True, "reset": session or "all"})
        if path == "/admin/reload":
            n = self.store.load()
            return self._json(200, {"ok": True, "loaded": n, **self.store.stats()})
        if path == "/admin/mode":
            try:
                body = json.loads(raw_body or b"{}")
                mode = body.get("mode")
            except Exception:
                mode = None
            if mode not in MODES:
                return self._json(
                    400, error_body("invalid_request", f"mode 必须是 {MODES}")
                )
            self.cfg.mode = mode
            return self._json(200, {"ok": True, "mode": mode})
        if path == "/admin/why":
            return self._why_post(raw_body)
        if path.endswith("/chat/completions") or path in (
            "/v1/completions",
            "/chat/completions",
            "/completions",
        ):
            return self.handle_chat(raw_body, headers, path)
        # 其它 /v1/* 端点：replay 模式绝不外呼
        if self.cfg.mode == "replay":
            return self._json(
                501,
                error_body(
                    "replay_unsupported",
                    f"{path} 在 replay 模式下不支持（replay 保证 0 次真实 API 调用）",
                ),
            )
        return self._forward_generic("POST", path, raw_body, headers)

    def _forward_generic(
        self, method: str, path: str, raw_body: bytes, headers: Any
    ) -> Resp:
        try:
            conn, resp = self.upstream.open(method, path, raw_body, headers)
        except Exception as exc:
            with self.stats._lock:
                self.stats.upstream_errors += 1
            return self._json(502, error_body("upstream_error", str(exc)))
        try:
            raw = resp.read()
            status = resp.status
            ctype = resp.getheader("Content-Type") or "application/json"
        finally:
            conn.close()
        with self.stats._lock:
            self.stats.forwarded += 1
        return Resp(status=status, headers={"Content-Type": ctype}, body=raw)


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llm-mock/0.1"
    service: MockService

    # -- 工具 --------------------------------------------------------------
    def _read_body(self) -> bytes:
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            chunks: list[bytes] = []
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    break
                size = int(line.split(b";")[0] or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)
            return b"".join(chunks)
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _emit(self, resp: Resp) -> None:
        try:
            if resp.stream is None:
                body = resp.body or b""
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)
                return
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for chunk in resp.stream:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log.debug("[llm-mock] 客户端断开")
        finally:
            resp.close()

    # -- 路由 --------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        with self.service.stats._lock:
            self.service.stats.requests += 1
        resp = self.service.handle_get(parts.path, parse_qs(parts.query), self.headers)
        self._emit(resp)

    def do_POST(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        with self.service.stats._lock:
            self.service.stats.requests += 1
        body = self._read_body()
        handler = getattr(self.service, "handle_post", None)
        assert handler is not None
        try:
            resp = self.service.handle_post(parts.path, body, self.headers)
        except Exception as exc:  # 保证任何内部异常都变成 500，而不是断连
            log.exception("[llm-mock] 内部错误")
            resp = self.service._json(
                500, error_body("internal_error", f"{type(exc).__name__}: {exc}")
            )
        self._emit(resp)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        log.debug("[http] " + fmt, *args)


class _ThreadingHTTPServer(ThreadingHTTPServer):
    """监听 backlog 默认只有 5，几十个并发连接就会 ConnectionReset；
    评测场景动辄上百并发，这里放大到 256。"""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256


def make_server(cfg: Config, service: MockService | None = None) -> ThreadingHTTPServer:
    svc = service or MockService(cfg)
    handler = type("BoundHandler", (_Handler,), {"service": svc})
    httpd = _ThreadingHTTPServer((cfg.host, cfg.port), handler)
    httpd.service = svc  # type: ignore[attr-defined]
    return httpd


def serve(cfg: Config) -> None:
    httpd = make_server(cfg)
    host, port = httpd.server_address[:2]
    log.info("[llm-mock] listening on http://%s:%s  (mode=%s)", host, port, cfg.mode)
    print(
        f"llm-mock 已启动: http://{host}:{port}/v1  mode={cfg.mode}  "
        f"store={cfg.store_path}  上游={cfg.upstream_url}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
