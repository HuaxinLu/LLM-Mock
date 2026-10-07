"""命令行入口：python -m llm_mock ..."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .proxy import MODES, Config, MockService, serve

ENV_PREFIX = "LLM_MOCK_"


def _env(name: str, default=None):
    return os.environ.get(ENV_PREFIX + name, default)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm_mock",
        description="LLM Mock —— 本地监听的 LLM 录制/回放代理（OpenAI 兼容）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default=_env("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(_env("PORT", "8800")))
    p.add_argument(
        "--mode",
        default=_env("MODE", "hybrid"),
        choices=list(MODES),
        help="live=只转发 record=转发+录制 replay=只回放(未命中报错) hybrid=命中回放/未命中转发",
    )
    p.add_argument(
        "--store",
        default=_env("STORE", "./llm_mock_records.jsonl"),
        help="录制存储（JSONL，追加写；兼容既有 llm_cache.jsonl 格式）",
    )
    p.add_argument(
        "--upstream",
        default=_env("UPSTREAM", "https://api.deepseek.com"),
        help="真实 API base_url",
    )
    p.add_argument(
        "--upstream-key",
        default=_env("UPSTREAM_KEY"),
        help="真实 API key（也会读环境变量 LLM_MOCK_UPSTREAM_KEY / OPENAI_API_KEY）",
    )
    p.add_argument(
        "--upstream-key-file",
        default=_env("UPSTREAM_KEY_FILE"),
        help="从文件读取 API key（避免命令行泄露）",
    )
    p.add_argument(
        "--upstream-header",
        action="append",
        default=[],
        metavar="K:V",
        help="附加到上游请求的头，可重复",
    )
    p.add_argument(
        "--replay-order",
        default=_env("REPLAY_ORDER", "exact,fuzzy,positional"),
        help="匹配顺序，逗号分隔：exact / fuzzy / positional 的任意组合",
    )
    p.add_argument(
        "--replay-latency",
        default=_env("REPLAY_LATENCY", "recorded"),
        choices=["recorded", "none"],
        help="回放时是否按录制时延 sleep",
    )
    p.add_argument(
        "--replay-latency-cap",
        type=float,
        default=float(_env("REPLAY_LATENCY_CAP", "60")),
        help="单次回放时延上限（秒）",
    )
    p.add_argument(
        "--fuzzy-mask-tool-output",
        dest="mask_tool_output",
        action="store_true",
        default=_env("FUZZY_MASK_TOOL_OUTPUT", "1") not in ("0", "false", "False"),
        help="模糊匹配：把 role=tool/function 的消息内容替换为占位符（默认开）",
    )
    p.add_argument(
        "--no-fuzzy-mask-tool-output",
        dest="mask_tool_output",
        action="store_false",
        help="关闭工具输出屏蔽（模糊键退化为精准键）",
    )
    p.add_argument(
        "--fuzzy-normalize-volatile",
        dest="volatile",
        action="store_true",
        default=_env("FUZZY_NORMALIZE_VOLATILE", "0") in ("1", "true", "True"),
        help="模糊匹配：额外洗掉耗时/时间戳/临时路径等噪声",
    )
    p.add_argument(
        "--session-regex",
        action="append",
        default=[],
        metavar="RE",
        help="从消息中抽取会话 id 的正则（1 或 2 个捕获组），可重复",
    )
    p.add_argument("--session-header", default=_env("SESSION_HEADER", "x-llm-mock-session"))
    p.add_argument("--turn-header", default=_env("TURN_HEADER", "x-llm-mock-turn"))
    p.add_argument(
        "--implicit-session",
        default=_env("IMPLICIT_SESSION"),
        help="抽不到会话 id 时使用的固定 id（设置后位置回放才有意义）",
    )
    p.add_argument(
        "--positional-verify",
        default=_env("POSITIONAL_VERIFY", "auto"),
        choices=["auto", "none", "fuzzy", "exact"],
        help="位置命中后的内容校验：auto=显式会话不校验/推断会话校验模糊键",
    )
    p.add_argument(
        "--key-default-temperature",
        default=_env("KEY_DEFAULT_TEMPERATURE"),
        help="精准键的 temperature 默认值（历史录制常为 0）",
    )
    p.add_argument(
        "--key-default-max-tokens",
        default=_env("KEY_DEFAULT_MAX_TOKENS"),
        help="精准键的 max_tokens 默认值（历史录制常为 4096）",
    )
    p.add_argument("--timeout", type=float, default=float(_env("TIMEOUT", "600")))
    p.add_argument(
        "--max-records",
        type=int,
        default=int(_env("MAX_RECORDS", "0")) or None,
        help="最多加载多少条录制（0 = 不限）",
    )
    p.add_argument("--no-load-store", dest="load_store", action="store_false", default=True)
    p.add_argument("--log-level", default=_env("LOG_LEVEL", "INFO"))
    p.add_argument(
        "--why-on-miss",
        default=_env("WHY_ON_MISS", "auto"),
        choices=["auto", "always", "never"],
        help="未命中时是否自动算诊断报告：auto=只在 replay 模式算，always=每次都算，never=只留档",
    )
    p.add_argument(
        "--why-top", type=int, default=int(_env("WHY_TOP", "3")),
        help="诊断时列出最接近的 N 条录制",
    )
    p.add_argument(
        "--why-max-scan", type=int, default=int(_env("WHY_MAX_SCAN", "5000")),
        help="诊断时最多扫描多少条录制（0=全量）",
    )
    p.add_argument(
        "--miss-log", type=int, default=int(_env("MISS_LOG", "20")),
        help="保留最近多少次未命中的请求体用于诊断",
    )
    p.add_argument(
        "--scrub", dest="scrub", action="store_true",
        default=_env("SCRUB", "1") not in ("0", "false", "False"),
        help="落盘前对请求/响应脱敏（默认开；密钥永不写入 JSONL）",
    )
    p.add_argument(
        "--no-scrub", dest="scrub", action="store_false",
        help="关闭脱敏（录制库会保存原始请求，仅在你确知内容无密钥时使用）",
    )
    p.add_argument(
        "--scrub-mode", default=_env("SCRUB_MODE", "hmac"), choices=["hmac", "redact"],
        help="hmac=相同密钥替换结果相同（不影响回放键），redact=统一占位符",
    )
    p.add_argument(
        "--scrub-key", default=None,
        help="HMAC 占位符用的 key（也可用 env LLM_MOCK_SCRUB_KEY；不建议写在命令行）",
    )
    p.add_argument(
        "--scrub-key-file", default=None,
        help="从文件读取 HMAC key（推荐）",
    )
    p.add_argument(
        "--scrub-extra-regex", action="append", default=[], metavar="RE",
        help="额外的密钥正则（可重复），用于项目自有格式",
    )
    p.add_argument(
        "--scrub-dry-run", action="store_true",
        help="只统计命中、不真的替换（用来先量影响面）",
    )
    p.add_argument(
        "--print-config", action="store_true", help="打印生效配置后退出"
    )
    return p


def _parse_headers(items) -> dict:
    out = {}
    for item in items or []:
        if ":" not in item:
            raise SystemExit(f"--upstream-header 需要 K:V 形式: {item}")
        k, v = item.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _resolve_key(args) -> str | None:
    if args.upstream_key_file:
        return Path(args.upstream_key_file).read_text(encoding="utf-8").strip()
    return (
        args.upstream_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
    )


def config_from_args(args) -> Config:
    order = tuple(s.strip() for s in str(args.replay_order).split(",") if s.strip())
    return Config(
        host=args.host,
        port=args.port,
        mode=args.mode,
        store_path=args.store,
        upstream_url=args.upstream,
        upstream_key=_resolve_key(args),
        upstream_headers=_parse_headers(args.upstream_header),
        replay_order=order or ("exact", "fuzzy", "positional"),
        replay_latency=args.replay_latency,
        replay_latency_cap=args.replay_latency_cap,
        mask_tool_output=args.mask_tool_output,
        volatile=args.volatile,
        session_patterns=tuple(args.session_regex) or None,
        session_header=args.session_header,
        turn_header=args.turn_header,
        implicit_session=args.implicit_session,
        positional_verify=args.positional_verify,
        key_default_temperature=args.key_default_temperature,
        key_default_max_tokens=args.key_default_max_tokens,
        timeout=args.timeout,
        load_store=args.load_store,
        max_records=args.max_records,
        why_on_miss=args.why_on_miss,
        why_top=args.why_top,
        why_max_scan=args.why_max_scan,
        miss_log=args.miss_log,
        scrub=args.scrub,
        scrub_mode=args.scrub_mode,
        scrub_key=args.scrub_key or _env("SCRUB_KEY"),
        scrub_key_file=args.scrub_key_file or _env("SCRUB_KEY_FILE"),
        scrub_extra_patterns=tuple(args.scrub_extra_regex or ()),
        scrub_dry_run=args.scrub_dry_run,
    )


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("why", "diagnose", "miss"):
        from .why_cli import main as why_main

        return why_main(argv[1:])
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = config_from_args(args)
    cfg.validate()
    if args.print_config:
        import json
        import dataclasses

        data = dataclasses.asdict(cfg)
        data["upstream_key"] = "***" if cfg.upstream_key else None
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    try:
        serve(cfg)
    except OSError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1
    return 0


def build_service(argv=None) -> MockService:
    """给测试/工具用：从命令行参数直接构造服务（不起监听）。"""
    args = build_parser().parse_args(argv)
    return MockService(config_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
