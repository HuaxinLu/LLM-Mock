"""`python -m llm_mock why` —— 离线/在线 miss 诊断命令行。

    # 离线：直接对一份存储诊断某个请求
    python -m llm_mock why --store records.jsonl --request req.json

    # 在线：让正在跑的代理诊断（会用它自己的最新存储与配置）
    python -m llm_mock why --url http://127.0.0.1:8800 --request req.json

退出码：0 = 诊断出某种策略本来能命中；2 = 确实无法命中；1 = 用法/IO 错误。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .diagnose import Diagnoser, DiagOptions, format_report
from .session import SessionResolver
from .store import RecordStore


def _load_payload(path: str) -> dict:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or "messages" not in payload:
        raise SystemExit("请求体必须是 OpenAI Chat Completions JSON（至少含 messages）")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm_mock why",
        description="解释一次 LLM 请求为什么没有命中录制，并给出可执行建议",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--store", help="录制存储（JSONL）")
    src.add_argument("--url", help="正在运行的 llm-mock 地址，如 http://127.0.0.1:8800")
    p.add_argument("--request", required=True, help="请求体 JSON 文件（- 表示 stdin）")
    p.add_argument("--session", default=None, help="会话 id（不给则从消息里正则抽取）")
    p.add_argument("--turn", type=int, default=0, help="轮次序号")
    p.add_argument(
        "--session-regex", action="append", default=[], metavar="RE",
        help="抽取会话 id 的正则，可重复",
    )
    p.add_argument("--session-header", default="x-llm-mock-session")
    p.add_argument("--implicit-session", default=None)
    p.add_argument("--replay-order", default="exact,fuzzy,positional")
    p.add_argument(
        "--positional-verify", default="auto",
        choices=["auto", "none", "fuzzy", "exact"],
    )
    p.add_argument(
        "--fuzzy-mask-tool-output", dest="mask_tool_output", action="store_true",
        default=True, help="与回放侧保持一致（默认开）",
    )
    p.add_argument(
        "--no-fuzzy-mask-tool-output", dest="mask_tool_output", action="store_false",
    )
    p.add_argument(
        "--fuzzy-normalize-volatile", dest="volatile", action="store_true",
        default=False,
    )
    p.add_argument("--key-default-temperature", default=None)
    p.add_argument("--key-default-max-tokens", default=None)
    p.add_argument("--top", type=int, default=3, help="列出最接近的 N 条录制")
    p.add_argument("--max-scan", type=int, default=0, help="最多扫描多少条（0=全量）")
    p.add_argument("--json", action="store_true", help="输出原始 JSON 报告")
    p.add_argument("--timeout", type=float, default=120.0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    payload = _load_payload(args.request)
    order = tuple(s.strip() for s in args.replay_order.split(",") if s.strip())

    if args.url:
        url = args.url.rstrip("/") + "/admin/why"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            print(f"服务返回 HTTP {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"请求服务失败: {exc}", file=sys.stderr)
            return 1
        report = body.get("report") or {}
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else body.get("text", ""))
        return 0 if report.get("verdict", "").endswith("-hit") else 2

    resolver = SessionResolver(
        patterns=tuple(args.session_regex) or None,
        header=args.session_header,
        implicit=args.implicit_session,
    )
    session = args.session
    source = "cli"
    if session is None:
        session, source = resolver.resolve(None, payload.get("messages"))
    store = RecordStore(
        args.store,
        mask_tool_output=args.mask_tool_output,
        volatile=args.volatile,
        session_of=resolver.from_messages,
    )
    print(
        f"[why] 载入 {len(store.by_key)} 条录制 / {len(store.by_session)} 会话 来自 {args.store}",
        file=sys.stderr,
    )
    diag = Diagnoser(
        store,
        DiagOptions(
            mask_tool_output=args.mask_tool_output,
            volatile=args.volatile,
            replay_order=order or ("exact", "fuzzy", "positional"),
            positional_verify=args.positional_verify,
            key_default_temperature=args.key_default_temperature,
            key_default_max_tokens=args.key_default_max_tokens,
            top=args.top,
            max_scan=args.max_scan,
        ),
    )
    report = diag.explain(
        payload, session=session, session_source=source, turn=args.turn, mode="replay"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_report(report))
    return 0 if report["verdict"].endswith("-hit") else 2


if __name__ == "__main__":
    raise SystemExit(main())
