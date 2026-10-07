#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真·OpenAI 客户端演示：只改 base_url，业务代码一行不动。

    python examples/openai_client_demo.py --base-url http://127.0.0.1:8800/v1 \
        --session my-run-1 --turns 3
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from openai import OpenAI

SYSTEM = "你是一个修 bug 的 coding agent。"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "在 /testbed 执行 shell 命令",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8800/v1")
    ap.add_argument("--api-key", default="llm-mock")
    ap.add_argument("--model", default="fake-model")
    ap.add_argument("--session", default="demo-session")
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--stats-url", default=None, help="默认取 base_url 的 /health")
    args = ap.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "仓库 astropy/astropy @ d16bfe05a744\nissue: 修一下"},
        {"role": "user", "content": "USE_TOOL 请先看看目录"},
    ]
    extra_headers = {"X-LLM-Mock-Session": args.session}
    for turn in range(args.turns):
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=TOOLS,
            temperature=0,
            max_tokens=4096,
            extra_headers=extra_headers,
        )
        msg = resp.choices[0].message
        print(f"[turn {turn}] finish={resp.choices[0].finish_reason} content={msg.content!r}")
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            print(f"          tool_call {tc.function.name}({tc.function.arguments})")
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                    ],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": "file1\nfile2\nreal 0m0.031s"}
            )

    stats_url = args.stats_url or args.base_url.replace("/v1", "") + "/health"
    with urllib.request.urlopen(stats_url, timeout=30) as fh:
        stats = json.load(fh)
    print(json.dumps(
        {k: stats[k] for k in ("mode", "chat_requests", "hits", "misses",
                               "replayed", "forwarded", "upstream_calls")},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
