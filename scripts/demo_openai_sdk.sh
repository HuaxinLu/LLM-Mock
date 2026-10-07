#!/usr/bin/env bash
# 用**真 openai SDK**验证兼容性：只把 base_url 指向 llm-mock，业务代码不动。
#   record 阶段：2 轮真实调用（假上游）
#   replay 阶段：同一脚本重跑 -> 全部回放，upstream 指向死端口，0 外呼
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"; export PYTHONPATH="$ROOT"

UP_PORT=18899; REC_PORT=18800; REP_PORT=18801
STORE=/tmp/llm-mock-sdk.jsonl; LOGDIR=/tmp/llm-mock-sdk
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"; rm -f "$STORE"
PIDS=()
cleanup() { for pid in "${PIDS[@]:-}"; do [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null; done; sleep 0.3; }
trap cleanup EXIT
wait_port() { for _ in $(seq 1 50); do curl -s -o /dev/null "http://127.0.0.1:$1/health" && return 0; sleep 0.2; done; return 1; }

python3 -m llm_mock.testing --port "$UP_PORT" > "$LOGDIR/upstream.log" 2>&1 & PIDS+=($!)
wait_port "$UP_PORT"
python3 -m llm_mock --mode record --port "$REC_PORT" --upstream "http://127.0.0.1:$UP_PORT" \
  --store "$STORE" --replay-latency none > "$LOGDIR/record.log" 2>&1 & PIDS+=($!)
wait_port "$REC_PORT"

echo "### A. record：openai SDK 打 llm-mock(record)"
python3 examples/openai_client_demo.py --base-url "http://127.0.0.1:$REC_PORT/v1" \
  --session sdk-demo --turns 2
echo "录制条数: $(wc -l < "$STORE")"

echo
echo "### B. replay：同一脚本，base_url 换成 llm-mock(replay)，上游是死端口"
python3 -m llm_mock --mode replay --port "$REP_PORT" --upstream "http://127.0.0.1:1" \
  --store "$STORE" --replay-latency none > "$LOGDIR/replay.log" 2>&1 & PIDS+=($!)
wait_port "$REP_PORT"
python3 examples/openai_client_demo.py --base-url "http://127.0.0.1:$REP_PORT/v1" \
  --session sdk-demo --turns 2

echo
echo "--- 假上游累计调用次数（record 阶段的 2 次；replay 阶段必须是 0 增量）"
curl -s "http://127.0.0.1:$UP_PORT/__stats"; echo
