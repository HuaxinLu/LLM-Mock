#!/usr/bin/env bash
# 离线端到端演示（0 token、0 外网）：
#   假上游 + llm-mock(record) -> 录制；llm-mock(replay) -> 三种回放；llm-mock(hybrid) -> 命中回放/未命中转发
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

UP_PORT=18899
REC_PORT=18800
REP_PORT=18801
HYB_PORT=18802
STORE=/tmp/llm-mock-demo.jsonl
LOGDIR=/tmp/llm-mock-demo
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"; rm -f "$STORE"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null
  done
  sleep 0.3
}
trap cleanup EXIT

wait_port() {  # $1=port
  for _ in $(seq 1 50); do
    curl -s -o /dev/null "http://127.0.0.1:$1/health" && return 0
    sleep 0.2
  done
  echo "端口 $1 未就绪"; return 1
}

echo "### 0. 起假上游（模拟真实 API，会统计被调用次数）"
python3 -m llm_mock.testing --port "$UP_PORT" --volatile > "$LOGDIR/upstream.log" 2>&1 &
PIDS+=($!)
wait_port "$UP_PORT"

echo
echo "### 1. record 模式：两次真实调用并落盘"
python3 -m llm_mock --mode record --port "$REC_PORT" \
  --upstream "http://127.0.0.1:$UP_PORT" --store "$STORE" \
  --replay-latency none > "$LOGDIR/record.log" 2>&1 &
PIDS+=($!)
wait_port "$REC_PORT"

TURN_HEADER=(-H 'Content-Type: application/json')

req() {  # $1=port $2=body $3..=extra headers
  local port=$1 body=$2; shift 2
  curl -s -D "$LOGDIR/headers.txt" -o "$LOGDIR/body.json" \
    -w '%{http_code}' -X POST "http://127.0.0.1:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' "$@" -d "$body"
}

BODY1='{"model":"fake-model","messages":[{"role":"system","content":"sys"},{"role":"user","content":"仓库 astropy/astropy @ d16bfe05a744\nissue: fix separability"}],"temperature":0,"max_tokens":4096}'
BODY2='{"model":"fake-model","messages":[{"role":"user","content":"仓库 astropy/astropy @ d16bfe05a744\nissue: fix separability"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_a","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]},{"role":"tool","tool_call_id":"call_a","content":"Thu Oct 5 12:00:01 UTC 2026\nreal 0m1.234s\nfile1"}],"temperature":0,"max_tokens":4096}'

code=$(req "$REC_PORT" "$BODY1"); echo "turn0 -> HTTP $code  $(python3 -c "import json;print(json.load(open('$LOGDIR/body.json'))['choices'][0]['message']['content'])")"
code=$(req "$REC_PORT" "$BODY2"); echo "turn1 -> HTTP $code  $(python3 -c "import json;print(json.load(open('$LOGDIR/body.json'))['choices'][0]['message']['content'])")"
echo "存储行数: $(wc -l < "$STORE")"

echo
echo "### 2. replay 模式：同一份存储，0 次真实调用"
python3 -m llm_mock --mode replay --port "$REP_PORT" \
  --upstream "http://127.0.0.1:1" --store "$STORE" \
  --replay-latency none > "$LOGDIR/replay.log" 2>&1 &
PIDS+=($!)
wait_port "$REP_PORT"

show_hit() { grep -i '^x-llm-mock-hit' "$LOGDIR/headers.txt" | tr -d '\r' | awk '{print $2}'; }

code=$(req "$REP_PORT" "$BODY1"); echo "exact      -> HTTP $code  hit=$(show_hit)"
BODY2_DRIFT='{"model":"fake-model","messages":[{"role":"user","content":"仓库 astropy/astropy @ d16bfe05a744\nissue: fix separability"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_zzz","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]},{"role":"tool","tool_call_id":"call_zzz","content":"Fri Oct 6 03:44:59 UTC 2026\nreal 0m9.876s\npytest-7\nfile2\nfile3"}],"temperature":0,"max_tokens":4096}'
code=$(req "$REP_PORT" "$BODY2_DRIFT" -H 'X-LLM-Mock-Turn: 1'); echo "fuzzy      -> HTTP $code  hit=$(show_hit)   (工具输出/时间戳/工具调用 id 全变)"
BODY3='{"model":"fake-model","messages":[{"role":"user","content":"与录制完全无关的请求"}],"temperature":0,"max_tokens":4096}'
code=$(req "$REP_PORT" "$BODY3" -H 'X-LLM-Mock-Session: astropy/astropy@d16bfe05a744' -H 'X-LLM-Mock-Turn: 0'); echo "positional -> HTTP $code  hit=$(show_hit)   (内容无关，只看会话+轮次)"
BODY4='{"model":"fake-model","messages":[{"role":"user","content":"没录过的任务"}],"temperature":0,"max_tokens":4096}'
code=$(req "$REP_PORT" "$BODY4" -H 'X-LLM-Mock-Session: not-recorded')
MISS_ID=$(python3 -c "import json;print(json.load(open('$LOGDIR/body.json'))['error'].get('miss_id'))")
echo "miss       -> HTTP $code  replay_miss  (miss_id=$MISS_ID)"

echo
echo "--- miss 诊断：GET /admin/why?id=$MISS_ID&format=text"
curl -s "http://127.0.0.1:$REP_PORT/admin/why?id=$MISS_ID&format=text"

echo
echo "--- replay 模式 /health（注意 upstream_calls）"
curl -s "http://127.0.0.1:$REP_PORT/health" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print(json.dumps({k:d[k] for k in ("mode","chat_requests","hits","misses","replayed","forwarded","upstream_calls")}, ensure_ascii=False, indent=2))'

echo
echo "### 3. hybrid 模式：命中回放，未命中转发真实 API 并录制"
python3 -m llm_mock --mode hybrid --port "$HYB_PORT" \
  --upstream "http://127.0.0.1:$UP_PORT" --store "$STORE" \
  --replay-latency none > "$LOGDIR/hybrid.log" 2>&1 &
PIDS+=($!)
wait_port "$HYB_PORT"

code=$(req "$HYB_PORT" "$BODY2_DRIFT" -H 'X-LLM-Mock-Turn: 1'); echo "已录制的轮次 -> HTTP $code  hit=$(show_hit)"
BODY_NEW='{"model":"fake-model","messages":[{"role":"user","content":"仓库 django/django @ abcdef123456\nissue: 全新任务"}],"temperature":0,"max_tokens":4096}'
code=$(req "$HYB_PORT" "$BODY_NEW"); echo "全新请求     -> HTTP $code  hit=$(show_hit)  (空 = 已转发真实 API)  $(python3 -c "import json;print(json.load(open('$LOGDIR/body.json'))['choices'][0]['message']['content'])")"
code=$(req "$HYB_PORT" "$BODY_NEW"); echo "刚录完重放   -> HTTP $code  hit=$(show_hit)"
echo "存储行数: $(wc -l < "$STORE")"

echo
echo "--- 假上游累计被调用次数（应等于真实转发次数）"
curl -s "http://127.0.0.1:$UP_PORT/__stats"; echo
echo "--- hybrid /health"
curl -s "http://127.0.0.1:$HYB_PORT/health" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print(json.dumps({k:d[k] for k in ("mode","chat_requests","hits","misses","replayed","recorded","forwarded","upstream_calls")}, ensure_ascii=False, indent=2))'

echo
echo "### 4. API 脱敏：含密钥的请求不会把密钥写进存储"
# 假密钥：故意用"环境变量赋值 + 高熵值"而不是任何厂商的密钥前缀，
# 免得被 GitHub 推送保护 / secret scanner 误判成真密钥。
FAKE_SECRET='not-a-real-key-just-a-demo'
BODY_KEY='{"model":"fake-model","messages":[{"role":"user","content":"仓库 astropy/astropy @ d16bfe05a744\nOPENAI_API_KEY='"$FAKE_SECRET"'"}],"temperature":0,"max_tokens":4096}'
code=$(req "$REC_PORT" "$BODY_KEY")
echo "含密钥请求 -> HTTP $code   (上游收到的是原文，客户端拿到的也是原文)"
count=$(grep -c "$FAKE_SECRET" "$STORE" || true)
echo "存储里假密钥出现次数: $count   (应为 0)"
echo "存储里的脱敏占位符（按 kind 统计）:"
grep -o 'REDACTED:[a-z_0-9]*' "$STORE" | sed 's/REDACTED://' | sort | uniq -c | sed 's/^/   /'
curl -s "http://127.0.0.1:$REC_PORT/health" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("   /health.scrub =", json.dumps(d["scrub"], ensure_ascii=False))'

echo
echo "### 演示结束（进程已清理）"
