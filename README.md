# LLM Mock

**本地监听的 LLM 请求录制 / 回放代理。** 客户端只改一个 `base_url`，业务代码一行不动。

```
OpenAI 兼容客户端 ──▶ http://127.0.0.1:8800/v1 ──▶ llm-mock ──┬─▶ 录制库（JSONL）
   (openai SDK / LangChain / 自研 harness)                     └─▶ 真实 API（仅在允许时）
```

一次付费、之后免费：跑一遍真实 API 把交互录下来，之后判分、调试、回归全部本地回放，**0 token**。

- **四种模式**：`live` / `record` / `replay`（未命中直接报错，绝不外呼）/ `hybrid`（默认）
- **三种回放**：精准（请求哈希）、模糊（**把工具输出从键里洗掉**）、位置（会话 + 轮次，不看内容）
- **未命中可诊断**：逐字段比对，告诉你差在哪、属于"可容忍"还是"致命"、下一步该开哪个开关
- **密钥不落盘**：默认开启 API 脱敏，请求/响应里的密钥在写文件之前就被替换成占位符
- **零依赖**：纯 Python 标准库（3.9+）

---

## 1. 快速开始

```bash
# ① 建缓存：转发真实 API 并把交互录下来
python -m llm_mock --mode record --port 8800 \
  --upstream https://api.deepseek.com --upstream-key-file /path/to/key \
  --store ./records.jsonl

# 客户端：只改 base_url
export OPENAI_BASE_URL=http://127.0.0.1:8800/v1

# ② 0 token 回放（未命中直接报错，不会偷偷调 API）
python -m llm_mock --mode replay --port 8800 --store ./records.jsonl

# ③ 混合：命中回放，未命中转发并录制
python -m llm_mock --mode hybrid --port 8800 --store ./records.jsonl \
  --upstream https://api.deepseek.com --upstream-key-file /path/to/key \
  --replay-latency none
```

`/health` 里始终有 `upstream_calls` 计数 —— 这是"到底有没有真的外呼"的唯一判据。

## 2. 四种模式

| 模式 | 命中录制 | 未命中 | 用途 |
|---|---|---|---|
| `live` | — | 转发，不录制 | 纯透明代理 |
| `record` | 不回放 | 转发 **+ 录制** | 建缓存、跑首次全量 |
| `replay` | 回放 | **HTTP 409 报错**（绝不外呼） | 判分 / 迭代，保证 0 token |
| `hybrid`（默认） | 回放 | 转发 + 录制 | 日常开发：省钱且不漏 |

`replay` 未命中的响应体：

```json
{"error":{"type":"replay_miss","session":"...","turn":3,
          "key":"...","fuzzy_key":"...","miss_id":7,
          "hint":"诊断: GET /admin/why?id=7"}}
```

## 3. 三种回放策略

按 `--replay-order`（默认 `exact,fuzzy,positional`）依次尝试，先命中先用。

| 策略 | 键 | 抗什么漂移 | 风险 |
|---|---|---|---|
| `exact` 精准 | `sha256(canonical(messages + tools + model + temperature + max_tokens))` | 无 | 一个字节变了就 miss |
| `fuzzy` 模糊 | 同上，但**工具输出被替换为占位符** | 工具输出内容、工具调用 id（可选：耗时/时间戳/临时路径） | 键变粗，需防误命中 |
| `positional` 位置 | 会话 id + 轮次序号（**不看内容**） | 全部内容漂移 | 内容无关，可能取错轮次 |

### 3.1 模糊匹配 = 不带工具输出

`--fuzzy-mask-tool-output`（默认开）做两件事：

1. `role=tool` / `role=function` 的消息内容 → 统一替换为 `<TOOL_OUTPUT>` 占位符；
2. 丢弃 `tool_call_id` 与 `tool_calls[].id`（这些由上游生成，每次都不一样）。

再加可选开关 `--fuzzy-normalize-volatile`（默认关）：把耗时（`real 0m1.234s`、`0.31s`）、
ISO 时间戳、`/tmp/pytest-of-*` 临时路径、ANSI 转义也替换成固定占位。

为什么值得这么做——在一份 12,563 条的既有录制上统计：

| 指标 | 数值 |
|---|---|
| 含 `role=tool` 消息的记录 | **12,067 / 12,563 = 96.1%** |
| `tool` 消息总数 | 203,008（平均每条记录 **16.2** 条） |
| messages 总字节 / 其中工具输出 | 0.621 GB / **0.468 GB = 75.4%** |
| 不同的工具输出内容数 / 不同 `tool_call_id` 数 | 11,381 / 12,354 |

即：请求体的四分之三是"环境执行结果"，把它们留在键里，第 N 轮必然 miss。
实测把全部工具输出替换成随机时间戳/耗时/临时路径后，**400/400 条（100%）的模糊键保持不变**。

### 3.2 位置回放的会话身份

优先级：请求头 `X-LLM-Mock-Session` → 从消息里正则抽取 → `--implicit-session` 兜底。
轮次 = 该会话内的第几次请求（从 0 开始），可用 `X-LLM-Mock-Turn` 覆盖，或 `POST /admin/reset` 清零。

⚠ `--positional-verify auto`（默认）是一道安全阀：

* 客户端**显式声明**会话（header / implicit）→ 认定它要重放轨迹，内容不校验；
* 会话是**从消息里猜**出来的（regex）→ 只当弱身份，位置命中前再比对一次模糊键。

要完全内容无关的纯轨迹回放：`--positional-verify none`。

## 4. HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI 兼容入口（支持 `stream=true` 的 SSE） |
| GET | `/v1/models` | 从录制库汇总的模型列表 |
| GET | `/health`、`/admin/stats` | 模式、命中/未命中、**upstream_calls**、存储与脱敏统计 |
| GET | `/admin/sessions?limit=N` | 会话与轮数概览 |
| GET | `/admin/records?session=&limit=&offset=` | 录制条目（默认不含消息体） |
| GET | `/admin/misses?limit=N` | 最近未命中列表（含 `miss_id`） |
| GET | `/admin/why[?id=N][&format=text]` | 未命中诊断（不给 id 取最近一次） |
| POST | `/admin/why` | 对任意请求体做诊断 |
| POST | `/admin/reset` `{"session":"..."}` | 重置轮次计数（省略 session = 全部） |
| POST | `/admin/reload` | 重新加载存储 |
| POST | `/admin/mode` `{"mode":"live"}` | 运行期切换模式 |
| 其他 `/v1/*` | | `live/record/hybrid` 原样转发；`replay` 返回 501（绝不外呼） |

回放响应带诊断头：`X-LLM-Mock-Hit: exact|fuzzy|positional`、`X-LLM-Mock-Key`、
`X-LLM-Mock-Session`、`X-LLM-Mock-Turn`。

## 5. 未命中诊断（为什么没命中）

`replay` 模式**自动**算好报告并在 409 里带上 `miss_id`；`hybrid` 只留档、等你来问
（避免首次录制时每轮都全库扫描）：

```console
$ curl -s 'http://127.0.0.1:8800/admin/why?format=text'
=== llm-mock miss 诊断 ===
模式=replay  会话=django/django@0ab58c120939（来源=regex）  轮次=5  model=deepseek-v4-pro
消息 5 条 / 工具 2 个  回放顺序=exact,fuzzy,positional
模糊参数: mask_tool_output=True normalize_volatile=False positional_verify=fuzzy

--- 三种策略 ---
  [精准] 未命中  候选键 3 个
        原因: 完整请求（含工具输出、tool_call id）与任何录制都不完全一致
  [模糊] 未命中
        原因: 屏蔽工具输出后仍与任何录制不一致（未开噪声归一化）
  [位置] 未命中  状态=rejected-by-verify  该会话 50 轮
        原因: 第 5 轮存在，但被 positional_verify=fuzzy 拒绝（内容差异过大）

--- 最接近的 2 条录制 ---
  #1 相似度(语义)=0.6 原始=0.25  session=django/django@0ab58c120939 turn=5 key=01324fe844ed…
       ~ 可容忍 messages[3].content: 工具输出内容不同
       ~ 可容忍 messages[4].content: 工具输出内容不同
       ! 致命   messages[1].content: 正文内容不同（模糊匹配只屏蔽工具输出，不吸收这个）

--- 结论: content-drift ---
  * 差异落在 system/user 正文或工具参数上，模糊匹配不吸收 → 要么重新录制该轮（hybrid），要么改用位置回放
  * 位置回放本来命中，被内容校验拦下 → 若确实只想复现轨迹，用 --positional-verify none
```

离线也能查（不起服务，直接对存储诊断）：

```bash
python -m llm_mock why --store records.jsonl --request req.json           # 文本报告
python -m llm_mock why --store records.jsonl --request req.json --json    # JSON
python -m llm_mock why --url http://127.0.0.1:8800 --request req.json     # 问正在跑的服务
# 退出码：0 = 某种策略本来能命中；2 = 确实命不中；1 = 用法/IO 错误
```

结论取值：`exact-hit` / `fuzzy-hit` / `tolerable-only`（只差可容忍项却没命中 → 多半是两边参数不一致）
/ `content-drift` / `no-candidate`。相关开关：`--why-on-miss auto|always|never`、`--why-top N`、
`--why-max-scan N`、`--miss-log N`。

## 6. API 脱敏（密钥永不落盘，默认开）

脱敏在**请求进入代理时做一次**，下游的一切（精准/模糊键、JSONL、miss 留档、诊断报告、
`/admin/*` 输出）都是脱敏后的数据；**转发给真实 API 的仍是原始请求体**，客户端拿到的响应也不变。

| 规则 | 覆盖 |
|---|---|
| 按**名字** | 键名命中 `api_key` / `token` / `secret` / `password` / `credential` / `authorization` / `cookie` … 的字符串值（`max_tokens`、`tool_call_id` 等协议字段白名单放行） |
| 按**形状** | `sk-…`、`sk-ant-…`、`ghp_`/`github_pat_`、`glpat-`、`xox…`、`AIza…`、`AKIA/ASIA…`、`hf_…`、`npm_…`、`Bearer …`、Basic base64、JWT、`-----BEGIN … PRIVATE KEY-----`、`scheme://user:pass@host`、`OPENAI_API_KEY=…` 这类大写环境变量赋值 |
| 自定义 | `--scrub-extra-regex 'MYCO-[A-Z0-9]{10}'`（可重复） |

```bash
python -m llm_mock --mode record --scrub-key-file /path/to/scrub.key   # 默认就脱敏；这里给 HMAC key
python -m llm_mock --mode record --scrub-dry-run                       # 只统计命中、不替换：先量影响面
python -m llm_mock --mode record --no-scrub                            # 关闭（录制库保存原文）
```

* 替换文本默认 `[REDACTED:<kind>:<8位hmac>]` —— **同一个密钥每次替换结果相同**，因此不影响
  "录制时算的键"与"回放时算的键"的一致性；`--scrub-mode redact` 则统一成 `[REDACTED:<kind>]`。
* 影响面实测（611.7 MB / 12,563 条录制）：canonical 形式被改变的记录 **945 / 12,563 = 7.52%**、
  命中 286 处；代理会为这些记录额外建立**等价精准键**，抽样自检 **200/200 = 100%** 仍精准命中同一条。
* **诚实边界**：没有规则能覆盖"任意形状的随机串"。没有厂商前缀、又不在敏感键名下的密钥，需要
  `--scrub-extra-regex`，建议先 `--scrub-dry-run`。脱敏也会降低原文保真度（源码里的
  `SECRET_KEY = "…"` 同样会被打码），且**回放时若模型原样吐回过密钥，你看到的是占位符**。

## 7. 存储格式

```json
{"key":"<exact sha256>","ts":1791205658,"model":"deepseek-v4-pro",
 "request":{"messages":[...],"tools":[...],"temperature":0,"max_tokens":4096,"stream":false},
 "response":{"content":"...","tool_calls":[{"id","name","arguments"}],"finish_reason":"tool_calls"},
 "usage":{...},"latency_ms":3842,
 "fuzzy_key":"<sha256>","session":"astropy/astropy@d16bfe05a744","turn":0,"source":"record"}
```

* 前 7 个字段与既有的 `llm_cache.jsonl` 格式同名同义、键算法一致 —— **历史录制文件无需转换**即可
  三种回放全用；缺的 `fuzzy_key`/`session`/`turn` 在加载时就地补算。
* 体检一份录制文件：`python tools/store_report.py --store <file>`

```
key 复算一致 12563/12563 = 100.00%    会话 495 个，100% 记录都落进轨迹
轮次 p50 23 / p90 45 / max 50         加载 7.8 s（开脱敏 32 s），RSS 0.97 GiB
```

## 8. 命令速查

```bash
python -m llm_mock --help
python -m llm_mock --print-config                  # 打印生效配置
python -m pytest tests -q                          # 53 个测试（端到端 + 诊断 + 脱敏）
python -m llm_mock why --store r.jsonl --request req.json   # 未命中诊断
python -m llm_mock.testing --port 8899             # 内置假上游（离线演示/测试用）
bash scripts/demo_offline.sh                       # 离线端到端演示（0 token，含诊断与脱敏）
bash scripts/demo_openai_sdk.sh                    # 真 openai SDK 兼容性演示
python tools/store_report.py --store r.jsonl [--scrub]   # 录制文件体检
python tools/replay_probe.py --store r.jsonl       # 真实录制上的三种回放 + 诊断（上游=死端口）
```

环境变量：所有参数都支持 `LLM_MOCK_` 前缀（`LLM_MOCK_MODE`、`LLM_MOCK_UPSTREAM_KEY`、`LLM_MOCK_STORE`…）。

## 9. 提交到仓库前请注意

* **不要把录制文件提交进仓库。** `*.jsonl` 里是 LLM 交互原文：即使默认开了脱敏，
  提示词与工具输出本身也可能带业务数据、内网信息或未被规则覆盖的凭据。`.gitignore` 已经
  默认忽略 `*.jsonl`、`*.key`、`.env*`。
* 代理自身不会把密钥写进录制：客户端/上游的 `Authorization` 等**请求头一律不落盘**，
  `--print-config` 会把 `--upstream-key` 打码，启动日志只打印 store 路径与上游地址。
* 真要用真实 API 跑录制，把 key 放在**文件或环境变量**里（`--upstream-key-file` /
  `LLM_MOCK_UPSTREAM_KEY`），不要写在命令行（会进 shell history 与进程列表）。
* 脱敏是**尽力而为**的兜底，不是合规保证：没有厂商前缀、又不在敏感键名下的随机串它抓不到，
  请按需补 `--scrub-extra-regex` 并先用 `--scrub-dry-run` 量一遍。

## 10. 已知边界（诚实清单）

* **流式**：录制的流会被拼回完整消息再落盘；回放时反向合成 SSE（分片边界与原始流不同，
  但重组后的消息一致）。`stream_options.include_usage` 的 usage 会保留。
* **模糊键无法吸收的漂移**：内嵌在 user 消息里的工具输出、`ls`/`find` 的遍历顺序、截断长度变化
  —— 前者建议把首轮请求做成纯确定的；后两者用 `--fuzzy-normalize-volatile` 或位置回放。
* **位置回放不看内容**：会话身份猜错就会回放错轮次；因此默认 `auto` 校验。
* **脱敏的代价**：见「API 脱敏」一节 —— 原文保真度下降、回放里密钥显示为占位符、大存储加载变慢（`--no-scrub` 可关）。
* **录制库只追加**：同 key 后写覆盖（重载后 last-wins），但不自动压缩/回收历史行。
* **仅 OpenAI Chat Completions**：Anthropic / Gemini 协议未适配。

## 11. 布局

```
llm_mock/
  canon.py       键计算 + 工具输出屏蔽 + 噪声归一化
  store.py       JSONL 存储 + 精准/模糊/位置三份索引
  matcher.py     策略顺序与位置命中校验
  session.py     会话身份解析 + 轮次计数
  openai_api.py  Chat Completions 编解码 + SSE 拼装/合成
  upstream.py    上游转发（http.client，强制 identity 编码）
  scrub.py       密钥脱敏
  diagnose.py    未命中诊断
  proxy.py       代理服务 + HTTP 层 + /health /admin/*
  why_cli.py     `llm-mock why` 命令行
  testing.py     内置假上游
  cli.py         命令行入口
tests/           53 个测试
tools/           录制文件体检、真实回放探针
scripts/         两个端到端演示
```
