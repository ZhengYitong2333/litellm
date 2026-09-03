# Sophnet-Azure 测试与修复闭环

本文档描述在 `deploy/sophnet-azure` 本地 Proxy 上发现问题、修复 LiteLLM、验证并提交的**标准流程**。代理真实路径测试统一使用 `test_corner_cases.py`。

## 1. 检查问题

按顺序缩小范围：

| 步骤 | 命令 / 动作 | 目的 |
|------|-------------|------|
| Proxy 存活 | `curl -sf http://localhost:4000/health/liveliness` | 排除容器未就绪 |
| 容器状态 | `docker compose ps` / `docker compose logs litellm --tail 80` | Prisma、路由、上游 4xx/5xx |
| 资源 | `docker stats --no-stream` | CPU 打满会导致 CC/Codex「卡住」 |
| 模型列表 | `curl -s -H "Authorization: Bearer $KEY" localhost:4000/v1/models` | 别名是否与 `config.yaml` 一致 |
| 直连上游 | 用 Sophnet/Azure/DeepSeek 官方 curl（绕过 Proxy） | 区分「网关额度/429」与「LiteLLM 转换 bug」 |
| bind-mount | `docker compose exec litellm python3 -c "import litellm; print(litellm.__file__)"` | 必须为 `/app/litellm/__init__.py` |

常见根因分类：

- **LiteLLM 参数转换**：thinking 历史、tools 名、`output_config` → `response_format`、Responses vs chat bridge。
- **Provider 网关差异**：Sophnet Anthropic 非官方端点、opaque `<nil>` 400、`web_search_*` 校验。
- **上游额度**：402/429；router 多 key 随机命中欠费 key。
- **本地 Docker**：`.env` 未 `force-recreate`、未 `restart-proxy.sh`、端口占用。

## 2. 更新问题（改代码）

- 转换逻辑优先改 `litellm/llms/`、`litellm/responses/` 对应 provider 模块。
- 避免在代码里硬编码模型能力；能用 `model_prices_and_context_window.json` / `get_model_info` 则用。
- 改 `litellm/` 后部署：

```bash
cd deploy/sophnet-azure
./restart-proxy.sh
```

- 改 `config.yaml` 或 `.env`：配置用 restart；**密钥变更**需 `docker compose up -d --force-recreate litellm`。

## 3. 测试层次

### 3.1 单元测试（优先）

覆盖参数转换与分支，不依赖真实上游：

```bash
cd /path/to/litellm
uv run pytest tests/test_litellm/llms/anthropic/test_anthropic_common_utils.py -v
uv run pytest tests/test_litellm/llms/anthropic/experimental_pass_through/adapters/test_handler_output_config_passthrough.py -v
uv run pytest tests/test_litellm/responses/litellm_completion_transformation/test_handler_client_metadata.py -v
uv run pytest tests/test_litellm/llms/anthropic/experimental_pass_through/responses_adapters/test_responses_adapters_transformation.py -v
```

提交前对改动文件跑 `uv run black .`（或至少改动路径）。

### 3.2 代理 Corner-case（真实路径）

```bash
cd deploy/sophnet-azure
python3 test_corner_cases.py              # --quick 默认
python3 test_corner_cases.py --matrix     # CC/Codex x 全模型矩阵（并行）
python3 test_corner_cases.py --full       # full + matrix（streaming、reasoning、回归）
python3 test_corner_cases.py --stress     # 小并发，可能 429，勿作 CI 默认
python3 test_corner_cases.py --list       # 查看矩阵
# 并行 worker 数（默认 4）
LITELLM_TEST_WORKERS=6 python3 test_corner_cases.py --matrix --full
```

约定：

- **stdout 仅 JSON**；摘要在 stderr。
- 仅对 `429/502/503/504` 有限重试；400/参数错误立即失败。
- 新 bug 必须映射到**一个** case；若与已有 case 同路径同模型，只加强断言，不新增重复 case。

### 3.3 手工冒烟（可选）

```bash
# Sophnet GPT-5.5 原生 Responses（曾卡死在 chat bridge）
curl -s http://localhost:4000/v1/responses \
  -H "Authorization: Bearer sk-litellm-sophnet-azure-local" \
  -H "Content-Type: application/json" \
  -d '{"model":"sophnet-gpt-5.5","input":"OK","max_output_tokens":16}'

# Claude Messages + web_search 工具名
curl -s http://localhost:4000/v1/messages \
  -H "Authorization: Bearer sk-litellm-sophnet-azure-local" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"sophnet-claude-opus-4-7","max_tokens":32,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search_20250305","max_uses":1}]}'
```

## 4. 更新测试矩阵

在 `test_corner_cases.py` 的 `build_cases()` 中增加或扩展 `Case`：

| 字段 | 说明 |
|------|------|
| `id` | 稳定标识，如 `messages.claude.web_search_tool` |
| `tier` | `quick` / `full` / `matrix` / `stress` |
| `why` | 绑定的历史故障一句话 |
| `expect` | `ok`（默认）或 `fail`（期望 4xx） |

维护规则：

1. 先 `--list` 查是否已有等价 case。
2. `quick` 保持可在几分钟内跑完；全模型 chat 连通各一条即可。
3. CC/Codex x 全模型矩阵放 `matrix`；`--full` 自动包含 matrix。
4. streaming / burst 放 `full` 或 `stress`。
5. 上游 429 或 Sophnet opaque `<nil>` 400 计为 skip，不算 LiteLLM 失败。

## 5. Corner-case 矩阵（当前）

### quick tier

- `chat.basic.*` — Key、路由、模型别名、基础连通
- `responses.gpt55.native` — GPT-5.5 必须走原生 `/v1/responses`
- `messages.claude.basic` — Anthropic 原生 messages
- `messages.claude.thinking` — Extended thinking
- `messages.claude.web_search_tool` — `web_search_*` name 归一化
- `messages.claude.codex_tools` — shell/namespace/computer 过滤（上游 opaque 400 → skip）
- `messages.claude.thinking_history` — 无效 thinking 签名剥离
- `messages.claude.redacted_history` — redacted_thinking 剥离
- `adapter.glm.output_config_json` — `output_config.format` 适配
- `auth.bad_key` — 错误 master key → 401

### matrix tier（CC / Codex x 全模型，并行执行）

- `messages.basic.*` — CC `/v1/messages` x 每个 `CHAT_MODELS` 别名（Claude 原生，其余 adapter）
- `responses.basic.*` — Codex `/v1/responses` x 每个 `CHAT_MODELS` 别名（GPT 原生/桥接，其余 chat bridge）
- `messages.adapter.orphan_tool_call.{sophnet-gpt-5.5,sophnet-glm-5.2,deepseek-v4-pro}` — assistant `tool_use` 缺 `tool_result` 时插入占位（前两者走 adapter→chat，deepseek 走原生 Anthropic passthrough）
- `messages.adapter.empty_tool_result.{sophnet-gpt-5.5,sophnet-glm-5.2,deepseek-v4-pro}` — `tool_result` 的 `content: []` 仍发出 tool 消息
- `responses.bridge.interleaved_tool_result` — `function_call` 与 `function_call_output` 被 user 打断时重排（sophnet-gpt-5.5）
- `responses.bridge.orphan_tool_call` — 缺 `function_call_output` 时插入占位（sophnet-gpt-5.5）

### full tier

- `stream.chat.ttfb.*` — Chat 流式 TTFB
- `messages.claude.stream` — Messages 流式
- `adapter.gpt.thinking_tools` — GPT reasoning+tools 不挂起
- `adapter.glm.temperature_zero` — GLM temperature=0 应失败
- `messages.claude.output_format_json` — `output_format` JSON schema
- `messages.claude.tool_history` — tool_result 多轮
- `chat.reasoning.*` — Chat 路径 reasoning

### stress tier

- `stress.burst.*` — 小并发

## 6. 提交

```bash
git status   # 仅相关 litellm/、tests/、deploy/sophnet-azure/
uv run pytest <目标单测文件> -v
python3 deploy/sophnet-azure/test_corner_cases.py --quick
uv run black <改动的.py 文件>
git add ...
git commit -m "fix(anthropic): ..."
git fetch && git rebase origin/<branch>
git push
```

不要提交 `.env`、本地密钥或一次性复盘文档。

## 7. 相关文件

| 文件 | 作用 |
|------|------|
| `test_corner_cases.py` | 主测试入口 |
| `config.yaml` | 模型别名与上游 |
| `restart-proxy.sh` | 改 `litellm/` 后重启 |
| `VSCODE_CC_CODEX.md` | CC/Codex 客户端配置 |
| `~/.claude/skills/litellm-proxy-testing/SKILL.md` | Agent 技能（流程摘要） |

## 8. 单元 + 集成双层 corner case 测试

测试覆盖 `tests/test_litellm/llms/bedrock/test_modellist_corner_cases.py`(pytest 单元,不需要 live proxy)与 `deploy/sophnet-azure/test_corner_cases.py`(live proxy 集成)。两者职责分明:

### 单元层(`test_modellist_corner_cases.py`)

直接调用 `AmazonConverseConfig._transform_request_helper`,验证 Bedrock Converse 转换层在每个 modellist 条目上行为正确。CI 里不需要 proxy。

- `TestEffortBetaLeakRegression`: 9 models × 2 cases(output_config + header)— 守住 `effort-2025-11-24` 不再 leak 进 `anthropic_beta`
- `TestJsonDrivenBetaFilter`: 9 个 JSON null 的 betas 都被过滤 + 1 个 supported pass-through + empty headers
- `TestBedrockConverseStability`: 9 models × 3(basic + tools + cross-region prefix)

跑法:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_litellm/llms/bedrock/test_modellist_corner_cases.py -v
```

或全 bedrock 套件:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_litellm/llms/bedrock/ -n 4
```

### 集成层(`test_corner_cases.py`)

实际打 proxy 端点,验证 9 个 modellist 模型 × 3 endpoint(`/v1/chat/completions`、`/v1/messages`、`/v1/responses`)的稳定型 + corner case:

- **matrix tier(120 case 起步,2026-06 扩到 183 case)**: 每个 model × 每个 endpoint × 各类 case
  - `matrix.chat.stream.{model}` / `matrix.messages.stream.{model}` / `matrix.responses.stream.{model}` — 流式稳定
  - `matrix.chat.tools.{model}` / `matrix.messages.tools.{model}` / `matrix.responses.tools.{model}` — 工具调用稳定
  - `matrix.chat.stream_tools.{model}` / `matrix.messages.stream_tools.{model}` / `matrix.responses.stream_tools.{model}` — 流式 + 工具(CC/Codex 的核心路径)
  - `matrix.messages.tool_choice_any.{model}` — 强制工具调用
  - `matrix.chat.anthropic_beta_header.{model}` — beta header 过滤 end-to-end
  - `matrix.messages.output_config_effort.{model}` — `output_config.effort` 不 leak
  - `matrix.chat.long_context.{model}` / `matrix.chat.long_context_32k.{model}` — 长上下文
  - `matrix.chat.max_tokens_one.{model}` / `matrix.messages.system.{model}` — 边界 case
- **full tier**: 旧有 streaming/reasoning/structured output 等
- **stress tier**: 小并发

跑法:

```bash
# 冒烟(默认 ~30s)
cd deploy/sophnet-azure
python3 test_corner_cases.py

# 全 modellist 矩阵(~2 min,165 case)
python3 test_corner_cases.py --matrix

# 矩阵 + 旧 full tier(~3 min,183 case)
python3 test_corner_cases.py --full
```

### 何时跑哪一层

| 改了什么 | 跑哪个 |
|----------|--------|
| `litellm/llms/bedrock/chat/converse_transformation.py` 等转换层 | pytest 单元(`test_modellist_corner_cases.py` + 整套 bedrock pytest) |
| `litellm/anthropic_beta_headers_config.json` / manager | pytest 单元 + 集成 matrix(`--matrix`) |
| Proxy 路由 / 配置 / adapter | 集成 matrix(`--full`) |
| 端到端新功能 | 集成 matrix + pytest 单元 |

修 bedrock transformation 的 commit 应当两个都跑:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_litellm/llms/bedrock/test_modellist_corner_cases.py -v
cd deploy/sophnet-azure && python3 test_corner_cases.py --full
```
