# VS Code 配置 Claude Code（CC）与 Codex 教程

本教程基于当前分支的 **`deploy/sophnet-azure`** 本地 LiteLLM Proxy，在 VS Code 里把 **Claude Code** 和 **OpenAI Codex** 接到 Sophnet / Azure / DeepSeek 模型上。

- **Docker 部署 Proxy**：见 [§1 Docker 部署](#1-docker-部署-litellm-proxy)
- **CC / Codex 客户端配置**：见 [§3](#3-claude-codecc配置) / [§4](#4-openai-codex-配置)

```
VS Code 扩展
    │
    ├─ Claude Code  ──► POST /v1/messages  ──► LiteLLM :4000 ──► Sophnet / Azure
    │
    └─ Codex        ──► POST /v1/responses ──► LiteLLM :4000 ──► DeepSeek / Azure / Sophnet
```

---

## 0. 前置条件

| 项目 | 说明 |
|------|------|
| Docker Desktop / Docker Engine | 已安装并运行（macOS / Linux / WSL2） |
| Docker Compose v2 | `docker compose` 命令可用 |
| VS Code | 已安装扩展 **Claude Code**（Anthropic）和 **Codex**（`openai.chatgpt`） |
| 仓库 | 已 clone 本 LiteLLM 分支，含 `deploy/sophnet-azure/` |

---

## 1. Docker 部署 LiteLLM Proxy

CC / Codex 运行在宿主机（VS Code），LiteLLM 跑在 Docker 里，通过 **`localhost:4000`** 对外暴露。

### 1.1 服务架构

```
┌─────────────────────────────────────────────────────────┐
│  Docker Compose (deploy/sophnet-azure/)                 │
│                                                         │
│  ┌──────────────┐      ┌─────────────────────────────┐  │
│  │  db          │      │  litellm                    │  │
│  │  postgres:16 │◄─────│  litellm-sophnet-azure:local│  │
│  │  :5432       │      │  :4000 ──► 宿主机 :4000     │  │
│  └──────────────┘      │                             │  │
│                        │  挂载:                      │  │
│                        │  · config.yaml              │  │
│                        │  · ../../litellm/ (只读)    │  │
│                        └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         ▲                           │
         │  ANTHROPIC_BASE_URL       │  Sophnet / Azure / DeepSeek API
         │  openai_base_url          ▼
   VS Code (CC / Codex)         上游模型服务
```

| 容器 | 镜像 | 端口 | 作用 |
|------|------|------|------|
| `litellm_sophnet_db` | `postgres:16` | 内部 5432 | Proxy 元数据 / 用量 DB |
| `sophnet-azure-litellm-1` | `litellm-sophnet-azure:local` | **4000→4000** | LiteLLM 网关 |

基础镜像：`ghcr.io/berriai/litellm:main-stable`（见 `Dockerfile.patch`）。本地 `litellm/` 源码通过 **bind-mount** 覆盖镜像内 pip 包，便于开发调试。

### 1.2 目录与关键文件

```
deploy/sophnet-azure/
├── docker-compose.yml    # 编排 db + litellm
├── Dockerfile.patch      # 基于官方镜像，设置 PYTHONPATH
├── config.yaml           # 模型路由（Sophnet / Azure / DeepSeek 别名）
├── .env.example          # 环境变量模板
├── .env                  # 实际密钥（勿提交 git）
├── restart-proxy.sh      # 改 litellm/ 代码后热重启
├── test_corner_cases.py  # 代理 corner-case 测试（--quick / --full / --stress）
├── TESTING_WORKFLOW.md   # 检查→修复→测试→部署→提交闭环
└── VSCODE_CC_CODEX.md    # 本文档
```

### 1.3 首次部署（逐步）

**Step 1 — 进入目录并准备 `.env`**

```bash
cd deploy/sophnet-azure
cp .env.example .env
```

编辑 `.env`，填入真实密钥：

```bash
LITELLM_MASTER_KEY=sk-litellm-sophnet-azure-local   # CC/Codex 用这个当 Bearer
LITELLM_SALT_KEY=litellm-salt-key-local-dev
POSTGRES_PASSWORD=litellm123

SOPHNET_API_KEY_1=<你的 Sophnet Key 1>
SOPHNET_API_KEY_2=<你的 Sophnet Key 2>
AZURE_API_KEY=<你的 Azure Key>
DEEPSEEK_API_KEY=<你的 DeepSeek Key>   # Codex 推荐 deepseek-v4-* 直连

UI_USERNAME=admin
UI_PASSWORD=admin
```

**Step 2 — 检查 `config.yaml`**

模型别名与上游地址已预置，一般无需改。增删模型或换 Key 映射时编辑此文件。

**Step 3 — 构建并启动**

```bash
docker compose up --build -d
```

首次启动会拉镜像、建 PostgreSQL 卷、跑 Prisma 迁移，**约 1–2 分钟** 才就绪（healthcheck `start_period: 90s`）。

**Step 4 — 等待健康**

```bash
# 轮询直到返回 200
until curl -sf http://localhost:4000/health/liveliness; do sleep 3; done
echo "Proxy ready"
```

或查看容器状态：

```bash
docker compose ps
# litellm 应显示 healthy
```

**Step 5 — 验证 API**

```bash
# 模型列表
curl -s -H "Authorization: Bearer sk-litellm-sophnet-azure-local" \
  http://localhost:4000/v1/models | python3 -m json.tool

# CC 通路 (/v1/messages)
curl -s http://localhost:4000/v1/messages \
  -H "Authorization: Bearer sk-litellm-sophnet-azure-local" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"sophnet-claude-opus-4-7","max_tokens":16,"messages":[{"role":"user","content":"OK"}]}'

# Codex 通路 (/v1/responses)
curl -s http://localhost:4000/v1/responses \
  -H "Authorization: Bearer sk-litellm-sophnet-azure-local" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","input":"OK","max_output_tokens":16}'
```

**Step 6 — 确认 bind-mount 补丁生效**

开发分支会把宿主机 `litellm/` 挂进容器，需确认加载的是挂载代码而非镜像内置包：

```bash
docker compose exec litellm python3 -c "import litellm; print(litellm.__file__)"
# 期望输出: /app/litellm/__init__.py
```

**Step 7 —（可选）打开 Admin UI**

浏览器访问 http://localhost:4000/ui ，使用 `.env` 中的 `UI_USERNAME` / `UI_PASSWORD` 登录，可查看模型、用量与日志。

### 1.4 日常运维

| 场景 | 命令 | 说明 |
|------|------|------|
| 改 `litellm/` Python 代码 | `./restart-proxy.sh` | 重启容器并清 cooldown 缓存；**无需 rebuild** |
| 改 `config.yaml` | `./restart-proxy.sh` | 配置只读挂载，重启即生效 |
| 改 `.env` 密钥 | `docker compose up -d --force-recreate litellm` | `docker compose restart` **不会**重载 env |
| 改 `Dockerfile.patch` / 基础镜像 | `docker compose up --build -d` | 需重新 build |
| 查看日志 | `docker compose logs -f litellm` | 排查 402 / model not found |
| 查看状态 | `docker compose ps` | 确认 healthy |
| 停止 | `docker compose down` | 保留 DB 卷 |
| 停止并删数据 | `docker compose down -v` | **会清空 PostgreSQL 数据** |

`restart-proxy.sh` 等价于：

```bash
docker compose restart litellm
# 等待 /health/liveliness 就绪 + 校验 bind-mount
```

### 1.5 修改配置示例

**更换 Sophnet Key（`.env`）**

```bash
vim .env   # 修改 SOPHNET_API_KEY_*
docker compose up -d --force-recreate litellm
```

**临时去掉欠费 Key 的路由（`config.yaml`）**

删除所有 `api_key: os.environ/SOPHNET_API_KEY_1` 的 deployment 块，然后：

```bash
./restart-proxy.sh
```

**新增模型别名**

在 `config.yaml` 的 `model_list` 追加条目，再 `./restart-proxy.sh`。CC / Codex 里使用的名字必须与 `model_name` 一致。

### 1.6 部署到远程服务器

若 Proxy 跑在另一台机器（如 `192.168.1.100`），后续 CC / Codex 配置里把所有 `http://localhost:4000` 换成：

```
http://192.168.1.100:4000
```

并确保：

1. `docker-compose.yml` 中 `ports: "4000:4000"` 已映射
2. 防火墙放行 4000 端口
3. 远程机器的 `LITELLM_MASTER_KEY` 与 VS Code 里填的 Bearer 一致

### 1.7 Docker 部署常见问题

| 现象 | 处理 |
|------|------|
| `litellm` 一直 `starting` | 等满 90s；`docker compose logs litellm --tail 50` 看 Prisma/DB 错误 |
| `connection refused` on :4000 | `docker compose ps` 确认容器在跑；端口未被占用 |
| 改代码不生效 | 确认 `litellm.__file__` 为 `/app/litellm/...`；执行 `./restart-proxy.sh` |
| 改 `.env` 不生效 | 用 `--force-recreate`，不要只用 `restart` |
| UI 登录失败 | 检查 `.env` 的 `UI_USERNAME`/`UI_PASSWORD` 后 force-recreate |

---

## 2. 可用模型别名

Proxy 在 `config.yaml` 中注册的 **`model_name`**（客户端填写的名字必须与此完全一致）。

### 2.1 Claude Code（`/v1/messages`）

| 别名 | 上游 | 推荐 | 说明 |
|------|------|------|------|
| `sophnet-claude-opus-4-7` | Sophnet Anthropic | ⭐ **Sonnet 默认** | 原生 Anthropic 通道 |
| `azure-gpt-5.5` | Azure OpenAI | ⭐ **Opus 推荐** | 规划 / 复杂推理 |
| `deepseek-v4-flash` | DeepSeek 官方 | ⭐ **Haiku 推荐** | 快、便宜，thinking |
| `deepseek-v4-pro` | DeepSeek 官方 | 高质量 | 1M 上下文 |
| `sophnet-glm-5.1` | Sophnet OpenAI 兼容 | 轻量 | 经 LiteLLM 适配器 |
| `sophnet-gpt-5.5` | Sophnet OpenAI 兼容 | 通用 | 可能 402 |
| `sophnet-deepseekv4-pro` | Sophnet OpenAI 兼容 | 备选 | Sophnet 版 DeepSeek |
| `sophnet-deepseekv4-flash` | Sophnet OpenAI 兼容 | 备选 | Sophnet 版轻量 |
| `azure-gpt-5.5` | Azure OpenAI | 稳定 | 同 Opus，GPT 系列 |
| `azure-gpt-5.4` | Azure OpenAI | 经济 | GPT 系列 |

### 2.2 Codex（`/v1/responses`）

| 别名 | 上游 | 推荐 | 说明 |
|------|------|------|------|
| `deepseek-v4-flash` | DeepSeek 官方 | ⭐ **默认推荐** | 快、便宜，已验证 Codex 通路 |
| `deepseek-v4-pro` | DeepSeek 官方 | 高质量 | 1M 上下文，thinking 默认开启 |
| `azure-gpt-5.4` | Azure OpenAI | 稳定 | GPT 系列，无 thinking 干扰 |
| `azure-gpt-5.5` | Azure OpenAI | 稳定 | 同上 |
| `sophnet-gpt-5.5` | Sophnet | 备选 | 可能因 KEY_1 欠费随机 402 |
| `sophnet-deepseekv4-pro` | Sophnet | 备选 | Sophnet 版 DeepSeek |
| `sophnet-deepseekv4-flash` | Sophnet | 备选 | Sophnet 版 DeepSeek 轻量 |

查询当前可用列表：

```bash
curl -s -H "Authorization: Bearer sk-litellm-sophnet-azure-local" \
  http://localhost:4000/v1/models | python3 -m json.tool
```

> **注意**：Sophnet `SOPHNET_API_KEY_1` 欠费时，带 `sophnet-` 前缀的模型可能随机 402；DeepSeek 直连与 Azure 不受影响。

---

## 3. Claude Code（CC）配置

Claude Code 走 **Anthropic Messages API**（`/v1/messages`）。LiteLLM 会把非 Anthropic 模型适配到该接口，因此 CC 可以使用上表全部别名。

**CC 模型菜单**（`/model` 选单固定 4 项）：

| 档位 | LiteLLM 别名 | 配置方式 |
|------|--------------|----------|
| Sonnet | `sophnet-claude-opus-4-7` | `ANTHROPIC_DEFAULT_SONNET_MODEL` |
| Opus | `azure-gpt-5.5` | `ANTHROPIC_DEFAULT_OPUS_MODEL` |
| Haiku | `deepseek-v4-flash` | `ANTHROPIC_DEFAULT_HAIKU_MODEL` |
| **GLM 5.1** | `sophnet-glm-5.1` | `ANTHROPIC_CUSTOM_MODEL_OPTION`（第 4 项） |

> Claude Code 的 `/model` 选单只有 **Sonnet / Opus / Haiku + 1 个自定义项** 共 4 个选项。GLM 5.1 必须通过 `ANTHROPIC_CUSTOM_MODEL_OPTION` 添加；`availableModels` 填具体 model id 会因去重导致选单异常。

**三档 + GLM 默认映射：**

| CC 档位 | LiteLLM 别名 | 用途 |
|---------|--------------|------|
| Sonnet | `sophnet-claude-opus-4-7` | 日常编码主模型 |
| Opus | `azure-gpt-5.5` | 规划、复杂任务 |
| Haiku | `deepseek-v4-flash` | 轻量、快速 |
| GLM 5.1 | `sophnet-glm-5.1` | 轻量中文 / 低成本 |

### 3.1 推荐：项目级 + 用户级配置（macOS）

**① 项目内** `deploy/sophnet-azure/.vscode/settings.json`（或你打开的工作区根目录）：

```json
{
  "claudeCode.disableLoginPrompt": true,
  "claudeCode.selectedModel": "sophnet-claude-opus-4-7",
  "claudeCode.environmentVariables": [
    { "name": "LITELLM_PROXY_URL", "value": "http://localhost:4000" },
    { "name": "LITELLM_PROXY_API_KEY", "value": "sk-litellm-sophnet-azure-local" },
    { "name": "ANTHROPIC_BASE_URL", "value": "http://localhost:4000" },
    { "name": "ANTHROPIC_AUTH_TOKEN", "value": "sk-litellm-sophnet-azure-local" },
    { "name": "ANTHROPIC_DEFAULT_SONNET_MODEL", "value": "sophnet-claude-opus-4-7" },
    { "name": "ANTHROPIC_DEFAULT_OPUS_MODEL", "value": "azure-gpt-5.5" },
    { "name": "ANTHROPIC_DEFAULT_HAIKU_MODEL", "value": "deepseek-v4-flash" },
    { "name": "ANTHROPIC_CUSTOM_MODEL_OPTION", "value": "sophnet-glm-5.1" },
    { "name": "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME", "value": "GLM 5.1" },
    { "name": "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION", "value": "轻量 · Sophnet GLM-5.1" },
    { "name": "ENABLE_TOOL_SEARCH", "value": "true" },
    { "name": "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "value": "1" }
  ]
}
```

**② 用户级** `~/.claude/settings.json`（扩展从 Dock 启动时也生效，与 VS Code 设置互补）：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:4000",
    "ANTHROPIC_AUTH_TOKEN": "sk-litellm-sophnet-azure-local",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "sophnet-claude-opus-4-7",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "azure-gpt-5.5",
    "ANTHROPIC_DEFAULT_OPUS_MODEL_DESCRIPTION": "规划 · Azure GPT-5.5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL_DESCRIPTION": "极速 · DeepSeek V4 Flash（官方）",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "sophnet-glm-5.1",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "GLM 5.1",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": "轻量 · Sophnet GLM-5.1",
    "ENABLE_TOOL_SEARCH": "true",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  },
  "availableModels": ["sonnet", "opus", "haiku"]
}
```

配置说明：

| 变量 | 作用 |
|------|------|
| `ANTHROPIC_BASE_URL` | 指向 LiteLLM，而非 `api.anthropic.com` |
| `ANTHROPIC_AUTH_TOKEN` | 填 LiteLLM `master_key`，Proxy 用它鉴权 |
| `LITELLM_PROXY_*` | VS Code 扩展专用，与 `disableLoginPrompt` 配合跳过 Anthropic 登录页 |
| `ANTHROPIC_DEFAULT_*_MODEL` | 覆盖 CC 默认的 `claude-sonnet-4-5` 等名字，必须与 `config.yaml` 别名一致 |
| `ANTHROPIC_CUSTOM_MODEL_OPTION` | 第 4 个 `/model` 选项（本方案为 `sophnet-glm-5.1`） |
| `ENABLE_TOOL_SEARCH` | 走第三方网关时建议开启 MCP tool 转发 |

**③ 完全退出并重启 VS Code**（`Cmd+Q`），再打开本项目。

### 3.2 在 CC 面板内切换模型

会话中输入：

```
/model azure-gpt-5.5          # Opus
/model deepseek-v4-flash      # Haiku
/model sophnet-glm-5.1        # GLM 5.1（或选单第 4 项）
/model sophnet-claude-opus-4-7
```

或在设置里改 `claudeCode.selectedModel`。

### 3.3 从 Dock 启动 VS Code 时环境变量不生效？

macOS 从 Dock 打开 VS Code 不会继承 shell 的 `export`。可选：

```bash
# 写入 LaunchAgent 环境（重启 VS Code 后生效）
launchctl setenv ANTHROPIC_BASE_URL "http://localhost:4000"
launchctl setenv ANTHROPIC_AUTH_TOKEN "sk-litellm-sophnet-azure-local"
```

更稳妥的做法是依赖 **`~/.claude/settings.json`** 的 `env` 块（见 3.1 ②）。

### 3.4 验证 CC 通路

```bash
cd deploy/sophnet-azure
python3 test_corner_cases.py --quick
```

或手动：

```bash
curl -s http://localhost:4000/v1/messages \
  -H "Authorization: Bearer sk-litellm-sophnet-azure-local" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "deepseek-v4-flash",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "Reply OK"}]
  }'
```

---

## 4. OpenAI Codex 配置

Codex 使用 **OpenAI Responses API**（`/v1/responses`）。配置写在 **用户级** `~/.codex/config.toml`（**不要**写在项目 `.codex/config.toml` 里，`openai_base_url` 会被忽略并告警）。

CLI / VS Code 里的 **Select Model** 菜单来自 `model_catalog_json`（`~/.codex/models.json`），不是自动读 LiteLLM `/v1/models`。增删模型别名后需同步更新该 JSON。

### 4.1 推荐完整配置（含模型选择菜单）

`~/.codex/config.toml`：

```toml
model = "deepseek-v4-flash"
model_provider = "litellm"
model_catalog_json = "/Users/zyt/.codex/models.json"

approval_policy = "on-request"
sandbox_mode = "workspace-write"
model_reasoning_effort = "medium"

[model_providers.litellm]
name = "LiteLLM Proxy"
base_url = "http://localhost:4000/v1"
wire_api = "responses"
experimental_bearer_token = "sk-litellm-sophnet-azure-local"
```

`~/.codex/models.json` 中当前 9 个可选模型（按菜单顺序）：

| 别名 | 标签 |
|------|------|
| `sophnet-claude-opus-4-7` | 主模型 |
| `sophnet-glm-5.1` | 轻量 |
| `deepseek-v4-flash` | **推荐**（DeepSeek 官方） |
| `deepseek-v4-pro` | 推理（DeepSeek 官方） |
| `sophnet-gpt-5.5` | 通用 |
| `sophnet-deepseekv4-pro` | 推理（Sophnet） |
| `sophnet-deepseekv4-flash` | 极速（Sophnet） |
| `azure-gpt-5.5` | 规划 |
| `azure-gpt-5.4` | 经济 |

新增 LiteLLM 别名时：在 `models.json` 的 `models` 数组里复制一条现有条目，改 `slug` / `display_name` / `description` / `priority` 即可。

### 4.2 简化配置（无自定义模型菜单）

**方案 A：DeepSeek 直连（推荐，默认 flash）**

```toml
model = "deepseek-v4-flash"
model_provider = "openai"
openai_base_url = "http://localhost:4000/v1"
```

高质量任务改用 `model = "deepseek-v4-pro"`。

**方案 B：Azure GPT（稳定，无 thinking）**

```toml
model = "azure-gpt-5.4"
model_provider = "openai"
openai_base_url = "http://localhost:4000/v1"
```

**方案 C：Sophnet GPT**

```toml
model = "sophnet-gpt-5.5"
model_provider = "openai"
openai_base_url = "http://localhost:4000/v1"
```

**方案 D：多模型 Profile（Codex 内快速切换）**

```toml
model = "deepseek-v4-flash"
model_provider = "openai"
openai_base_url = "http://localhost:4000/v1"

[profiles.deepseek-flash]
model = "deepseek-v4-flash"

[profiles.deepseek-pro]
model = "deepseek-v4-pro"

[profiles.azure-gpt54]
model = "azure-gpt-5.4"

[profiles.azure-gpt55]
model = "azure-gpt-5.5"

[profiles.sophnet-gpt]
model = "sophnet-gpt-5.5"
```

**方案 E：自定义 Provider 块（多网关时）**

```toml
model = "deepseek-v4-flash"
model_provider = "litellm"

[model_providers.litellm]
name = "LiteLLM sophnet-azure"
base_url = "http://localhost:4000/v1"
wire_api = "responses"
env_key = "OPENAI_API_KEY"
```

所有方案均需在 `~/.codex/.env` 设置：

```bash
OPENAI_API_KEY=sk-litellm-sophnet-azure-local
```

### 4.3 设置 API Key

在 `~/.codex/.env`（或 `~/.zshrc`）中：

```bash
export OPENAI_API_KEY="sk-litellm-sophnet-azure-local"
```

> Codex 新版本优先读 `config.toml` 的 `openai_base_url`；`OPENAI_BASE_URL` 环境变量已弃用，仅作后备。

### 4.4 VS Code Codex 扩展

1. 安装扩展 **Codex**（发布者 OpenAI，ID：`openai.chatgpt`）
2. 一般 **无需** 改 `chatgpt.cliExecutable`
3. 修改 `~/.codex/config.toml` 后：**重新加载 VS Code 窗口**（`Cmd+Shift+P` → `Developer: Reload Window`）
4. 在 Codex 侧栏选择 `model`，或在 CLI 用 `--profile deepseek-pro` 切换（若配置了 Profile）

**当前可用 Codex 模型名**（与 `config.yaml` 一致）：

```
deepseek-v4-flash      ← 推荐默认
deepseek-v4-pro
azure-gpt-5.4
azure-gpt-5.5
sophnet-gpt-5.5
sophnet-deepseekv4-pro
sophnet-deepseekv4-flash
```

若使用代理/VPN，可在 `~/.codex/.env` 增加：

```bash
http_proxy="http://127.0.0.1:7890"
https_proxy="http://127.0.0.1:7890"
```

### 4.5 验证 Codex 通路

```bash
# DeepSeek flash（推荐）
curl -s http://localhost:4000/v1/responses \
  -H "Authorization: Bearer sk-litellm-sophnet-azure-local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "input": "Say OK in one word",
    "max_output_tokens": 32
  }' | python3 -m json.tool | head -20

# Azure GPT
curl -s http://localhost:4000/v1/responses \
  -H "Authorization: Bearer sk-litellm-sophnet-azure-local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "azure-gpt-5.4",
    "input": "Say OK in one word",
    "max_output_tokens": 32
  }' | python3 -m json.tool | head -20
```

CLI 快速测试（已安装 `@openai/codex` 时）：

```bash
export OPENAI_API_KEY="sk-litellm-sophnet-azure-local"
codex --model deepseek-v4-flash "Say OK"
codex --model deepseek-v4-pro "Say OK"
codex --model azure-gpt-5.4 "Say OK"
```

---

## 5. 配置对照总览

| 工具 | 协议 | Base URL | 认证 | 模型名示例 |
|------|------|----------|------|------------|
| Claude Code | `/v1/messages` | `http://localhost:4000` | `ANTHROPIC_AUTH_TOKEN` = master key | `sophnet-claude-opus-4-7` |
| Codex | `/v1/responses` | `http://localhost:4000/v1` | `OPENAI_API_KEY` = master key | `deepseek-v4-flash`（推荐） |

---

## 6. 常见问题

### CC 仍弹出 Anthropic 登录页

1. 确认 `claudeCode.disableLoginPrompt`: `true`
2. 确认 `LITELLM_PROXY_URL` / `LITELLM_PROXY_API_KEY` 已写入 VS Code 设置
3. 同步配置到 `~/.claude/settings.json`
4. 完全退出 VS Code 后重开

### CC 报 `model not found`

CC 请求的模型名与 `config.yaml` 的 `model_name` 不一致。在 LiteLLM 日志或 UI 里查看实际请求的 model，然后：

- 改 `ANTHROPIC_DEFAULT_*_MODEL`，或
- 在 CC 里 `/model <别名>`

### CC / Codex 报 402「余额不足」

Sophnet 上游 Key 欠费。查询余额：

```bash
curl -s -H "Authorization: Bearer <SOPHNET_API_KEY>" \
  https://www.sophnet.com/api/open-apis/projects/balance
```

### Codex 仍请求 `api.openai.com`

1. 确认 `openai_base_url` 在 **`~/.codex/config.toml` 顶层**，不在 `[model_providers.openai]`
2. 不要在项目 `.codex/config.toml` 里写 `openai_base_url`
3. 重载 VS Code 窗口

### CC 扩展删除了 `claudeCode.*` 设置

已知问题（部分版本）。改用 `~/.claude/settings.json` 的 `env` 块，或 `launchctl setenv` 注入系统环境变量。

### Proxy 改代码后不生效

见 [§1.4 日常运维](#14-日常运维)：`./restart-proxy.sh`，并确认 bind-mount 路径正确。

### Docker 容器 unhealthy / 启动失败

见 [§1.7 Docker 部署常见问题](#17-docker-部署常见问题)。

---

## 7. 一键回归测试

Corner-case 矩阵见 `TESTING_WORKFLOW.md`。主入口：

```bash
cd deploy/sophnet-azure

# 默认：quick（连通 + 历史故障路径）
python3 test_corner_cases.py
python3 test_corner_cases.py --quick

# 含 streaming、reasoning、adapter 扩展用例
python3 test_corner_cases.py --full

# 小并发压力（可能触发上游 429，勿在 CI 默认跑）
python3 test_corner_cases.py --stress

# 列出全部 case 与绑定原因
python3 test_corner_cases.py --list
```

stdout 仅为 JSON；人类摘要写在 stderr。管道解析示例：

```bash
python3 test_corner_cases.py --quick 2>/dev/null | python3 -m json.tool
```

---

## 8. 参考

- LiteLLM CC 快速入门：[`cookbook/ai_coding_tool_guides/claude_code_quickstart/guide.md`](../../cookbook/ai_coding_tool_guides/claude_code_quickstart/guide.md)
- LiteLLM Codex： https://docs.litellm.ai/docs/tutorials/openai_codex
- Claude Code 环境变量：https://code.claude.com/docs/en/env-vars
- Codex 高级配置：https://developers.openai.com/codex/config-advanced
- 本分支 Proxy 运维： [`.cursor/rules/sophnet-azure-proxy.mdc`](../../.cursor/rules/sophnet-azure-proxy.mdc)
