# Agent Quick Deploy: `fix/responses-reasoning-item-id`

本 runbook 面向 agent，用于把
`https://github.com/ZhengYitong2333/litellm/tree/fix/responses-reasoning-item-id`
快速部署到本地验证。

## 当前判断

- 推荐路径：使用本目录的 Docker Compose bundle，基于官方 `main-stable` 镜像，通过 bind-mount `deploy/litellm-local/patch/` 中的补丁文件启动代理。
- 当前分支：`fix/responses-reasoning-item-id`。
- 官方远端：`upstream=https://github.com/BerriAI/litellm.git`。
- 2026-06-04 检查结果：`upstream/main` 已更新到 `5be0797d24`，tag 为 `v1.88.0-rc.1`。
- 分叉状态：当前分支和 `upstream/main` 不是快进关系；`upstream/main...HEAD` 当时为官方多 99 个提交、当前分支多 93 个提交。不要为了本地快速部署直接 merge/rebase 官方 main。

## 前置条件

```bash
docker --version
docker compose version
git --version
```

如果是第一次运行本地 LiteLLM Docker 网络：

```bash
docker network create litellm-net || true
```

## Fresh Clone

```bash
git clone https://github.com/ZhengYitong2333/litellm.git
cd litellm
git checkout fix/responses-reasoning-item-id
git remote add upstream https://github.com/BerriAI/litellm.git || true
git fetch origin fix/responses-reasoning-item-id
git fetch upstream --tags --prune
```

确认分支：

```bash
git branch --show-current
git log --oneline -5
```

期望看到 `fix/responses-reasoning-item-id`，并且顶部提交来自这个修复分支。

## Existing Repo

如果仓库已存在，先不要覆盖用户改动：

```bash
git status --short
git remote -v
git fetch origin fix/responses-reasoning-item-id
git fetch upstream --tags --prune
```

如果工作区干净，可以切分支：

```bash
git checkout fix/responses-reasoning-item-id
git pull --ff-only origin fix/responses-reasoning-item-id
```

如果工作区不干净，先停下来判断改动是否属于用户，不要执行 `git reset --hard` 或 `git checkout -- <file>`。

## 官方更新检查

用于回答“LiteLLM 官方有没有更新”：

```bash
git fetch upstream --tags --prune
git log --oneline --decorate -8 upstream/main
git describe --tags --abbrev=0 upstream/main
git rev-list --left-right --count upstream/main...HEAD
```

解释：

- `git log upstream/main` 显示官方 main 最新提交。
- `git describe upstream/main` 显示官方 main 附近最新 tag。
- `git rev-list --left-right --count upstream/main...HEAD` 的左值是官方相对当前分支多出的提交数，右值是当前分支相对官方多出的提交数。

## 快速部署: Docker Patch Bundle

进入部署目录：

```bash
cd deploy/litellm-local
```

准备环境变量：

```bash
cp .env.example .env
```

`.env` 至少需要：

```dotenv
LITELLM_MASTER_KEY=sk-openclaw-local
```

准备 `config.yaml`。最小示例：

```yaml
model_list:
  - model_name: fake-openai-endpoint
    litellm_params:
      model: openai/fake-model
      api_key: fake-key
      api_base: https://fake-api.example.com

general_settings:
  master_key: sk-openclaw-local

litellm_settings:
  drop_params: true
  telemetry: false
```

如果已有外部配置文件：

```bash
cp /path/to/litellm_config.yaml ./config.yaml
```

启动：

```bash
docker compose up -d
```

等待健康检查：

```bash
curl -sf http://127.0.0.1:4000/health/liveliness
```

查看日志：

```bash
docker compose logs -f --tail=200 litellm
```

## 客户端接入

Codex Responses API：

```text
base_url = http://127.0.0.1:4000/v1
api_key = sk-openclaw-local
wire_api = responses
```

Claude Code Anthropic Messages API：

```text
base_url = http://127.0.0.1:4000
api_key = sk-openclaw-local
```

Compose 中已设置：

```dotenv
LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES=true
```

## 更新补丁后重启

如果 agent 修改了 `litellm/` 源码，并且需要同步到 Docker bind-mount 补丁：

```bash
cd deploy/litellm-local
./sync-patch.sh
./restart-proxy.sh
```

如果只是改了 `deploy/litellm-local/patch/` 里的文件：

```bash
cd deploy/litellm-local
./restart-proxy.sh
```

## 源码方式启动

如果需要完整运行当前分支而不是 patch bundle：

```bash
uv sync --group proxy-dev --extra proxy
cat > /tmp/litellm-local-config.yaml <<'YAML'
model_list:
  - model_name: fake-openai-endpoint
    litellm_params:
      model: openai/fake-model
      api_key: fake-key
      api_base: https://fake-api.example.com

general_settings:
  master_key: sk-openclaw-local

litellm_settings:
  drop_params: true
  telemetry: false
YAML
uv run litellm --config /tmp/litellm-local-config.yaml --port 4000
```

源码方式适合验证分支全部代码，但启动和依赖安装更慢。Docker patch bundle 适合快速给本机 agent 使用。

## 验证请求

健康检查：

```bash
curl -sf http://127.0.0.1:4000/health/liveliness
```

模型列表：

```bash
curl -s http://127.0.0.1:4000/v1/models \
  -H 'Authorization: Bearer sk-openclaw-local'
```

Responses smoke test：

```bash
curl -s http://127.0.0.1:4000/v1/responses \
  -H 'Authorization: Bearer sk-openclaw-local' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "fake-openai-endpoint",
    "input": "Say hello in one sentence."
  }'
```

注意：如果使用 fake upstream，这个请求可能会因 upstream 不存在而失败；只要 LiteLLM 能启动并完成路由、认证、参数解析，就说明本地代理链路基本可用。真实验证需要把 `config.yaml` 指向可用模型。

## 常见失败点

- `network litellm-net declared as external, but could not be found`：运行 `docker network create litellm-net`。
- `401 Unauthorized`：请求的 Bearer token 必须等于 `general_settings.master_key` 或 `.env` 中实际 master key。
- `Connection refused`：代理还在启动，等 15 到 30 秒后重试，或看 `docker compose logs -f litellm`。
- `Unknown model`：`config.yaml` 中的 `model_name` 和请求体里的 `model` 不一致。
- patch 修改未生效：运行 `./sync-patch.sh && ./restart-proxy.sh`，再确认容器日志中没有旧异常。
- 官方 main 有更新：只记录状态，不要在快速部署任务中合并官方 main，除非用户明确要求处理冲突。

