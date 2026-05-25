# LiteLLM Local Deploy (main-stable + bind-mount patches)

在 **官方 `main-stable` 镜像** 上通过 bind-mount 8 个补丁文件，支持：

- **Codex** `/v1/responses`（工具 normalize、Sophnet/MiniMax chat bridge）
- **Claude Code** `/v1/messages`（`output_config` 清洗、reasoning 映射）
- **DeepSeek V4** 多轮 `reasoning_content`

> 本 bundle **不** 依赖 `fix/codex-responses-tool-bridge` 的 upstream staging 合并（与 main-stable 单文件 patch 不兼容）。

## 前置条件

```bash
docker network create litellm-net   # 首次
cp .env.example .env                # 填入 LITELLM_MASTER_KEY 等
cp /path/to/litellm_config.yaml ./config.yaml   # 或设置 LITELLM_CONFIG_PATH
```

## 启动

```bash
docker compose up -d
curl http://127.0.0.1:4000/health/liveliness
```

## 更新补丁（从分支 litellm/ 源码）

```bash
./sync-patch.sh
./restart-proxy.sh
```

## 补丁文件

```
patch/
├── llms/openai/openai.py
├── llms/openai/chat/gpt_transformation.py
├── llms/deepseek/chat/transformation.py
├── llms/azure/chat/gpt_transformation.py
├── llms/anthropic/experimental_pass_through/adapters/handler.py
├── responses/main.py
└── responses/litellm_completion_transformation/
    ├── handler.py
    └── transformation.py
```

## 客户端配置

| 客户端 | Base URL | 说明 |
|--------|----------|------|
| Codex | `http://127.0.0.1:4000/v1` | `wire_api = "responses"` |
| Claude Code | `http://127.0.0.1:4000` | Anthropic `/v1/messages` |

环境变量：`LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES=true`（已在 compose 中设置）
