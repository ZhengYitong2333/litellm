#!/usr/bin/env bash
# 从本分支 litellm/ 源码同步补丁到 deploy/litellm-local/patch/
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/../../litellm"
PATCH="$DIR/patch"

PATHS=(
  llms/deepseek/chat/transformation.py
  llms/azure/chat/gpt_transformation.py
  llms/anthropic/experimental_pass_through/adapters/handler.py
  responses/main.py
  responses/litellm_completion_transformation/handler.py
  responses/litellm_completion_transformation/transformation.py
)

# 以下 OpenAI patch 保持仓库内兼容版本，勿从分支覆盖
# llms/openai/openai.py
# llms/openai/chat/gpt_transformation.py

for rel in "${PATHS[@]}"; do
  dest="$PATCH/$rel"
  mkdir -p "$(dirname "$dest")"
  cp "$SRC/$rel" "$dest"
done

echo "Synced patch from branch litellm/ -> $PATCH"

# Optional live deploy directory (bind-mount target for running litellm-local container)
LIVE_PATCH="${LITELLM_LOCAL_PATCH_DIR:-$HOME/litellm-local/patch}"
if [[ -d "$(dirname "$LIVE_PATCH")" ]]; then
  for rel in "${PATHS[@]}"; do
    live_dest="$LIVE_PATCH/$rel"
    mkdir -p "$(dirname "$live_dest")"
    cp "$SRC/$rel" "$live_dest"
  done
  echo "Synced patch from branch litellm/ -> $LIVE_PATCH"
fi
