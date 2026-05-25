#!/usr/bin/env bash
# 从本分支 litellm/ 源码同步补丁到 deploy/litellm-local/patch/
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/../../litellm"
PATCH="$DIR/patch"

PATHS=(
  llms/openai/openai.py
  llms/openai/chat/gpt_transformation.py
  llms/deepseek/chat/transformation.py
  llms/azure/chat/gpt_transformation.py
  llms/anthropic/experimental_pass_through/adapters/handler.py
  responses/main.py
  responses/litellm_completion_transformation/handler.py
  responses/litellm_completion_transformation/transformation.py
)

for rel in "${PATHS[@]}"; do
  dest="$PATCH/$rel"
  mkdir -p "$(dirname "$dest")"
  cp "$SRC/$rel" "$dest"
done

echo "Synced patch from branch litellm/ -> $PATCH"
