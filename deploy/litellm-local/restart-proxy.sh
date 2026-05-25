#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "Restarting litellm-local..."
docker compose restart litellm

echo -n "Waiting for health"
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:4000/health/liveliness >/dev/null 2>&1; then
    echo " OK"
    exit 0
  fi
  echo -n "."
  sleep 2
done
echo " TIMEOUT" >&2
exit 1
