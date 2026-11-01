#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export INTERJECT_MODEL="${INTERJECT_MODEL:-./models/qwen2.5-1.5b}"
PORT="${INTERJECT_PORT:-8100}"
echo "模型: $INTERJECT_MODEL"
echo "地址: http://127.0.0.1:$PORT/chat"
python -m interject.server --port "$PORT"