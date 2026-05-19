#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${OPEN_QUANT_AGENT_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
SLEEP_SECONDS="${OPEN_QUANT_AGENT_SLEEP_SECONDS:-900}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

while true; do
  "$PYTHON" -m open_quant_agent.cli \
    --offline-synthetic \
    --max-iterations 1 \
    --universe-size "${OPEN_QUANT_AGENT_UNIVERSE_SIZE:-48}" \
    --log "${OPEN_QUANT_AGENT_LOG:-logs/open_quant_agent.log}"
  sleep "$SLEEP_SECONDS"
done
