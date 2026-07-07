#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CONFIG_PATH:?Set CONFIG_PATH to an inference-ready YAML file.}"
DEVICE="${DEVICE:-cuda:0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"

cd "$ROOT_DIR"

if [ -n "$VENV_ACTIVATE" ]; then
  # shellcheck source=/dev/null
  . "$VENV_ACTIVATE"
fi

python -m NullFlow.evaluation.solve \
  --config "$CONFIG_PATH" \
  --device "$DEVICE"
