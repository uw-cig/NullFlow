#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-configs/ffhq256_inpainting_test.yaml}"
DEVICE="${DEVICE:-cuda:0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
ONESTEP_RUN_ID="${ONESTEP_RUN_ID:-$$-$(date +%s)}"
ONESTEP_RUN_TIMESTAMP="${ONESTEP_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT_DIR"

if [ -n "$VENV_ACTIVATE" ]; then
  # shellcheck source=/dev/null
  . "$VENV_ACTIVATE"
fi

export ONESTEP_RUN_ID
export ONESTEP_RUN_TIMESTAMP

if [ "$NPROC_PER_NODE" -gt 1 ]; then
  torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    -m NullFlow.training.train \
    --config "$CONFIG_PATH" \
    --device cuda
else
  python -m NullFlow.training.train \
    --config "$CONFIG_PATH" \
    --device "$DEVICE"
fi
