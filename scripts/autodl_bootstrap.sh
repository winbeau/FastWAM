#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${FASTWAM_AUTODL_CACHE_ROOT:-/root/autodl-tmp}"
ARTIFACT_DIR="${FASTWAM_AUTODL_ARTIFACT_DIR:-${ROOT}/artifacts/autodl}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv-cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/cache}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$ARTIFACT_DIR"

cd "$ROOT"
uv sync --frozen
FASTWAM_PREFLIGHT_OUTPUT="$ARTIFACT_DIR/preflight.json" \
  uv run --frozen python scripts/autodl_preflight.py
uv run --frozen python scripts/autodl_smoke.py import --output "$ARTIFACT_DIR/import.json"
uv run --frozen python scripts/validate_panthera_dataset.py \
  tests/fixtures/panthera_lerobot_v3_minimal \
  --output "$ARTIFACT_DIR/dataset-preflight.json"
uv run --frozen python scripts/autodl_smoke.py data --output "$ARTIFACT_DIR/data.json"
uv run --frozen python scripts/autodl_smoke.py structural --output "$ARTIFACT_DIR/structural.json"
