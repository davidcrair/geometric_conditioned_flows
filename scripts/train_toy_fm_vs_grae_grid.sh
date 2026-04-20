#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

"$ROOT_DIR/scripts/train_toy_fm_orthogonal_grid.sh" "$@"
"$ROOT_DIR/scripts/train_toy_grae_grid.sh" "$@"
"$ROOT_DIR/scripts/train_toy_fm_grae_grid.sh" "$@"
