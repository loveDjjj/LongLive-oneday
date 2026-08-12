#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPARSE_METHOD=sla_cag
export SP_SIZE="${SP_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export MAX_ITERS="${MAX_ITERS:-1000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-5}"
exec bash "${SCRIPT_DIR}/run_sparse_cag.sh" "$@"
