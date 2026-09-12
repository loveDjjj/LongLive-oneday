#!/usr/bin/env bash
set -euo pipefail

# ---------- 可覆盖参数 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPARSE_METHOD=hsa_sla_cag
if [[ "${LLV2_DEVICE:-npu}" == "cuda" ]]; then
    export LONGLIVE_SP_SIZE="${LONGLIVE_SP_SIZE:-${SP_SIZE:-4}}"
    export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
else
    export LONGLIVE_SP_SIZE="${LONGLIVE_SP_SIZE:-${SP_SIZE:-8}}"
    export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
fi
export MAX_ITERS="${MAX_ITERS:-200}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-5}"
exec bash "${SCRIPT_DIR}/run_sparse_cag.sh" "$@"
