#!/usr/bin/env bash
set -euo pipefail

# ---- Runtime layout. Edit here or override with environment variables. ----
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-12}"
export SP_SIZE="${SP_SIZE:-2}"
export DP_SIZE="${DP_SIZE:-$((NPROC_PER_NODE / SP_SIZE))}"
export AISBENCH_MAX_WORKERS="${AISBENCH_MAX_WORKERS:-${NPROC_PER_NODE}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29530}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK=standard exec bash "${SCRIPT_DIR}/run_npu_vbench_quality_pipeline.sh" "$@"
