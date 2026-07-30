#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- Runtime layout. Edit here or override with environment variables. ----
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-12}"
export SP_SIZE="${SP_SIZE:-2}"
export DP_SIZE="${DP_SIZE:-$((NPROC_PER_NODE / SP_SIZE))}"
export AISBENCH_MAX_WORKERS="${AISBENCH_MAX_WORKERS:-${NPROC_PER_NODE}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29730}"
export GENERATION_ENV="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}"
export AISBENCH_ENV="${AISBENCH_ENV:-/mnt/share/r50063443/conda_envs/aisbench_npu}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/mnt/weight/vbench_models/}"

WAN22_AUGMENTED_CONFIG_PATH="${WAN22_AUGMENTED_CONFIG_PATH:-configs/benchmarks/wan22_vbench_standard_20pct_augmented_125f_npu_bf16.yaml}"
WAN22_STANDARD_CONFIG_PATH="${WAN22_STANDARD_CONFIG_PATH:-configs/benchmarks/wan22_vbench_standard_20pct_125f_npu_bf16.yaml}"
WAN22_AUGMENTED_RUN_ID="${WAN22_AUGMENTED_RUN_ID:-}"
WAN22_STANDARD_RUN_ID="${WAN22_STANDARD_RUN_ID:-}"

run_stage() {
  local label="$1" launcher="$2" config_path="$3" resume_run_id="$4"
  echo "[wan22-all] starting ${label}"
  if [[ -n "${resume_run_id}" ]]; then
    RUN_ID="${resume_run_id}" CONFIG_PATH="${config_path}" bash "${launcher}"
  else
    env -u RUN_ID CONFIG_PATH="${config_path}" bash "${launcher}"
  fi
  echo "[wan22-all] completed ${label}"
}

echo "[wan22-all] devices=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[wan22-all] layout: nproc=${NPROC_PER_NODE}, SP${SP_SIZE} x DP${DP_SIZE}"
echo "[wan22-all] protocol: native BF16, 50-step UniPC, CFG 5.0, 125 frames"

run_stage \
  "VBench 20% augmented" \
  "${SCRIPT_DIR}/run_npu_wan22_vbench_augmented_pipeline.sh" \
  "${WAN22_AUGMENTED_CONFIG_PATH}" \
  "${WAN22_AUGMENTED_RUN_ID}"

run_stage \
  "VBench 20% standard" \
  "${SCRIPT_DIR}/run_npu_wan22_vbench_standard_pipeline.sh" \
  "${WAN22_STANDARD_CONFIG_PATH}" \
  "${WAN22_STANDARD_RUN_ID}"

echo "[wan22-all] both native Wan2.2 VBench stages completed"
