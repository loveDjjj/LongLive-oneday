#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- Edit these defaults for the target server. ----
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11"
export NPROC_PER_NODE="12"
export DP_SIZE="6"
# This adapter does not explicitly bind VBench workers to different NPUs. Keep
# evaluation serial unless the installed AISBench version provides that mapping.
export AISBENCH_MAX_WORKERS="${AISBENCH_MAX_WORKERS:-1}"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29530"
export GENERATION_ENV="/mnt/share/r50063443/conda_envs/longlive"
export AISBENCH_ENV="/mnt/share/r50063443/conda_envs/aisbench_npu"
export VBENCH_CACHE_DIR="/mnt/weight/vbench_models/"

AUGMENTED_CONFIG_PATH="${AUGMENTED_CONFIG_PATH:-configs/benchmarks/vbench_standard_20pct_augmented_5s_npu_bf16.yaml}"
STANDARD_CONFIG_PATH="${STANDARD_CONFIG_PATH:-configs/benchmarks/vbench_standard_20pct_5s_npu_bf16.yaml}"

# Leave these empty for new runs. Set an existing run id to resume one stage.
AUGMENTED_RUN_ID="${AUGMENTED_RUN_ID:-}"
STANDARD_RUN_ID="${STANDARD_RUN_ID:-}"

run_stage() {
  local label="$1"
  local launcher="$2"
  local config_path="$3"
  local resume_run_id="$4"

  echo "[all] starting ${label}: config=${config_path}"
  if [[ -n "${resume_run_id}" ]]; then
    RUN_ID="${resume_run_id}" CONFIG_PATH="${config_path}" bash "${launcher}"
  else
    env -u RUN_ID CONFIG_PATH="${config_path}" bash "${launcher}"
  fi
  echo "[all] completed ${label}"
}

echo "[all] devices=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[all] layout: nproc=${NPROC_PER_NODE}, SP2 x DP${DP_SIZE}"
echo "[all] order: augmented -> standard"

run_stage \
  "VBench 20% augmented" \
  "${SCRIPT_DIR}/run_npu_vbench_augmented_pipeline.sh" \
  "${AUGMENTED_CONFIG_PATH}" \
  "${AUGMENTED_RUN_ID}"

run_stage \
  "VBench 20% standard" \
  "${SCRIPT_DIR}/run_npu_vbench_standard_pipeline.sh" \
  "${STANDARD_CONFIG_PATH}" \
  "${STANDARD_RUN_ID}"

echo "[all] both VBench generation and AISBench evaluation stages completed"
