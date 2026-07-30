#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="${BENCHMARK:-standard}"

export GENERATION_ENTRYPOINT="inference_wan22_sp.py"
export GENERATION_CONFIG_KIND="wan22"

case "${BENCHMARK}" in
  standard)
    export CONFIG_PATH="${CONFIG_PATH:-configs/benchmarks/wan22_vbench_standard_20pct_125f_npu_bf16.yaml}"
    export RUN_TAG_OVERRIDE="${RUN_TAG_OVERRIDE:-wan22_standard_20pct_125f}"
    ;;
  augmented)
    export CONFIG_PATH="${CONFIG_PATH:-configs/benchmarks/wan22_vbench_standard_20pct_augmented_125f_npu_bf16.yaml}"
    export RUN_TAG_OVERRIDE="${RUN_TAG_OVERRIDE:-wan22_augmented_20pct_125f}"
    ;;
  *)
    echo "[error] BENCHMARK must be standard or augmented, got: ${BENCHMARK}" >&2
    exit 2
    ;;
esac

exec bash "${SCRIPT_DIR}/run_npu_vbench_quality_pipeline.sh" "$@"
