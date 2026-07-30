#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="${BENCHMARK:-standard}"

export GENERATION_ENTRYPOINT="inference_wan22_sp.py"
export GENERATION_CONFIG_KIND="wan22"

case "${BENCHMARK}" in
  standard)
    export CONFIG_PATH="${CONFIG_PATH:-configs/benchmarks/wan22_vbench_mini_5pct_125f_npu_bf16.yaml}"
    export GENERATION_PROMPTS="${GENERATION_PROMPTS:-data/benchmarks/vbench_mini/prompts.txt}"
    export NAMING_PROMPTS="${NAMING_PROMPTS:-${GENERATION_PROMPTS}}"
    export FULL_INFO="${FULL_INFO:-data/benchmarks/vbench_mini/VBench_full_info.json}"
    export RUN_TAG_OVERRIDE="${RUN_TAG_OVERRIDE:-wan22_mini_5pct_125f}"
    ;;
  augmented)
    export CONFIG_PATH="${CONFIG_PATH:-configs/benchmarks/wan22_vbench_mini_5pct_augmented_125f_npu_bf16.yaml}"
    export GENERATION_PROMPTS="${GENERATION_PROMPTS:-data/benchmarks/vbench_mini_augmented_wan21_qwen25_seed42/prompts.txt}"
    export NAMING_PROMPTS="${NAMING_PROMPTS:-data/benchmarks/vbench_mini_augmented_wan21_qwen25_seed42/original_prompts.txt}"
    export FULL_INFO="${FULL_INFO:-data/benchmarks/vbench_mini_augmented_wan21_qwen25_seed42/VBench_full_info.json}"
    export RUN_TAG_OVERRIDE="${RUN_TAG_OVERRIDE:-wan22_mini_augmented_5pct_125f}"
    ;;
  *)
    echo "[error] BENCHMARK must be standard or augmented, got: ${BENCHMARK}" >&2
    exit 2
    ;;
esac

exec bash "${SCRIPT_DIR}/run_npu_vbench_quality_pipeline.sh" "$@"
