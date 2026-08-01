#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Four logical NPUs run SP4; logical npu:4 is exclusively reserved for WanVAE.
# Edit the physical IDs here or override them before launching.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export SP_SIZE="${SP_SIZE:-4}"
export DP_SIZE="${DP_SIZE:-1}"
export VAE_DEVICE="${VAE_DEVICE:-npu:${NPROC_PER_NODE}}"
export CONFIG_PATH="${CONFIG_PATH:-configs/benchmarks/perf_32s_npu_bf16_async_vae.yaml}"
export WARMUP_PER_RANK="${WARMUP_PER_RANK:-1}"

visible_count="$(awk -F, '{print NF}' <<< "${ASCEND_RT_VISIBLE_DEVICES}")"
if [[ "${visible_count}" -lt $((NPROC_PER_NODE + 1)) ]]; then
  echo "[error] async VAE needs ${NPROC_PER_NODE} worker NPUs plus one VAE NPU; " \
       "got ${ASCEND_RT_VISIBLE_DEVICES}" >&2
  exit 1
fi
if [[ "${DP_SIZE}" -ne 1 ]]; then
  echo "[error] one dedicated VAE device currently requires DP_SIZE=1" >&2
  exit 1
fi

exec bash "${SCRIPT_DIR}/run_npu_generation_benchmark.sh"
