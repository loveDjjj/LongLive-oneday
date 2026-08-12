#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-benchmark}"
METHODS="${METHODS:-dense,hsa_cag,sla_cag}"
DURATIONS="${DURATIONS:-5s,32s,64s}"
MODES="${MODES:-dit_only,sync_vae,async_vae}"
PERF_DEVICES="${PERF_DEVICES:-0,1,2,3,4}"
SUITE_ID="${SUITE_ID:-sparse_matrix_$(date +%Y%m%d_%H%M%S)}"

if [[ "${TASK}" != "benchmark" && "${TASK}" != "msprof" ]]; then
  echo "[error] TASK must be benchmark or msprof" >&2
  exit 2
fi
IFS=',' read -r -a devices <<<"${PERF_DEVICES}"
if (( ${#devices[@]} < 5 )); then
  echo "[error] PERF_DEVICES must contain at least five NPU ids" >&2
  exit 2
fi
workers="${devices[0]},${devices[1]},${devices[2]},${devices[3]}"
workers_and_vae="${workers},${devices[4]}"

IFS=',' read -r -a methods <<<"${METHODS}"
IFS=',' read -r -a durations <<<"${DURATIONS}"
IFS=',' read -r -a modes <<<"${MODES}"
for method in "${methods[@]}"; do
  if [[ ! "${method}" =~ ^(dense|hsa_cag|sla_cag)$ ]]; then
    echo "[error] unsupported method: ${method}" >&2
    exit 2
  fi
  for duration in "${durations[@]}"; do
    if [[ ! "${duration}" =~ ^(5s|32s|64s)$ ]]; then
      echo "[error] unsupported duration: ${duration}" >&2
      exit 2
    fi
    for mode in "${modes[@]}"; do
      if [[ ! "${mode}" =~ ^(dit_only|sync_vae|async_vae)$ ]]; then
        echo "[error] unsupported mode: ${mode}" >&2
        exit 2
      fi
      visible="${workers}"
      [[ "${mode}" == "async_vae" ]] && visible="${workers_and_vae}"
      echo "[suite] task=${TASK} method=${method} duration=${duration} mode=${mode}"
      if [[ "${TASK}" == "benchmark" ]]; then
        ASCEND_RT_VISIBLE_DEVICES="${visible}" \
        LONGLIVE_SPARSE_METHOD="${method}" \
        BENCHMARK_MODE="${mode}" \
        RUN_ID="${SUITE_ID}-${method}-${duration}-${mode}" \
          bash scripts/evaluation/run_benchmark.sh "${duration}"
      else
        ASCEND_RT_VISIBLE_DEVICES="${visible}" \
        LONGLIVE_SPARSE_METHOD="${method}" \
        MSPROF_MODE="${mode}" \
        RUN_ID="${SUITE_ID}-${method}-${duration}-${mode}" \
          bash scripts/evaluation/run_msprof.sh "${duration}"
      fi
    done
  done
done
