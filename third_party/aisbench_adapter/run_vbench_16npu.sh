#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export LONGLIVE_VBENCH_DATA_PATH="${LONGLIVE_VBENCH_DATA_PATH:-${REPO_ROOT}/videos/benchmarks/vbench_mini_5s_vbench}"
export LONGLIVE_VBENCH_FULL_INFO="${LONGLIVE_VBENCH_FULL_INFO:-${REPO_ROOT}/data/benchmarks/vbench_mini/VBench_full_info.json}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${HOME}/.cache/vbench}"

if ! command -v ais_bench >/dev/null 2>&1; then
  echo "[error] ais_bench is not available in the current environment." >&2
  echo "        Activate the AISBench environment or install the repository first." >&2
  exit 1
fi

if [[ ! -d "${LONGLIVE_VBENCH_DATA_PATH}" ]]; then
  echo "[error] prepared video directory not found: ${LONGLIVE_VBENCH_DATA_PATH}" >&2
  exit 1
fi

if [[ ! -f "${LONGLIVE_VBENCH_FULL_INFO}" ]]; then
  echo "[error] VBench metadata not found: ${LONGLIVE_VBENCH_FULL_INFO}" >&2
  exit 1
fi

ais_bench "${SCRIPT_DIR}/eval_longlive_vbench.py" \
  --mode eval \
  --max-num-workers "${AISBENCH_MAX_WORKERS:-16}"
