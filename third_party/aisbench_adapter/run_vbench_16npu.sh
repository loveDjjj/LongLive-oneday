#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export LONGLIVE_VBENCH_DATA_PATH="${LONGLIVE_VBENCH_DATA_PATH:-${REPO_ROOT}/videos/benchmarks/vbench_mini_5s_vbench}"
export LONGLIVE_VBENCH_FULL_INFO="${LONGLIVE_VBENCH_FULL_INFO:-${REPO_ROOT}/data/benchmarks/vbench_mini/VBench_full_info.json}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${HOME}/.cache/vbench}"

# Prefer the Conda C++ runtime over an older /usr/lib64/libstdc++.so.6.
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if ! command -v ais_bench >/dev/null 2>&1; then
  echo "[error] ais_bench is not available in the current environment." >&2
  echo "        Activate the AISBench environment or install the repository first." >&2
  exit 1
fi

if ! DECORD_IMPORT_ERROR="$(python -c 'import decord; print(decord.__version__)' 2>&1)"; then
  echo "[error] decord cannot load in the current environment:" >&2
  echo "${DECORD_IMPORT_ERROR}" >&2
  echo >&2
  echo "The usual cause is an old libstdc++.so.6 missing GLIBCXX_3.4.32." >&2
  echo "Check the active Conda runtime with:" >&2
  echo "  strings \"\${CONDA_PREFIX}/lib/libstdc++.so.6\" | grep GLIBCXX_3.4.32" >&2
  echo "If the symbol is absent, install GCC 13 runtime libraries or rebuild decord from source." >&2
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
