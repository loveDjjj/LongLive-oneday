#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export LONGLIVE_VBENCH_DATA_PATH="${LONGLIVE_VBENCH_DATA_PATH:-${REPO_ROOT}/videos/benchmarks/vbench_mini_5s_vbench}"
export LONGLIVE_VBENCH_FULL_INFO="${LONGLIVE_VBENCH_FULL_INFO:-${REPO_ROOT}/data/benchmarks/vbench_mini/VBench_full_info.json}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/mnt/weight/vbench_models/}"

# Restore CANN/HCCL paths before changing the C++ runtime search order.
CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "${CANN_ENV_SCRIPT}" ]]; then
  echo "[error] CANN environment script not found: ${CANN_ENV_SCRIPT}" >&2
  echo "        Override it with CANN_ENV_SCRIPT=/actual/path/set_env.sh" >&2
  exit 1
fi
set +u
# shellcheck disable=SC1090
source "${CANN_ENV_SCRIPT}"
set -u

# Prefer the Conda C++ runtime over an older /usr/lib64/libstdc++.so.6.
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if ! command -v ais_bench >/dev/null 2>&1; then
  echo "[error] ais_bench is not available in the current environment." >&2
  echo "        Activate the AISBench environment or install the repository first." >&2
  exit 1
fi

if ! NPU_IMPORT_ERROR="$(python -c 'import torch; import torch_npu; print(torch_npu.npu.device_count())' 2>&1)"; then
  echo "[error] torch_npu cannot load the Ascend runtime:" >&2
  echo "${NPU_IMPORT_ERROR}" >&2
  echo >&2
  echo "Source the CANN environment before running this launcher, for example:" >&2
  echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh" >&2
  echo "Do not concatenate LD_LIBRARY_PATH entries without a separating colon." >&2
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

if [[ ! -d "${VBENCH_CACHE_DIR}" ]]; then
  echo "[error] VBench cache directory not found: ${VBENCH_CACHE_DIR}" >&2
  exit 1
fi

RENDERED_CONFIG="$(mktemp "${TMPDIR:-/tmp}/longlive_aisbench.XXXXXX.py")"
trap 'rm -f "${RENDERED_CONFIG}"' EXIT

python - "${SCRIPT_DIR}/eval_longlive_vbench.py" "${RENDERED_CONFIG}" <<'PY'
import os
import sys
from pathlib import Path

template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
config = template_path.read_text(encoding="utf-8")
replacements = {
    '"__LONGLIVE_VBENCH_DATA_PATH__"': repr(os.environ["LONGLIVE_VBENCH_DATA_PATH"]),
    '"__LONGLIVE_VBENCH_FULL_INFO__"': repr(os.environ["LONGLIVE_VBENCH_FULL_INFO"]),
    '"__VBENCH_CACHE_DIR__"': repr(os.environ["VBENCH_CACHE_DIR"]),
}
for placeholder, value in replacements.items():
    if placeholder not in config:
        raise RuntimeError(f"missing config placeholder: {placeholder}")
    config = config.replace(placeholder, value)
output_path.write_text(config, encoding="utf-8")
PY

# AISBench resolves its default outputs/ directory against the current working
# directory. Keep results under LongLive even when this script is called by an
# absolute path from another repository.
cd "${REPO_ROOT}"

ais_bench "${RENDERED_CONFIG}" \
  --mode eval \
  --max-num-workers "${AISBENCH_MAX_WORKERS:-16}"
