#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [merged_generator_checkpoint]" >&2
  exit 2
fi
shared_checkpoint="${1:-${LONGLIVE_GENERATOR_CKPT:-}}"

METHODS="${METHODS:-dense,hsa_cag,sla_cag,hsa_sla_cag}"
VBENCH_PRESETS="${VBENCH_PRESETS:-longlive2_standard_20pct}"
SUITE_ID="${SUITE_ID:-vbench_sparse_matrix_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_SUITE="${RESUME_SUITE:-1}"
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "[error] DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESUME_SUITE}" != "0" && "${RESUME_SUITE}" != "1" ]]; then
  echo "[error] RESUME_SUITE must be 0 or 1" >&2
  exit 2
fi
IFS=',' read -r -a methods <<<"${METHODS}"
IFS=',' read -r -a presets <<<"${VBENCH_PRESETS}"

checkpoint_for_method() {
  local method="$1" variable_name checkpoint
  variable_name="$(tr '[:lower:]' '[:upper:]' <<<"${method}")_GENERATOR_CKPT"
  checkpoint="${!variable_name:-${shared_checkpoint}}"
  if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
    echo "[error] provide an existing checkpoint for ${method} via ${variable_name} or the shared argument" >&2
    exit 2
  fi
  printf '%s' "${checkpoint}"
}

for preset in "${presets[@]}"; do
  for method in "${methods[@]}"; do
    if [[ ! "${method}" =~ ^(dense|hsa_cag|sla_cag|hsa_sla_cag)$ ]]; then
      echo "[error] unsupported method: ${method}" >&2
      exit 2
    fi
    checkpoint="$(checkpoint_for_method "${method}")"
    echo "[suite] checkpoint=${checkpoint} preset=${preset} method=${method}"
    run_id="${SUITE_ID}-${preset}-${method}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi
    run_dir="runs/vbench/${run_id}"
    if [[ -f "${run_dir}/vbench_results.json" && "${RESUME_SUITE}" == "1" ]]; then
      echo "[resume] completed case skipped: ${run_id}"
      continue
    fi
    LONGLIVE_GENERATOR_CKPT="${checkpoint}" \
    LONGLIVE_SPARSE_METHOD="${method}" \
    RUN_ID="${run_id}" \
      bash scripts/evaluation/run_vbench.sh "${preset}"
  done
done

if [[ "${DRY_RUN}" == "0" ]]; then
  suite_python="${GENERATION_ENV:-/path/to/conda_envs/longlive}/bin/python"
  [[ -x "${suite_python}" ]] || suite_python="python"
  "${suite_python}" scripts/evaluation/summarize_suite.py vbench \
    --suite-id "${SUITE_ID}"
fi
