#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [merged_generator_checkpoint]" >&2
  exit 2
fi
checkpoint="${1:-${LONGLIVE_GENERATOR_CKPT:-}}"
if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
  echo "[error] provide an existing merged generator checkpoint" >&2
  exit 2
fi

METHODS="${METHODS:-dense,hsa_cag,sla_cag,hsa_sla_cag}"
VBENCH_PRESETS="${VBENCH_PRESETS:-longlive2_standard_20pct}"
SUITE_ID="${SUITE_ID:-vbench_sparse_matrix_$(date +%Y%m%d_%H%M%S)}"
IFS=',' read -r -a methods <<<"${METHODS}"
IFS=',' read -r -a presets <<<"${VBENCH_PRESETS}"

for preset in "${presets[@]}"; do
  for method in "${methods[@]}"; do
    if [[ ! "${method}" =~ ^(dense|hsa_cag|sla_cag|hsa_sla_cag)$ ]]; then
      echo "[error] unsupported method: ${method}" >&2
      exit 2
    fi
    echo "[suite] checkpoint=${checkpoint} preset=${preset} method=${method}"
    LONGLIVE_GENERATOR_CKPT="${checkpoint}" \
    LONGLIVE_SPARSE_METHOD="${method}" \
    RUN_ID="${SUITE_ID}-${preset}-${method}" \
      bash scripts/evaluation/run_vbench.sh "${preset}"
  done
done
