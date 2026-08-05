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
if [[ -z "${checkpoint}" ]]; then
  echo "[error] provide the merged checkpoint as the first argument or set LONGLIVE_GENERATOR_CKPT" >&2
  exit 2
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "[error] merged generator checkpoint not found: ${checkpoint}" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_id_prefix="${RUN_ID_PREFIX:-hsa_cag_20pct_${timestamp}}"
if [[ "${run_id_prefix}" == */* ]]; then
  echo "[error] RUN_ID_PREFIX must be a directory-name prefix: ${run_id_prefix}" >&2
  exit 2
fi

export LONGLIVE_GENERATOR_CKPT="${checkpoint}"
export LONGLIVE_SPARSE_METHOD=hsa_cag

echo "[suite] checkpoint=${LONGLIVE_GENERATOR_CKPT} method=${LONGLIVE_SPARSE_METHOD}"
echo "[suite] 1/2 standard 20%"
RUN_ID="${run_id_prefix}_standard" \
  bash scripts/evaluation/run_vbench.sh longlive2_standard_20pct

echo "[suite] 2/2 augmented 20%"
RUN_ID="${run_id_prefix}_augmented" \
  bash scripts/evaluation/run_vbench.sh longlive2_augmented_20pct

echo "[suite] completed: ${run_id_prefix}_standard, ${run_id_prefix}_augmented"
