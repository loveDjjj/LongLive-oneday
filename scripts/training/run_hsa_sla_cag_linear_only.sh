#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPARSE_METHOD=hsa_sla_cag
export GENERATOR_TRAIN_SCOPE=linear_only
export GENERATOR_LR="${GENERATOR_LR:-2.0e-5}"
export MAX_ITERS="${MAX_ITERS:-1000}"
exec bash "${SCRIPT_DIR}/run_sparse_cag.sh" "$@"
