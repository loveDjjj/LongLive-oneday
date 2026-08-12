#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPARSE_METHOD=hsa_cag
exec bash "${SCRIPT_DIR}/run_sparse_cag.sh" "$@"
