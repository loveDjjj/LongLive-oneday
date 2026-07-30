#!/usr/bin/env bash

set -euo pipefail

echo "[deprecated] use scripts/run_npu_vbench_all.sh; forwarding with the same environment" >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_npu_vbench_all.sh" "$@"
