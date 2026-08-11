#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 path/to/latent.pt" >&2
  exit 2
fi

LATENT_PATH="$1"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export GENERATION_ENV="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}"
export CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh}"
VAE_CHUNK_FRAMES="${VAE_CHUNK_FRAMES:-8}"
VAE_PROFILE_LATENT_FRAMES="${VAE_PROFILE_LATENT_FRAMES:-16}"
MSPROF_AI_CORE="${MSPROF_AI_CORE:-true}"

if [[ ! -f "${LATENT_PATH}" || ! -f "${CANN_ENV_SCRIPT}" ]]; then
  echo "[error] missing latent or CANN environment: ${LATENT_PATH}, ${CANN_ENV_SCRIPT}" >&2
  exit 1
fi
if [[ ! -x "${GENERATION_ENV}/bin/python" ]]; then
  echo "[error] missing generation Python: ${GENERATION_ENV}/bin/python" >&2
  exit 1
fi
if [[ ! "${VAE_CHUNK_FRAMES}" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "${VAE_PROFILE_LATENT_FRAMES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] VAE_CHUNK_FRAMES and VAE_PROFILE_LATENT_FRAMES must be positive integers" >&2
  exit 1
fi
if (( VAE_PROFILE_LATENT_FRAMES % VAE_CHUNK_FRAMES != 0 )); then
  echo "[error] VAE_PROFILE_LATENT_FRAMES must be divisible by VAE_CHUNK_FRAMES" >&2
  exit 1
fi
if [[ "${MSPROF_AI_CORE}" != "true" && "${MSPROF_AI_CORE}" != "false" ]]; then
  echo "[error] MSPROF_AI_CORE must be true or false" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "${CANN_ENV_SCRIPT}"
set -u
export CONDA_PREFIX="${GENERATION_ENV}"
export PATH="${GENERATION_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${GENERATION_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

for command_name in msprof msprof-analyze; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "[error] command not found: ${command_name}" >&2
    exit 1
  fi
done

timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="${RUN_ID:-vae-msprof-${timestamp}}"
if [[ "${run_id}" == */* ]]; then
  echo "[error] RUN_ID must be a directory name: ${run_id}" >&2
  exit 1
fi
run_dir="runs/vae_msprof/${run_id}"
log_dir="logs/vae_msprof/${run_id}"
profile_dir="${run_dir}/profiling/raw"
analysis_dir="${run_dir}/profiling/analysis"
raw_log="${log_dir}/msprof.log"
mkdir -p "${profile_dir}" "${analysis_dir}" "${log_dir}"

"${GENERATION_ENV}/bin/python" - "${run_dir}/manifest.json" "${LATENT_PATH}" \
  "${ASCEND_RT_VISIBLE_DEVICES}" "${VAE_CHUNK_FRAMES}" "${VAE_PROFILE_LATENT_FRAMES}" <<'PY'
import json
import sys

path, latent, devices, chunk, frames = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "task": "vae_msprof",
            "latent": latent,
            "visible_devices": devices,
            "chunk_frames": int(chunk),
            "profile_latent_frames": int(frames),
        },
        handle,
        indent=2,
    )
PY

profiler_args=(
  --output="${profile_dir}"
  --type=db
  --storage-limit=20000MB
  --ascendcl=on
  --model-execution=on
  --runtime-api=on
  --task-time=on
  --aicpu=on
  --sys-hardware-mem=on
)
if [[ "${MSPROF_AI_CORE}" == "true" ]]; then
  profiler_args+=(--ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization)
fi

echo "[run] task=vae_msprof run_id=${run_id} device=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[run] latent=${LATENT_PATH} frames=${VAE_PROFILE_LATENT_FRAMES} chunk=${VAE_CHUNK_FRAMES}"
set +e
LLV2_DEVICE=npu msprof "${profiler_args[@]}" \
  "${GENERATION_ENV}/bin/python" tests/npu/benchmark_vae_decode.py \
    --latent "${LATENT_PATH}" \
    --device npu:0 \
    --chunk-frames "${VAE_CHUNK_FRAMES}" \
    --max-latent-frames "${VAE_PROFILE_LATENT_FRAMES}" \
    --iterations 1 \
  2>&1 | tee "${raw_log}"
profile_status="${PIPESTATUS[0]}"
set -e

mapfile -t prof_dirs < <(find "${profile_dir}" -type d -name 'PROF_*' | sort)
if [[ "${#prof_dirs[@]}" -eq 0 ]]; then
  echo "[error] no PROF_* data found under ${profile_dir}" >&2
  if [[ "${profile_status}" -ne 0 ]]; then
    exit "${profile_status}"
  fi
  exit 1
fi
analyze_input="${profile_dir}"
if [[ "${#prof_dirs[@]}" -eq 1 ]]; then
  analyze_input="${prof_dirs[0]}"
fi
msprof-analyze -m compute_op_sum -d "${analyze_input}" \
  -o "${analysis_dir}/compute_op_sum" --export_type text --parallel_mode concurrent
msprof-analyze -m free_analysis -d "${analyze_input}" \
  -o "${analysis_dir}/free_analysis" --export_type text || true

echo "[done] run=${run_dir}"
if [[ "${profile_status}" -ne 0 ]]; then
  exit "${profile_status}"
fi
