#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4}"
export GENERATION_ENV="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}"
export CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh}"
unset MASTER_PORT

CONFIG_PATH="${CONFIG_PATH:-configs/inference/msprof.yaml}"
PRESET="${1:-32s}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-3}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-1}"
BENCHMARK_LATENTS_ONLY="${BENCHMARK_LATENTS_ONLY:-0}"
BENCHMARK_MODE="${BENCHMARK_MODE:-async_vae}"

if [[ ! "${BENCHMARK_REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] BENCHMARK_REPEATS must be a positive integer" >&2
  exit 1
fi
if [[ ! "${BENCHMARK_WARMUP}" =~ ^[0-9]+$ ]]; then
  echo "[error] BENCHMARK_WARMUP must be a non-negative integer" >&2
  exit 1
fi
if [[ "${BENCHMARK_LATENTS_ONLY}" != "0" && "${BENCHMARK_LATENTS_ONLY}" != "1" ]]; then
  echo "[error] BENCHMARK_LATENTS_ONLY must be 0 or 1" >&2
  exit 1
fi
if [[ ! "${BENCHMARK_MODE}" =~ ^(dit_only|sync_vae|async_vae)$ ]]; then
  echo "[error] BENCHMARK_MODE must be dit_only, sync_vae, or async_vae" >&2
  exit 1
fi
if [[ "${BENCHMARK_LATENTS_ONLY}" == "1" ]]; then
  BENCHMARK_MODE="dit_only"
fi
if [[ ! -f "${CONFIG_PATH}" || ! -f "${CANN_ENV_SCRIPT}" ]]; then
  echo "[error] missing config or CANN environment: ${CONFIG_PATH}, ${CANN_ENV_SCRIPT}" >&2
  exit 1
fi
if [[ ! -x "${GENERATION_ENV}/bin/python" || ! -x "${GENERATION_ENV}/bin/torchrun" ]]; then
  echo "[error] incomplete generation environment: ${GENERATION_ENV}" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "${CANN_ENV_SCRIPT}"
set -u
export CONDA_PREFIX="${GENERATION_ENV}"
export PATH="${GENERATION_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${GENERATION_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

total_prompts=$((BENCHMARK_REPEATS + BENCHMARK_WARMUP))
metadata_tmp="$(mktemp "${TMPDIR:-/tmp}/longlive_benchmark.XXXXXX.json")"
trap 'rm -f "${metadata_tmp}"' EXIT
resolve_args=(
  benchmark
  --config "${CONFIG_PATH}"
  --preset "${PRESET}"
  --num-prompts "${total_prompts}"
  --warmup-per-rank "${BENCHMARK_WARMUP}"
  --vae-mode "${BENCHMARK_MODE}"
)
"${GENERATION_ENV}/bin/python" scripts/evaluation/resolve_config.py \
  "${resolve_args[@]}" >"${metadata_tmp}"

json_field() {
  "${GENERATION_ENV}/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "${metadata_tmp}" "$1"
}

run_tag="$(json_field run_tag)"
sparsity_method="$(json_field sparsity_method)"
sparsity_backend="$(json_field sparsity_backend)"
sp_size="$(json_field sp_size)"
dp_size="$(json_field dp_size)"
nproc="$(json_field nproc_per_node)"
required_devices="$(json_field required_devices)"

visible_count="$(awk -F, '{print NF}' <<<"${ASCEND_RT_VISIBLE_DEVICES}")"
if [[ "${visible_count}" -ne "${required_devices}" ]]; then
  echo "[error] preset ${PRESET} requires ${required_devices} visible NPUs, " \
       "got ${ASCEND_RT_VISIBLE_DEVICES}" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="${RUN_ID:-${run_tag}-${timestamp}}"
if [[ "${run_id}" == */* ]]; then
  echo "[error] RUN_ID must be a directory name: ${run_id}" >&2
  exit 1
fi
run_dir="runs/performance/${run_id}"
log_dir="logs/performance/${run_id}"
video_dir="${run_dir}/videos"
raw_log="${log_dir}/torchrun.log"
summary_file="${run_dir}/summary.txt"
summary_json="${run_dir}/summary.json"
resolved_config="${run_dir}/resolved.yaml"
if [[ -e "${run_dir}" || -e "${log_dir}" ]]; then
  echo "[error] benchmark RUN_ID already exists: ${run_id}" >&2
  exit 1
fi
mkdir -p "${video_dir}" "${log_dir}"
cp "${metadata_tmp}" "${run_dir}/manifest.json"

"${GENERATION_ENV}/bin/python" scripts/evaluation/resolve_config.py \
  "${resolve_args[@]}" --output "${resolved_config}" \
  --output-folder "${video_dir}" >/dev/null

echo "[run] task=benchmark preset=${PRESET} run_id=${run_id}"
echo "[run] devices=${ASCEND_RT_VISIBLE_DEVICES} layout=SP${sp_size}xDP${dp_size}"
echo "[run] sparsity=${sparsity_method} backend=${sparsity_backend}"
echo "[run] warmup=${BENCHMARK_WARMUP} measured=${BENCHMARK_REPEATS}"
echo "[run] mode=${BENCHMARK_MODE}"
echo "[run] rendezvous=standalone"
echo "[run] config=${resolved_config}"

set +e
LLV2_DEVICE=npu "${GENERATION_ENV}/bin/torchrun" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${nproc}" \
  inference_sp.py \
  --config_path "${resolved_config}" \
  2>&1 | tee "${raw_log}"
benchmark_status="${PIPESTATUS[0]}"
set -e

if [[ "${benchmark_status}" -ne 0 ]]; then
  echo "[error] benchmark inference failed with status ${benchmark_status}" >&2
  exit "${benchmark_status}"
fi

"${GENERATION_ENV}/bin/python" scripts/evaluation/summarize_benchmark.py \
  "${raw_log}" --warmup-per-rank "${BENCHMARK_WARMUP}" \
  --json-output "${summary_json}" | tee "${summary_file}"

echo "[done] run=${run_dir}"
