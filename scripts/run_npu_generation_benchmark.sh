#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- Runtime layout. Edit here or override with environment variables. ----
# Change CONFIG_PATH to perf_32s_npu_bf16.yaml or perf_64s_npu_bf16.yaml as needed.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export CONFIG_PATH="${CONFIG_PATH:-configs/benchmarks/perf_16s_npu_bf16.yaml}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export SP_SIZE="${SP_SIZE:-8}"
export DP_SIZE="${DP_SIZE:-$((NPROC_PER_NODE / SP_SIZE))}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29520}"
export WARMUP_PER_RANK="${WARMUP_PER_RANK:-1}"
export CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
export GENERATION_ENV="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[error] config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CANN_ENV_SCRIPT}" ]]; then
  echo "[error] CANN environment script not found: ${CANN_ENV_SCRIPT}" >&2
  exit 1
fi
if [[ ! -x "${GENERATION_ENV}/bin/torchrun" || ! -x "${GENERATION_ENV}/bin/python" ]]; then
  echo "[error] LongLive environment is incomplete: ${GENERATION_ENV}" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "${CANN_ENV_SCRIPT}"
set -u
export CONDA_PREFIX="${GENERATION_ENV}"
export PATH="${GENERATION_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${GENERATION_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  visible_devices=""
  for ((device_index = 0; device_index < NPROC_PER_NODE; device_index++)); do
    visible_devices+="${visible_devices:+,}${device_index}"
  done
  export ASCEND_RT_VISIBLE_DEVICES="${visible_devices}"
fi

sp_size="${SP_SIZE}"
if [[ ! "${sp_size}" =~ ^[1-9][0-9]*$ || ! "${DP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] SP_SIZE and DP_SIZE must be positive integers: sp=${sp_size}, dp=${DP_SIZE}" >&2
  exit 1
fi
if [[ $((sp_size * DP_SIZE)) -ne NPROC_PER_NODE ]]; then
  echo "[error] parallel layout mismatch: sp_size=${sp_size:-missing}, " \
       "dp_size=${DP_SIZE}, nproc=${NPROC_PER_NODE}" >&2
  exit 1
fi

prompt_file="${PROMPTS_FILE:-data/benchmarks/performance/prompts.txt}"
if [[ ! -f "${prompt_file}" ]]; then
  echo "[error] performance prompt file not found: ${prompt_file}" >&2
  exit 1
fi
expected_videos="$(awk 'NF {count++} END {print count+0}' "${prompt_file}")"
if [[ "${expected_videos}" -eq 0 ]]; then
  echo "[error] performance prompt file is empty: ${prompt_file}" >&2
  exit 1
fi

run_id="$(date +%Y%m%d_%H%M%S)_$(basename "${CONFIG_PATH}" .yaml)_sp${sp_size}_dp${DP_SIZE}"
run_dir="logs/npu_benchmark/${run_id}"
video_dir="videos/benchmarks/perf_runs/${run_id}"
raw_log="${run_dir}/torchrun.log"
summary_file="${run_dir}/summary.txt"
rendered_config="$(mktemp "${TMPDIR:-/tmp}/longlive_perf.XXXXXX.yaml")"
mkdir -p "${run_dir}" "${video_dir}"

sed_args=(
  -e "s/^sp_size: .*/sp_size: ${sp_size}/"
  -e "s/^dp_size: .*/dp_size: ${DP_SIZE}/"
  -e "s|^output_folder: .*|output_folder: ${video_dir}|"
)
if [[ -n "${VAE_DEVICE:-}" ]]; then
  sed_args+=(-e "s|^  vae_device: .*|  vae_device: ${VAE_DEVICE}|")
fi
sed "${sed_args[@]}" "${CONFIG_PATH}" > "${rendered_config}"

child_pid=""
cleanup() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" 2>/dev/null || true
  fi
  rm -f "${rendered_config}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

draw_progress() {
  local completed="$1"
  local total="$2"
  local width=36
  local filled=$((completed * width / total))
  local empty=$((width - filled))
  local done_bar pending_bar
  printf -v done_bar '%*s' "${filled}" ''
  printf -v pending_bar '%*s' "${empty}" ''
  done_bar="${done_bar// /#}"
  pending_bar="${pending_bar// /-}"
  printf '\r[generate] [%s%s] %d/%d' "${done_bar}" "${pending_bar}" "${completed}" "${total}"
}

echo "[run] config=${CONFIG_PATH} nproc=${NPROC_PER_NODE} sp=${sp_size} dp=${DP_SIZE}"
echo "[run] generation_env=${GENERATION_ENV}"
if [[ -n "${VAE_DEVICE:-}" ]]; then
  echo "[run] dedicated VAE device=${VAE_DEVICE}"
fi
echo "[run] full log: ${raw_log}"

LLV2_DEVICE=npu "${GENERATION_ENV}/bin/torchrun" \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  inference_sp.py \
  --config_path "${rendered_config}" \
  >"${raw_log}" 2>&1 &
child_pid="$!"

while kill -0 "${child_pid}" 2>/dev/null; do
  completed="$(find "${video_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
  if [[ "${completed}" -gt "${expected_videos}" ]]; then
    completed="${expected_videos}"
  fi
  draw_progress "${completed}" "${expected_videos}"
  sleep 2
done

set +e
wait "${child_pid}"
status="$?"
set -e
child_pid=""
completed="$(find "${video_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
draw_progress "${completed}" "${expected_videos}"
printf '\n'

if [[ "${status}" -ne 0 ]]; then
  echo "[error] generation failed with exit code ${status}; log tail:" >&2
  tail -n 80 "${raw_log}" >&2
  exit "${status}"
fi

"${GENERATION_ENV}/bin/python" scripts/summarize_npu_benchmark.py \
  "${raw_log}" \
  --warmup-per-rank "${WARMUP_PER_RANK}" \
  | tee "${summary_file}"

echo "[done] videos: ${video_dir}"
echo "[done] summary: ${summary_file}"
