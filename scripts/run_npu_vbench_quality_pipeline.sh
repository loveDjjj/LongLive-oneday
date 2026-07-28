#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Select standard or augmented. Each run generates five seeds sequentially,
# prepares VBench-compatible names, and starts AISBench quality evaluation.
BENCHMARK="${BENCHMARK:-standard}"
SEEDS="${SEEDS:-0 1 2 3 4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
DP_SIZE="${DP_SIZE:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29530}"
CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
AISBENCH_ENV="${AISBENCH_ENV:-/mnt/share/r50063443/conda_envs/aisbench_npu}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/mnt/weight/vbench_models/}"

case "${BENCHMARK}" in
  standard)
    CONFIG_PATH="${CONFIG_PATH:-configs/benchmarks/vbench_standard_5s_npu_bf16.yaml}"
    GENERATION_PROMPTS="${GENERATION_PROMPTS:-data/benchmarks/vbench_standard/prompts.txt}"
    NAMING_PROMPTS="${NAMING_PROMPTS:-${GENERATION_PROMPTS}}"
    FULL_INFO="${FULL_INFO:-data/benchmarks/vbench_standard/VBench_full_info.json}"
    ;;
  augmented)
    CONFIG_PATH="${CONFIG_PATH:-configs/benchmarks/vbench_standard_augmented_5s_npu_bf16.yaml}"
    GENERATION_PROMPTS="${GENERATION_PROMPTS:-data/benchmarks/vbench_standard_augmented_wan21_qwen25_seed42/prompts.txt}"
    NAMING_PROMPTS="${NAMING_PROMPTS:-data/benchmarks/vbench_standard_augmented_wan21_qwen25_seed42/original_prompts.txt}"
    FULL_INFO="${FULL_INFO:-data/benchmarks/vbench_standard/VBench_full_info.json}"
    ;;
  *)
    echo "[error] BENCHMARK must be standard or augmented, got: ${BENCHMARK}" >&2
    exit 2
    ;;
esac

for required_file in "${CONFIG_PATH}" "${GENERATION_PROMPTS}" "${NAMING_PROMPTS}" "${FULL_INFO}" "${CANN_ENV_SCRIPT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[error] required file not found: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -x "${AISBENCH_ENV}/bin/ais_bench" ]]; then
  echo "[error] AISBench executable not found: ${AISBENCH_ENV}/bin/ais_bench" >&2
  echo "        Override AISBENCH_ENV with the deployed AISBench Conda environment." >&2
  exit 1
fi
if [[ ! -d "${VBENCH_CACHE_DIR}" ]]; then
  echo "[error] VBench model cache not found: ${VBENCH_CACHE_DIR}" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "${CANN_ENV_SCRIPT}"
set -u
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  visible_devices=""
  for ((device_index = 0; device_index < NPROC_PER_NODE; device_index++)); do
    visible_devices+="${visible_devices:+,}${device_index}"
  done
  export ASCEND_RT_VISIBLE_DEVICES="${visible_devices}"
fi

sp_size="$(awk '/^sp_size:/ {print $2; exit}' "${CONFIG_PATH}")"
if [[ -z "${sp_size}" || $((sp_size * DP_SIZE)) -ne NPROC_PER_NODE ]]; then
  echo "[error] parallel layout mismatch: sp=${sp_size:-missing}, dp=${DP_SIZE}, nproc=${NPROC_PER_NODE}" >&2
  exit 1
fi

prompt_count="$(awk 'NF {count++} END {print count+0}' "${GENERATION_PROMPTS}")"
naming_count="$(awk 'NF {count++} END {print count+0}' "${NAMING_PROMPTS}")"
read -r -a seed_array <<< "${SEEDS}"
if [[ "${prompt_count}" -eq 0 || "${prompt_count}" -ne "${naming_count}" ]]; then
  echo "[error] generation/naming prompt counts differ: ${prompt_count}/${naming_count}" >&2
  exit 1
fi
if [[ "${#seed_array[@]}" -ne 5 ]]; then
  echo "[error] formal VBench pipeline requires exactly five sequential seeds; got: ${SEEDS}" >&2
  exit 1
fi

run_id="$(date +%Y%m%d_%H%M%S)_vbench_${BENCHMARK}_5seed_sp${sp_size}_dp${DP_SIZE}"
run_dir="logs/npu_quality/${run_id}"
raw_root="videos/benchmarks/quality_runs/${run_id}/raw"
prepared_dir="videos/benchmarks/quality_runs/${run_id}/vbench"
mkdir -p "${run_dir}" "${raw_root}" "${prepared_dir}"

child_pid=""
rendered_config=""
cleanup() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" 2>/dev/null || true
  fi
  [[ -z "${rendered_config}" ]] || rm -f "${rendered_config}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

draw_progress() {
  local completed="$1" total="$2" label="$3" width=36
  local filled empty
  filled=$((completed * width / total))
  empty=$((width - filled))
  local done_bar pending_bar
  printf -v done_bar '%*s' "${filled}" ''
  printf -v pending_bar '%*s' "${empty}" ''
  printf '\r[%s] [%s%s] %d/%d' "${label}" "${done_bar// /#}" "${pending_bar// /-}" "${completed}" "${total}"
}

echo "[run] benchmark=${BENCHMARK}, prompts=${prompt_count}, seeds=${SEEDS}"
echo "[run] config=${CONFIG_PATH}, nproc=${NPROC_PER_NODE}, sp=${sp_size}, dp=${DP_SIZE}"
echo "[run] logs=${run_dir}"

for sample_index in "${!seed_array[@]}"; do
  seed="${seed_array[${sample_index}]}"
  seed_dir="${raw_root}/seed_${seed}"
  raw_log="${run_dir}/seed_${seed}.log"
  rendered_config="$(mktemp "${TMPDIR:-/tmp}/longlive_vbench.XXXXXX.yaml")"
  mkdir -p "${seed_dir}"

  sed \
    -e "s/^dp_size: .*/dp_size: ${DP_SIZE}/" \
    -e "s|^output_folder: .*|output_folder: ${seed_dir}|" \
    -e "s|^  data_path: .*|  data_path: ${GENERATION_PROMPTS}|" \
    -e "s/^  seed: .*/  seed: ${seed}/" \
    "${CONFIG_PATH}" > "${rendered_config}"

  LLV2_DEVICE=npu torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="$((MASTER_PORT + sample_index))" \
    inference_sp.py \
    --config_path "${rendered_config}" \
    >"${raw_log}" 2>&1 &
  child_pid="$!"

  while kill -0 "${child_pid}" 2>/dev/null; do
    completed="$(find "${seed_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
    ((completed > prompt_count)) && completed="${prompt_count}"
    draw_progress "${completed}" "${prompt_count}" "seed ${seed}"
    sleep 2
  done

  set +e
  wait "${child_pid}"
  status="$?"
  set -e
  child_pid=""
  completed="$(find "${seed_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
  draw_progress "${completed}" "${prompt_count}" "seed ${seed}"
  printf '\n'
  if [[ "${status}" -ne 0 ]]; then
    echo "[error] seed ${seed} failed with exit code ${status}; log tail:" >&2
    tail -n 80 "${raw_log}" >&2
    exit "${status}"
  fi

  python third_party/aisbench_adapter/prepare_vbench_videos.py \
    --benchmark standard \
    --src-dir "${seed_dir}" \
    --prompts-file "${NAMING_PROMPTS}" \
    --dst-dir "${prepared_dir}" \
    --sample-index "${sample_index}"
  rm -f "${rendered_config}"
  rendered_config=""
done

echo "[evaluate] AISBench VBench 1.0, videos=${prepared_dir}"
env \
  PATH="${AISBENCH_ENV}/bin:${PATH}" \
  CONDA_PREFIX="${AISBENCH_ENV}" \
  LONGLIVE_VBENCH_DATA_PATH="${REPO_ROOT}/${prepared_dir}" \
  LONGLIVE_VBENCH_FULL_INFO="${REPO_ROOT}/${FULL_INFO}" \
  VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR}" \
  AISBENCH_MAX_WORKERS="${AISBENCH_MAX_WORKERS:-16}" \
  bash third_party/aisbench_adapter/run_vbench_16npu.sh \
  2>&1 | tee "${run_dir}/aisbench.log"

echo "[done] prepared videos: ${prepared_dir}"
echo "[done] generation/AISBench logs: ${run_dir}"
