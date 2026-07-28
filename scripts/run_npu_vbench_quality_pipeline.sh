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
GENERATION_ENV="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}"
AISBENCH_ENV="${AISBENCH_ENV:-/mnt/share/r50063443/conda_envs/aisbench_npu}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/mnt/weight/vbench_models/}"

case "${BENCHMARK}" in
  standard)
    RUN_TAG="standard_20pct"
    CONFIG_PATH="${CONFIG_PATH:-configs/benchmarks/vbench_standard_20pct_5s_npu_bf16.yaml}"
    GENERATION_PROMPTS="${GENERATION_PROMPTS:-data/benchmarks/vbench_standard_20pct/prompts.txt}"
    NAMING_PROMPTS="${NAMING_PROMPTS:-${GENERATION_PROMPTS}}"
    FULL_INFO="${FULL_INFO:-data/benchmarks/vbench_standard_20pct/VBench_full_info.json}"
    ;;
  augmented)
    RUN_TAG="augmented_20pct"
    CONFIG_PATH="${CONFIG_PATH:-configs/benchmarks/vbench_standard_20pct_augmented_5s_npu_bf16.yaml}"
    GENERATION_PROMPTS="${GENERATION_PROMPTS:-data/benchmarks/vbench_standard_20pct_augmented_wan21_qwen25_seed42/prompts.txt}"
    NAMING_PROMPTS="${NAMING_PROMPTS:-data/benchmarks/vbench_standard_20pct_augmented_wan21_qwen25_seed42/original_prompts.txt}"
    FULL_INFO="${FULL_INFO:-data/benchmarks/vbench_standard_20pct/VBench_full_info.json}"
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
if [[ ! -x "${GENERATION_ENV}/bin/torchrun" || ! -x "${GENERATION_ENV}/bin/python" ]]; then
  echo "[error] LongLive generation environment is incomplete: ${GENERATION_ENV}" >&2
  echo "        Expected executable bin/torchrun and bin/python." >&2
  echo "        Override GENERATION_ENV with the deployed LongLive Conda environment." >&2
  exit 1
fi
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

# Wan2.2-TI2V-5B uses a (1, 2, 2) DiT patch. Odd latent H/W values are
# truncated by patch embedding and cannot be subtracted from the original
# diffusion state. Reject them before loading the model.
shape_values="$(sed -n 's/.*image_or_video_shape: *\[\([^]]*\)\].*/\1/p' "${CONFIG_PATH}" | head -n 1)"
if [[ -z "${shape_values}" ]]; then
  echo "[error] image_or_video_shape must use compact [B, T, C, H, W] syntax: ${CONFIG_PATH}" >&2
  exit 1
fi
IFS=',' read -r _shape_b _shape_t _shape_c latent_h latent_w <<< "${shape_values}"
latent_h="${latent_h//[[:space:]]/}"
latent_w="${latent_w//[[:space:]]/}"
if [[ ! "${latent_h}" =~ ^[0-9]+$ || ! "${latent_w}" =~ ^[0-9]+$ ]]; then
  echo "[error] invalid latent H/W in image_or_video_shape: H=${latent_h:-missing}, W=${latent_w:-missing}" >&2
  exit 1
fi
if [[ $((latent_h % 2)) -ne 0 || $((latent_w % 2)) -ne 0 ]]; then
  echo "[error] Wan DiT latent H/W must both be divisible by 2; got H=${latent_h:-missing}, W=${latent_w:-missing}" >&2
  echo "        Odd sizes are truncated by patch embedding (for example, H=45 becomes 44)." >&2
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

run_id="$(date +%Y%m%d_%H%M%S)_vbench_${RUN_TAG}_5seed_sp${sp_size}_dp${DP_SIZE}"
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
  local completed="$1" total="$2" global_completed="$3" global_total="$4"
  local elapsed_seconds="$5" label="$6" width=36
  local filled empty
  filled=$((completed * width / total))
  empty=$((width - filled))
  local done_bar pending_bar elapsed_text eta_text rate_text
  printf -v done_bar '%*s' "${filled}" ''
  printf -v pending_bar '%*s' "${empty}" ''
  printf -v elapsed_text '%02d:%02d:%02d' \
    "$((elapsed_seconds / 3600))" "$(((elapsed_seconds % 3600) / 60))" "$((elapsed_seconds % 60))"
  if ((global_completed > 0)); then
    local eta_seconds=$((elapsed_seconds * (global_total - global_completed) / global_completed))
    printf -v eta_text '%02d:%02d:%02d' \
      "$((eta_seconds / 3600))" "$(((eta_seconds % 3600) / 60))" "$((eta_seconds % 60))"
    rate_text="$(awk -v elapsed="${elapsed_seconds}" -v count="${global_completed}" \
      'BEGIN { if (count > 0) printf "%.1fs/video", elapsed / count; else print "--" }')"
  else
    eta_text="--:--:--"
    rate_text="--"
  fi
  printf '\r[%s] [%s%s] %d/%d | total %d/%d [%s<%s, %s]' \
    "${label}" "${done_bar// /#}" "${pending_bar// /-}" "${completed}" "${total}" \
    "${global_completed}" "${global_total}" "${elapsed_text}" "${eta_text}" "${rate_text}"
}

echo "[run] benchmark=${BENCHMARK}, prompts=${prompt_count}, seeds=${SEEDS}"
echo "[run] config=${CONFIG_PATH}, nproc=${NPROC_PER_NODE}, sp=${sp_size}, dp=${DP_SIZE}"
echo "[run] generation_env=${GENERATION_ENV}"
echo "[run] logs=${run_dir}"

pipeline_started_at="$(date +%s)"
total_videos=$((prompt_count * ${#seed_array[@]}))
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

  env \
    PATH="${GENERATION_ENV}/bin:${PATH}" \
    CONDA_PREFIX="${GENERATION_ENV}" \
    LD_LIBRARY_PATH="${GENERATION_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    LLV2_DEVICE=npu \
    "${GENERATION_ENV}/bin/torchrun" \
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
    global_completed=$((sample_index * prompt_count + completed))
    elapsed_seconds=$(($(date +%s) - pipeline_started_at))
    draw_progress "${completed}" "${prompt_count}" "${global_completed}" "${total_videos}" \
      "${elapsed_seconds}" "seed ${seed} ($((sample_index + 1))/${#seed_array[@]})"
    sleep 2
  done

  set +e
  wait "${child_pid}"
  status="$?"
  set -e
  child_pid=""
  completed="$(find "${seed_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
  ((completed > prompt_count)) && completed="${prompt_count}"
  global_completed=$((sample_index * prompt_count + completed))
  elapsed_seconds=$(($(date +%s) - pipeline_started_at))
  draw_progress "${completed}" "${prompt_count}" "${global_completed}" "${total_videos}" \
    "${elapsed_seconds}" "seed ${seed} ($((sample_index + 1))/${#seed_array[@]})"
  printf '\n'
  if [[ "${status}" -ne 0 ]]; then
    echo "[error] seed ${seed} failed with exit code ${status}; log tail:" >&2
    tail -n 80 "${raw_log}" >&2
    exit "${status}"
  fi

  "${GENERATION_ENV}/bin/python" third_party/aisbench_adapter/prepare_vbench_videos.py \
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
