#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# 部署参数。模型结构、数据、帧数和随机种子由 YAML 管理。
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export GENERATION_ENV="${GENERATION_ENV:-/mnt/a800_share/r50063443/conda_envs/longlive}"
export AISBENCH_ENV="${AISBENCH_ENV:-/mnt/a800_share/r50063443/conda_envs/aisbench_npu}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/mnt/a800_share/r50063443/vbench_models}"
export CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/mnt/a800_share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh}"
unset MASTER_PORT

CONFIG_PATH="${CONFIG_PATH:-configs/inference/vbench.yaml}"
PRESET="${1:-longlive2_standard_20pct}"

for required_path in "${CONFIG_PATH}" "${CANN_ENV_SCRIPT}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "[error] required file not found: ${required_path}" >&2
    exit 1
  fi
done
if [[ ! -x "${GENERATION_ENV}/bin/python" || ! -x "${GENERATION_ENV}/bin/torchrun" ]]; then
  echo "[error] incomplete generation environment: ${GENERATION_ENV}" >&2
  exit 1
fi
if [[ ! -x "${AISBENCH_ENV}/bin/ais_bench" ]]; then
  echo "[error] AISBench environment is incomplete: ${AISBENCH_ENV}" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "${CANN_ENV_SCRIPT}"
set -u
export CONDA_PREFIX="${GENERATION_ENV}"
export PATH="${GENERATION_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${GENERATION_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

metadata_tmp="$(mktemp "${TMPDIR:-/tmp}/longlive_vbench.XXXXXX.json")"
child_pid=""
cleanup() {
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" 2>/dev/null || true
  fi
  rm -f "${metadata_tmp}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
"${GENERATION_ENV}/bin/python" scripts/evaluation/resolve_config.py vbench \
  --config "${CONFIG_PATH}" --preset "${PRESET}" >"${metadata_tmp}"

json_field() {
  "${GENERATION_ENV}/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1]))[sys.argv[2]]; print(value)' \
    "${metadata_tmp}" "$1"
}
run_tag="$(json_field run_tag)"
sparsity_method="$(json_field sparsity_method)"
sparsity_backend="$(json_field sparsity_backend)"
entrypoint="$(json_field entrypoint)"
sp_size="$(json_field sp_size)"
dp_size="$(json_field dp_size)"
nproc="$(json_field nproc_per_node)"
required_devices="$(json_field required_devices)"
prompt_count="$(json_field prompt_count)"
naming_prompts="$(json_field naming_prompts)"
full_info="$(json_field full_info)"
seeds="$("${GENERATION_ENV}/bin/python" -c \
  'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["seeds"])))' "${metadata_tmp}")"

visible_count="$(awk -F, '{print NF}' <<<"${ASCEND_RT_VISIBLE_DEVICES}")"
if [[ "${visible_count}" -lt "${required_devices}" ]]; then
  echo "[error] preset ${PRESET} requires ${required_devices} worker NPUs, " \
       "got ${ASCEND_RT_VISIBLE_DEVICES}" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="${RUN_ID:-${run_tag}-${timestamp}}"
if [[ "${run_id}" == */* ]]; then
  echo "[error] RUN_ID must be a directory name: ${run_id}" >&2
  exit 1
fi
run_dir="runs/vbench/${run_id}"
log_dir="logs/vbench/${run_id}"
raw_root="${run_dir}/videos/raw"
prepared_dir="${run_dir}/videos/prepared"
mkdir -p "${raw_root}" "${prepared_dir}" "${log_dir}"
cp "${metadata_tmp}" "${run_dir}/manifest.json"

count_videos() {
  local directory="$1"
  if [[ ! -d "${directory}" ]]; then
    echo 0
    return
  fi
  find "${directory}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' '
}

format_duration() {
  local seconds="$1"
  printf '%02d:%02d:%02d' \
    "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

draw_progress() {
  local completed="$1" total="$2" elapsed_seconds="$3" label="$4"
  local rate_unit="$5" eta_completed="${6:-${completed}}" width=36
  local filled=$((completed * width / total))
  local empty=$((width - filled))
  local done_bar pending_bar elapsed_text eta_text rate_text
  printf -v done_bar '%*s' "${filled}" ''
  printf -v pending_bar '%*s' "${empty}" ''
  elapsed_text="$(format_duration "${elapsed_seconds}")"
  if ((eta_completed > 0)); then
    local eta_seconds=$((elapsed_seconds * (total - completed) / eta_completed))
    eta_text="$(format_duration "${eta_seconds}")"
    rate_text="$(awk -v elapsed="${elapsed_seconds}" -v count="${eta_completed}" \
      -v unit="${rate_unit}" 'BEGIN { printf "%.1fs/%s", elapsed / count, unit }')"
  else
    eta_text="--:--:--"
    rate_text="--"
  fi
  printf '\r[%s] [%s%s] %d/%d [%s<%s, %s]' \
    "${label}" "${done_bar// /#}" "${pending_bar// /-}" \
    "${completed}" "${total}" "${elapsed_text}" "${eta_text}" "${rate_text}"
}

count_completed_dimensions() {
  local directory="$1"
  if [[ ! -d "${directory}" ]]; then
    echo 0
    return
  fi
  find "${directory}" -type f \
    -path '*/results/vbench_eval/vbench_*.json' | wc -l | tr -d ' '
}

echo "[run] task=vbench preset=${PRESET} run_id=${run_id}"
echo "[run] devices=${ASCEND_RT_VISIBLE_DEVICES} layout=SP${sp_size}xDP${dp_size} prompts=${prompt_count}"
echo "[run] sparsity=${sparsity_method} backend=${sparsity_backend}"
read -r -a seed_array <<<"${seeds}"
generation_started_at="$(date +%s)"
total_videos=$((prompt_count * ${#seed_array[@]}))
new_videos_completed=0

for sample_index in "${!seed_array[@]}"; do
  seed="${seed_array[${sample_index}]}"
  seed_dir="${raw_root}/seed_${seed}"
  raw_log="${log_dir}/seed_${seed}.log"
  resolved_config="${run_dir}/resolved_seed_${seed}.yaml"
  mkdir -p "${seed_dir}"
  completed="$(count_videos "${seed_dir}")"
  ((completed > prompt_count)) && completed="${prompt_count}"
  seed_initial_completed="${completed}"
  if [[ "${completed}" -eq "${prompt_count}" ]]; then
    echo "[resume] seed=${seed} already complete (${completed}/${prompt_count})"
    generation_status=0
  else
    "${GENERATION_ENV}/bin/python" scripts/evaluation/resolve_config.py vbench \
      --config "${CONFIG_PATH}" --preset "${PRESET}" --seed "${seed}" \
      --output "${resolved_config}" --output-folder "${seed_dir}" >/dev/null
    echo "[generate] seed=${seed} ($((sample_index + 1))/${#seed_array[@]}) rendezvous=standalone"
    LLV2_DEVICE=npu "${GENERATION_ENV}/bin/torchrun" \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="${nproc}" \
      "${entrypoint}" \
      --config_path "${resolved_config}" \
      >"${raw_log}" 2>&1 &
    child_pid="$!"

    while kill -0 "${child_pid}" 2>/dev/null; do
      completed="$(count_videos "${seed_dir}")"
      ((completed > prompt_count)) && completed="${prompt_count}"
      global_completed=$((sample_index * prompt_count + completed))
      current_seed_new=$((completed - seed_initial_completed))
      ((current_seed_new < 0)) && current_seed_new=0
      new_completed=$((new_videos_completed + current_seed_new))
      elapsed_seconds=$(($(date +%s) - generation_started_at))
      draw_progress "${global_completed}" "${total_videos}" "${elapsed_seconds}" \
        "generate seed ${seed} ($((sample_index + 1))/${#seed_array[@]}) ${completed}/${prompt_count}" \
        "video" "${new_completed}"
      sleep 2
    done

    set +e
    wait "${child_pid}"
    generation_status="$?"
    set -e
    child_pid=""
  fi

  completed="$(count_videos "${seed_dir}")"
  ((completed > prompt_count)) && completed="${prompt_count}"
  global_completed=$((sample_index * prompt_count + completed))
  current_seed_new=$((completed - seed_initial_completed))
  ((current_seed_new < 0)) && current_seed_new=0
  new_completed=$((new_videos_completed + current_seed_new))
  elapsed_seconds=$(($(date +%s) - generation_started_at))
  draw_progress "${global_completed}" "${total_videos}" "${elapsed_seconds}" \
    "generate seed ${seed} ($((sample_index + 1))/${#seed_array[@]}) ${completed}/${prompt_count}" \
    "video" "${new_completed}"
  printf '\n'
  if [[ "${generation_status}" -ne 0 ]]; then
    echo "[error] seed ${seed} failed with exit code ${generation_status}; log tail:" >&2
    tail -n 100 "${raw_log}" >&2
    exit "${generation_status}"
  fi
  new_videos_completed="${new_completed}"

  "${GENERATION_ENV}/bin/python" third_party/aisbench_adapter/prepare_vbench_videos.py \
    --benchmark standard \
    --src-dir "${seed_dir}" \
    --prompts-file "${naming_prompts}" \
    --dst-dir "${prepared_dir}" \
    --sample-index "${sample_index}" \
    --overwrite
done

echo "[evaluate] videos=${prepared_dir}"
aisbench_log="${log_dir}/aisbench.log"
eval_session="$(date +%Y%m%d_%H%M%S)"
aisbench_work_dir="${run_dir}/aisbench/${eval_session}"
vbench_dimension_count=16
mkdir -p "${aisbench_work_dir}"
evaluation_started_at="$(date +%s)"
env \
  PATH="${AISBENCH_ENV}/bin:${PATH}" \
  CONDA_PREFIX="${AISBENCH_ENV}" \
  LD_LIBRARY_PATH="${AISBENCH_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  LONGLIVE_VBENCH_DATA_PATH="${REPO_ROOT}/${prepared_dir}" \
  LONGLIVE_VBENCH_FULL_INFO="${REPO_ROOT}/${full_info}" \
  VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR}" \
  AISBENCH_MAX_WORKERS="${nproc}" \
  ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
  CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT}" \
  AISBENCH_WORK_DIR="${REPO_ROOT}/${aisbench_work_dir}" \
  bash third_party/aisbench_adapter/run_vbench_eval.sh \
  >"${aisbench_log}" 2>&1 &
child_pid="$!"

while kill -0 "${child_pid}" 2>/dev/null; do
  completed_dimensions="$(count_completed_dimensions "${aisbench_work_dir}")"
  ((completed_dimensions > vbench_dimension_count)) && completed_dimensions="${vbench_dimension_count}"
  elapsed_seconds=$(($(date +%s) - evaluation_started_at))
  draw_progress "${completed_dimensions}" "${vbench_dimension_count}" \
    "${elapsed_seconds}" "evaluate dimensions" "dimension"
  sleep 2
done

set +e
wait "${child_pid}"
evaluation_status="$?"
set -e
child_pid=""
completed_dimensions="$(count_completed_dimensions "${aisbench_work_dir}")"
((completed_dimensions > vbench_dimension_count)) && completed_dimensions="${vbench_dimension_count}"
elapsed_seconds=$(($(date +%s) - evaluation_started_at))
draw_progress "${completed_dimensions}" "${vbench_dimension_count}" \
  "${elapsed_seconds}" "evaluate dimensions" "dimension"
printf '\n'

if [[ "${evaluation_status}" -ne 0 ]]; then
  echo "[error] AISBench failed with exit code ${evaluation_status}; log tail:" >&2
  tail -n 100 "${aisbench_log}" >&2
  exit "${evaluation_status}"
fi

if grep -q '\[RUNNER-TASK-001\]' "${aisbench_log}"; then
  echo "[error] one or more AISBench tasks failed; inspect ${aisbench_log}" >&2
  exit 1
fi
if [[ "${completed_dimensions}" -ne "${vbench_dimension_count}" ]]; then
  echo "[error] AISBench completed only ${completed_dimensions}/${vbench_dimension_count} dimensions; inspect ${aisbench_log}" >&2
  exit 1
fi
"${GENERATION_ENV}/bin/python" scripts/evaluation/summarize_vbench.py \
  --work-dir "${aisbench_work_dir}" \
  --output "${run_dir}/vbench_results.json" \
  --copy-summary-to "${run_dir}"
echo "[done] run=${run_dir}"
