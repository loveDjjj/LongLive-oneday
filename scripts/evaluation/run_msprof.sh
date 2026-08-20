#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# 部署参数。SP、DP 和专用 VAE 设备由 YAML 管理。
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4}"
export GENERATION_ENV="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}"
export CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh}"
unset MASTER_PORT

CONFIG_PATH="${CONFIG_PATH:-configs/inference/msprof.yaml}"
PRESET="${1:-32s}"
MSPROF_MODE="${MSPROF_MODE:-async_vae}"

if [[ ! "${MSPROF_MODE}" =~ ^(dit_only|sync_vae|async_vae)$ ]]; then
  echo "[error] MSPROF_MODE must be dit_only, sync_vae, or async_vae" >&2
  exit 1
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

for command_name in msprof msprof-analyze; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "[error] command not found: ${command_name}" >&2
    exit 1
  fi
done

metadata_tmp="$(mktemp "${TMPDIR:-/tmp}/longlive_msprof.XXXXXX.json")"
trap 'rm -f "${metadata_tmp}"' EXIT
"${GENERATION_ENV}/bin/python" scripts/evaluation/resolve_config.py msprof \
  --config "${CONFIG_PATH}" --preset "${PRESET}" \
  --vae-mode "${MSPROF_MODE}" >"${metadata_tmp}"

json_field() {
  "${GENERATION_ENV}/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1]))[sys.argv[2]]; print(value)' \
    "${metadata_tmp}" "$1"
}

run_tag="$(json_field run_tag)"
sparsity_method="$(json_field sparsity_method)"
sparsity_backend="$(json_field sparsity_backend)"
sp_size="$(json_field sp_size)"
dp_size="$(json_field dp_size)"
nproc="$(json_field nproc_per_node)"
required_devices="$(json_field required_devices)"
warmup_per_rank="$(json_field warmup_per_rank)"

# JSON 字典不适合直接传给 shell，使用结构化查询读取 profiler 字段。
json_nested() {
  "${GENERATION_ENV}/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]]; print(str(value).lower() if isinstance(value, bool) else value)' \
    "${metadata_tmp}" "$1" "$2"
}
output_type="$(json_nested msprof output_type)"
storage_limit="$(json_nested msprof storage_limit)"
task_time="$(json_nested msprof task_time)"
ai_core="${MSPROF_AI_CORE:-$(json_nested msprof ai_core)}"
aic_mode="$(json_nested msprof aic_mode)"
aic_metrics="$(json_nested msprof aic_metrics)"
recover_parse="$(json_nested msprof recover_parse)"
if [[ "${ai_core}" != "true" && "${ai_core}" != "false" ]]; then
  echo "[error] MSPROF_AI_CORE/msprof.ai_core must be true or false, got ${ai_core}" >&2
  exit 1
fi

visible_count="$(awk -F, '{print NF}' <<<"${ASCEND_RT_VISIBLE_DEVICES}")"
if [[ "${visible_count}" -ne "${required_devices}" ]]; then
  echo "[error] preset ${PRESET} requires ${required_devices} visible NPUs " \
       "(${nproc} workers plus dedicated VAE when enabled), got ${ASCEND_RT_VISIBLE_DEVICES}" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="${RUN_ID:-${run_tag}-${timestamp}}"
if [[ "${run_id}" == */* ]]; then
  echo "[error] RUN_ID must be a directory name: ${run_id}" >&2
  exit 1
fi
run_dir="runs/msprof/dit/${run_id}"
log_dir="logs/msprof/dit/${run_id}"
video_dir="${run_dir}/videos"
profile_dir="${run_dir}/profiling/raw"
analysis_dir="${run_dir}/profiling/analysis"
raw_log="${log_dir}/msprof.log"
summary_file="${run_dir}/summary.txt"
summary_json="${run_dir}/summary.json"
resolved_config="${run_dir}/resolved.yaml"
mkdir -p "${video_dir}" "${profile_dir}" "${analysis_dir}" "${log_dir}"
cp "${metadata_tmp}" "${run_dir}/manifest.json"

"${GENERATION_ENV}/bin/python" scripts/evaluation/resolve_config.py msprof \
  --config "${CONFIG_PATH}" --preset "${PRESET}" \
  --vae-mode "${MSPROF_MODE}" \
  --output "${resolved_config}" --output-folder "${video_dir}" >/dev/null

echo "[run] task=msprof preset=${PRESET} run_id=${run_id}"
echo "[run] devices=${ASCEND_RT_VISIBLE_DEVICES} layout=SP${sp_size}xDP${dp_size}"
echo "[run] sparsity=${sparsity_method} backend=${sparsity_backend}"
echo "[run] mode=${MSPROF_MODE}"
echo "[run] rendezvous=standalone"
echo "[run] config=${resolved_config} profile=${profile_dir}"
echo "[run] msprof_ai_core=${ai_core} task_time=${task_time}"

profiler_args=(
  --output="${profile_dir}"
  --type="${output_type}"
  --storage-limit="${storage_limit}"
  --ascendcl=on
  --model-execution=on
  --runtime-api=on
  --task-time="${task_time}"
  --aicpu=on
  --sys-hardware-mem=on
  --hccl=on
)
if [[ "${ai_core}" == "true" ]]; then
  profiler_args+=(
    --ai-core=on
    --aic-mode="${aic_mode}"
    --aic-metrics="${aic_metrics}"
  )
fi

set +e
LLV2_DEVICE=npu msprof "${profiler_args[@]}" \
  "${GENERATION_ENV}/bin/torchrun" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${nproc}" \
    inference_sp.py \
    --config_path "${resolved_config}" \
  2>&1 | tee "${raw_log}"
profile_status="${PIPESTATUS[0]}"
set -e

if [[ "${profile_status}" -ne 0 ]] && \
   grep -Eq "DrvFftsProfileStart failed|ADD_TO_LAUNCHER_LIST_AICORE failed|error code is 561103" "${raw_log}"; then
  echo "[error] msprof AICore/FFTS initialization failed before inference started" >&2
  echo "[hint] retry once with MSPROF_AI_CORE=false and a new RUN_ID to isolate workload errors from PMU errors" >&2
  echo "[hint] a PMU-disabled profile is only a task/HCCL baseline and cannot support AICore pipeline conclusions" >&2
  exit "${profile_status}"
fi

mapfile -t prof_dirs < <(find "${profile_dir}" -type d -name 'PROF_*' | sort)
if [[ "${#prof_dirs[@]}" -eq 0 ]]; then
  echo "[error] no PROF_* data found under ${profile_dir}" >&2
  if [[ "${profile_status}" -ne 0 ]]; then
    exit "${profile_status}"
  fi
  exit 1
fi

if [[ "${recover_parse}" == "true" || \
      ("${recover_parse}" == "auto" && "${profile_status}" -ne 0) ]]; then
  recovery_log="${log_dir}/recovery.log"
  for prof_dir in "${prof_dirs[@]}"; do
    msprof --export=on --output="${prof_dir}" 2>&1 | tee -a "${recovery_log}" || true
    msprof --analyze=on --output="${prof_dir}" 2>&1 | tee -a "${recovery_log}" || true
  done
fi

"${GENERATION_ENV}/bin/python" scripts/evaluation/summarize_benchmark.py \
  "${raw_log}" --warmup-per-rank "${warmup_per_rank}" \
  --json-output "${summary_json}" | tee "${summary_file}"

analyze_input="${profile_dir}"
if [[ "${#prof_dirs[@]}" -eq 1 ]]; then
  analyze_input="${prof_dirs[0]}"
fi
run_analysis() {
  local name="$1"
  shift
  echo "[analyze] ${name}"
  msprof-analyze "$@" || echo "[warning] analysis failed: ${name}" >&2
}
run_analysis all -m all -d "${analyze_input}" -o "${analysis_dir}/all" --export_type text --parallel_mode concurrent
run_analysis compute_op_sum -m compute_op_sum -d "${analyze_input}" -o "${analysis_dir}/compute_op_sum" --export_type text --parallel_mode concurrent
run_analysis hccl_sum -m hccl_sum -d "${analyze_input}" -o "${analysis_dir}/hccl_sum" --export_type text --top_num 30
run_analysis communication_time_sum -m communication_time_sum -d "${analyze_input}" -o "${analysis_dir}/communication_time_sum" --export_type text
run_analysis communication_matrix_sum -m communication_matrix_sum -d "${analyze_input}" -o "${analysis_dir}/communication_matrix_sum" --export_type text
run_analysis slow_rank -m slow_rank -d "${analyze_input}" -o "${analysis_dir}/slow_rank" --export_type text
run_analysis free_analysis -m free_analysis -d "${analyze_input}" -o "${analysis_dir}/free_analysis" --export_type text
run_analysis advisor advisor all -d "${analyze_input}" -o "${analysis_dir}/advisor"

echo "[done] run=${run_dir}"
echo "[note] latency and FPS in this run include msprof collection overhead"
if [[ "${profile_status}" -ne 0 ]]; then
  echo "[warning] msprof returned ${profile_status}; recovered data may be incomplete" >&2
  exit "${profile_status}"
fi
