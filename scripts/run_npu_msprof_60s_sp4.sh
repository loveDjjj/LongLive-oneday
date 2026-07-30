#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- Runtime settings. Edit here or override with environment variables. ----
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-12,13,14,15}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export SP_SIZE="${SP_SIZE:-4}"
export DP_SIZE="${DP_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29820}"
export CONDA_ENV="${CONDA_ENV:-/mnt/share/r50063443/conda_envs/longlive}"
export CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
export CONFIG_PATH="${CONFIG_PATH:-configs/benchmarks/msprof_longlive_60s_sp4_dp1_npu_bf16.yaml}"

# Some CANN releases accept "on", while releases exposing l0/l1 require "l1".
export MSPROF_TASK_TIME="${MSPROF_TASK_TIME:-on}"
export MSPROF_AIC_METRICS="${MSPROF_AIC_METRICS:-PipeUtilization}"
export MSPROF_STORAGE_LIMIT="${MSPROF_STORAGE_LIMIT:-50000MB}"
# msprof-analyze reads the profiling database directly. Avoid the much larger
# automatic Timeline/CSV export for long multi-process video generation.
export MSPROF_OUTPUT_TYPE="${MSPROF_OUTPUT_TYPE:-db}"
export MSPROF_RECOVER_PARSE="${MSPROF_RECOVER_PARSE:-auto}"

if [[ "${NPROC_PER_NODE}" -ne 4 || "${SP_SIZE}" -ne 4 || "${DP_SIZE}" -ne 1 ]]; then
  echo "[error] this profile is fixed to nproc=4, SP4 x DP1" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[error] config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CANN_ENV_SCRIPT}" ]]; then
  echo "[error] CANN environment script not found: ${CANN_ENV_SCRIPT}" >&2
  exit 1
fi

visible_count="$(awk -F, '{print NF}' <<< "${ASCEND_RT_VISIBLE_DEVICES}")"
if [[ "${visible_count}" -ne "${NPROC_PER_NODE}" ]]; then
  echo "[error] expected 4 visible NPUs, got: ${ASCEND_RT_VISIBLE_DEVICES}" >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[error] conda command not found; activate ${CONDA_ENV} before running" >&2
    exit 1
  fi
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

set +u
# shellcheck disable=SC1090
source "${CANN_ENV_SCRIPT}"
set -u
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

for command_name in msprof msprof-analyze torchrun; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "[error] command not found after environment activation: ${command_name}" >&2
    exit 1
  fi
done

config_tag="$(basename "${CONFIG_PATH}" .yaml)"
run_id="$(date +%Y%m%d_%H%M%S)_${config_tag}_sp4_dp1"
run_dir="logs/msprof/${run_id}"
profile_dir="${run_dir}/profiling"
analysis_dir="${run_dir}/analysis"
video_dir="videos/benchmarks/msprof_runs/${run_id}"
raw_log="${run_dir}/msprof.log"
benchmark_summary="${run_dir}/profiled_benchmark_summary.txt"
rendered_config="${run_dir}/config.yaml"
mkdir -p "${profile_dir}" "${analysis_dir}" "${video_dir}"

sed \
  -e 's/^sp_size: .*/sp_size: 4/' \
  -e 's/^dp_size: .*/dp_size: 1/' \
  -e "s|^output_folder: .*|output_folder: ${video_dir}|" \
  "${CONFIG_PATH}" > "${rendered_config}"

echo "[run] devices=${ASCEND_RT_VISIBLE_DEVICES}, layout=SP4 x DP1"
echo "[run] config=${rendered_config}"
echo "[run] profile output=${profile_dir}"
echo "[run] msprof output can be large; storage limit=${MSPROF_STORAGE_LIMIT}"
echo "[run] msprof output type=${MSPROF_OUTPUT_TYPE}"

set +e
LLV2_DEVICE=npu msprof \
  --output="${profile_dir}" \
  --type="${MSPROF_OUTPUT_TYPE}" \
  --storage-limit="${MSPROF_STORAGE_LIMIT}" \
  --ascendcl=on \
  --model-execution=on \
  --runtime-api=on \
  --task-time="${MSPROF_TASK_TIME}" \
  --ai-core=on \
  --aic-mode=task-based \
  --aic-metrics="${MSPROF_AIC_METRICS}" \
  --aicpu=on \
  --sys-hardware-mem=on \
  --hccl=on \
  torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    inference_sp.py \
    --config_path "${rendered_config}" \
  2>&1 | tee "${raw_log}"
profile_status="${PIPESTATUS[0]}"
set -e

mapfile -t prof_dirs < <(find "${profile_dir}" -type d -name 'PROF_*' | sort)
if [[ "${profile_status}" -ne 0 ]]; then
  echo "[warning] msprof returned ${profile_status} after the application exited" >&2
  echo "[hint] inspect ${raw_log}, disk capacity, and files under ${profile_dir}" >&2
  if [[ "${#prof_dirs[@]}" -eq 0 ]]; then
    echo "[error] no PROF_* data is available for recovery analysis" >&2
    exit "${profile_status}"
  fi
  echo "[warning] found ${#prof_dirs[@]} PROF_* directories; attempting recovery analysis" >&2
fi

# torchrun may leave one raw PROF_* directory per worker even when the model
# exits normally. msprof-analyze requires each directory to be exported and
# analyzed by msprof first. Repeating these commands on parsed data is safe.
if [[ "${MSPROF_RECOVER_PARSE}" == "true" || \
      ("${MSPROF_RECOVER_PARSE}" == "auto" && "${profile_status}" -ne 0) ]]; then
  recovery_log="${run_dir}/msprof_recovery.log"
  recovery_failures=0
  for prof_dir in "${prof_dirs[@]}"; do
    echo "[recover] export $(basename "${prof_dir}")"
    if ! msprof --export=on --output="${prof_dir}" 2>&1 | tee -a "${recovery_log}"; then
      echo "[warning] msprof export failed: ${prof_dir}" >&2
      recovery_failures=$((recovery_failures + 1))
      continue
    fi
    echo "[recover] analyze $(basename "${prof_dir}")"
    if ! msprof --analyze=on --output="${prof_dir}" 2>&1 | tee -a "${recovery_log}"; then
      echo "[warning] msprof analyze failed: ${prof_dir}" >&2
      recovery_failures=$((recovery_failures + 1))
    fi
  done
  if [[ "${recovery_failures}" -ne 0 ]]; then
    echo "[warning] ${recovery_failures} profiling directories failed offline parsing" >&2
    echo "[hint] inspect ${recovery_log}" >&2
  fi
fi

# This is useful for correlating the trace with the generated sample. It is not
# an official latency result because profiler collection adds overhead.
python scripts/summarize_npu_benchmark.py \
  "${raw_log}" --warmup-per-rank 0 | tee "${benchmark_summary}"

if [[ "${#prof_dirs[@]}" -eq 0 ]]; then
  echo "[error] no PROF_* directory found under ${profile_dir}" >&2
  exit 1
fi

# A single PROF_* directory is the common single-node output. If the installed
# profiler emits one directory per process, pass their common parent instead.
analyze_input="${profile_dir}"
if [[ "${#prof_dirs[@]}" -eq 1 ]]; then
  analyze_input="${prof_dirs[0]}"
fi
echo "[analyze] input=${analyze_input}, PROF directories=${#prof_dirs[@]}"

run_analysis() {
  local name="$1"
  shift
  echo "[analyze] ${name}"
  if ! msprof-analyze "$@"; then
    echo "[warning] analysis '${name}' is unsupported or failed; continuing" >&2
  fi
}

run_analysis all \
  -m all -d "${analyze_input}" -o "${analysis_dir}/all" \
  --export_type text --parallel_mode concurrent
run_analysis compute_op_sum \
  -m compute_op_sum -d "${analyze_input}" -o "${analysis_dir}/compute_op_sum" \
  --export_type text --parallel_mode concurrent
run_analysis hccl_sum \
  -m hccl_sum -d "${analyze_input}" -o "${analysis_dir}/hccl_sum" \
  --export_type text --top_num 30
run_analysis communication_time_sum \
  -m communication_time_sum -d "${analyze_input}" \
  -o "${analysis_dir}/communication_time_sum" --export_type text
run_analysis communication_matrix_sum \
  -m communication_matrix_sum -d "${analyze_input}" \
  -o "${analysis_dir}/communication_matrix_sum" --export_type text
run_analysis slow_rank \
  -m slow_rank -d "${analyze_input}" -o "${analysis_dir}/slow_rank" \
  --export_type text
run_analysis free_analysis \
  -m free_analysis -d "${analyze_input}" -o "${analysis_dir}/free_analysis" \
  --export_type text
run_analysis advisor \
  advisor all -d "${analyze_input}" -o "${analysis_dir}/advisor"

echo "[done] video=${video_dir}"
echo "[done] raw log=${raw_log}"
echo "[done] profiled benchmark summary=${benchmark_summary}"
echo "[done] profiling=${profile_dir}"
echo "[done] analysis=${analysis_dir}"
echo "[note] use an unprofiled run for final latency/FPS because msprof adds overhead"
if [[ "${profile_status}" -ne 0 ]]; then
  echo "[note] msprof collection returned ${profile_status}; treat recovered analysis as potentially incomplete"
fi
