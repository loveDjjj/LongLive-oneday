#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/evaluation/runtime.sh
source "${SCRIPT_DIR}/runtime.sh"
evaluation_runtime_init "0,1,2,3,4"

# ---------- 可覆盖参数 ----------
TASK="${TASK:-benchmark}"
METHODS="${METHODS:-dense,hsa_cag,sla_cag,hsa_sla_cag}"
DURATIONS="${DURATIONS:-5s,32s,64s}"
if [[ "${LLV2_DEVICE}" == "cuda" ]]; then
  MODES="${MODES:-dit_only,sync_vae}"
  LONGLIVE_SP_SIZE="${LONGLIVE_SP_SIZE:-4}"
else
  MODES="${MODES:-dit_only,sync_vae,async_vae}"
  LONGLIVE_SP_SIZE="${LONGLIVE_SP_SIZE:-1,4}"
fi
LONGLIVE_DP_SIZE="${LONGLIVE_DP_SIZE:-1}"
LONGLIVE_DENSE_PREFIX_CHUNKS="${LONGLIVE_DENSE_PREFIX_CHUNKS:-}"
PERF_DEVICES="${PERF_DEVICES:-${VISIBLE_DEVICES}}"
SUITE_ID="${SUITE_ID:-sparse_matrix_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_SUITE="${RESUME_SUITE:-1}"

if [[ "${TASK}" != "benchmark" && "${TASK}" != "msprof" ]]; then
  echo "[error] TASK must be benchmark or msprof" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "[error] DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESUME_SUITE}" != "0" && "${RESUME_SUITE}" != "1" ]]; then
  echo "[error] RESUME_SUITE must be 0 or 1" >&2
  exit 2
fi
if [[ "${LLV2_DEVICE}" == "cuda" && "${TASK}" == "msprof" ]]; then
  echo "[error] msprof only supports Ascend NPU; select TASK=benchmark for CUDA" >&2
  exit 2
fi
if [[ ! "${LONGLIVE_DP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] LONGLIVE_DP_SIZE must be a positive integer" >&2
  exit 2
fi
IFS=',' read -r -a methods <<<"${METHODS}"
IFS=',' read -r -a durations <<<"${DURATIONS}"
IFS=',' read -r -a modes <<<"${MODES}"
IFS=',' read -r -a sp_sizes <<<"${LONGLIVE_SP_SIZE}"
IFS=',' read -r -a devices <<<"${PERF_DEVICES}"
required_device_count=0
for sp_size in "${sp_sizes[@]}"; do
  if [[ ! "${sp_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[error] LONGLIVE_SP_SIZE must contain positive integers, got ${sp_size}" >&2
    exit 2
  fi
  if (( 24 % sp_size != 0 || 8 % sp_size != 0 )); then
    echo "[error] SP size ${sp_size} must divide both 24 heads and 8 latent frames" >&2
    exit 2
  fi
  case_required=$((sp_size * LONGLIVE_DP_SIZE))
  for mode in "${modes[@]}"; do
    if [[ "${mode}" == "async_vae" ]]; then
      if (( LONGLIVE_DP_SIZE != 1 )); then
        echo "[error] async_vae requires LONGLIVE_DP_SIZE=1" >&2
        exit 2
      fi
      case_required=$((sp_size + 1))
    fi
  done
  if (( case_required > required_device_count )); then
    required_device_count="${case_required}"
  fi
done
if (( ${#devices[@]} < required_device_count )); then
  echo "[error] selected modes require at least ${required_device_count} ${LLV2_DEVICE} device ids in PERF_DEVICES" >&2
  exit 2
fi

devices_for_case() {
  local sp_size="$1" mode="$2" count="$(( $1 * LONGLIVE_DP_SIZE ))" selected="" index
  if [[ "${mode}" == "async_vae" ]]; then
    count=$((sp_size + 1))
  fi
  for ((index = 0; index < count; index++)); do
    [[ -n "${selected}" ]] && selected+=","
    selected+="${devices[index]}"
  done
  printf '%s' "${selected}"
}

checkpoint_for_method() {
  local method="$1" variable_name checkpoint
  variable_name="$(tr '[:lower:]' '[:upper:]' <<<"${method}")_GENERATOR_CKPT"
  checkpoint="${!variable_name:-${LONGLIVE_GENERATOR_CKPT:-}}"
  if [[ -n "${checkpoint}" && ! -f "${checkpoint}" ]]; then
    echo "[error] checkpoint for ${method} does not exist: ${checkpoint}" >&2
    exit 2
  fi
  printf '%s' "${checkpoint}"
}

for method in "${methods[@]}"; do
  if [[ ! "${method}" =~ ^(dense|hsa_cag|sla_cag|hsa_sla_cag)$ ]]; then
    echo "[error] unsupported method: ${method}" >&2
    exit 2
  fi
  checkpoint="$(checkpoint_for_method "${method}")"
  for duration in "${durations[@]}"; do
    if [[ ! "${duration}" =~ ^(5s|32s|64s)$ ]]; then
      echo "[error] unsupported duration: ${duration}" >&2
      exit 2
    fi
    for sp_size in "${sp_sizes[@]}"; do
      for mode in "${modes[@]}"; do
        if [[ ! "${mode}" =~ ^(dit_only|sync_vae|async_vae)$ ]]; then
          echo "[error] unsupported mode: ${mode}" >&2
          exit 2
        fi
        visible="$(devices_for_case "${sp_size}" "${mode}")"
        echo "[suite] task=${TASK} method=${method} duration=${duration} mode=${mode} sp=${sp_size} checkpoint=${checkpoint:-config-default}"
        run_id="${SUITE_ID}-${method}-${duration}-${mode}-sp${sp_size}"
        if (( LONGLIVE_DP_SIZE > 1 )); then
          run_id+="-dp${LONGLIVE_DP_SIZE}"
        fi
        if [[ "${TASK}" == "benchmark" ]]; then
          run_dir="runs/performance/${run_id}"
        else
          run_dir="runs/msprof/dit/${run_id}"
        fi
        if [[ "${DRY_RUN}" == "1" ]]; then
          echo "[dry-run] devices=${visible} run_id=${run_id} run_dir=${run_dir} accelerator=${LLV2_DEVICE} dp=${LONGLIVE_DP_SIZE}"
          continue
        fi
        case_env=(
          env
          "${VISIBLE_DEVICES_VARIABLE}=${visible}"
          "LONGLIVE_DP_SIZE=${LONGLIVE_DP_SIZE}"
          "LONGLIVE_SP_SIZE=${sp_size}"
          "LONGLIVE_SPARSE_METHOD=${method}"
        )
        if [[ -n "${checkpoint}" ]]; then
          case_env+=("LONGLIVE_GENERATOR_CKPT=${checkpoint}")
        fi
        if [[ -f "${run_dir}/summary.json" && "${RESUME_SUITE}" == "1" ]]; then
          resume_python="${GENERATION_ENV}/bin/python"
          [[ -x "${resume_python}" ]] || resume_python="$(command -v python || command -v python3)"
          "${case_env[@]}" "${resume_python}" scripts/evaluation/resolve_config.py "${TASK}" \
            --config "${CONFIG_PATH:-configs/inference/msprof.yaml}" --preset "${duration}" \
            --vae-mode "${mode}" --validate-resume "${run_dir}/manifest.json" >/dev/null
          echo "[resume] completed case skipped: ${run_id}"
          continue
        fi
        if [[ -e "${run_dir}" ]]; then
          echo "[error] incomplete or existing case cannot be overwritten: ${run_dir}" >&2
          echo "[hint] inspect it, choose a new SUITE_ID, or remove it explicitly after preserving evidence" >&2
          exit 2
        fi
        if [[ "${TASK}" == "benchmark" ]]; then
          "${case_env[@]}" \
            BENCHMARK_MODE="${mode}" \
            RUN_ID="${run_id}" \
            bash scripts/evaluation/run_benchmark.sh "${duration}"
        else
          "${case_env[@]}" \
            MSPROF_MODE="${mode}" \
            RUN_ID="${run_id}" \
            bash scripts/evaluation/run_msprof.sh "${duration}"
        fi
      done
    done
  done
done

if [[ "${DRY_RUN}" == "0" ]]; then
  suite_python="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}/bin/python"
  [[ -x "${suite_python}" ]] || suite_python="python"
  "${suite_python}" scripts/evaluation/summarize_suite.py "${TASK}" \
    --suite-id "${SUITE_ID}"
fi
