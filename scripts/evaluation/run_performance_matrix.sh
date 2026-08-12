#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-benchmark}"
METHODS="${METHODS:-dense,hsa_cag,sla_cag,hsa_sla_cag}"
DURATIONS="${DURATIONS:-5s,32s,64s}"
MODES="${MODES:-dit_only,sync_vae,async_vae}"
PERF_DEVICES="${PERF_DEVICES:-0,1,2,3,4}"
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
IFS=',' read -r -a methods <<<"${METHODS}"
IFS=',' read -r -a durations <<<"${DURATIONS}"
IFS=',' read -r -a modes <<<"${MODES}"
IFS=',' read -r -a devices <<<"${PERF_DEVICES}"
required_device_count=4
for mode in "${modes[@]}"; do
  [[ "${mode}" == "async_vae" ]] && required_device_count=5
done
if (( ${#devices[@]} < required_device_count )); then
  echo "[error] selected modes require at least ${required_device_count} NPU ids in PERF_DEVICES" >&2
  exit 2
fi
workers="${devices[0]},${devices[1]},${devices[2]},${devices[3]}"
workers_and_vae="${workers}"
if (( required_device_count == 5 )); then
  workers_and_vae="${workers},${devices[4]}"
fi

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
    for mode in "${modes[@]}"; do
      if [[ ! "${mode}" =~ ^(dit_only|sync_vae|async_vae)$ ]]; then
        echo "[error] unsupported mode: ${mode}" >&2
        exit 2
      fi
      visible="${workers}"
      [[ "${mode}" == "async_vae" ]] && visible="${workers_and_vae}"
      echo "[suite] task=${TASK} method=${method} duration=${duration} mode=${mode} checkpoint=${checkpoint:-config-default}"
      run_id="${SUITE_ID}-${method}-${duration}-${mode}"
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] devices=${visible} run_id=${run_id}"
        continue
      fi
      run_dir="runs/${TASK}/${run_id}"
      if [[ -f "${run_dir}/summary.json" && "${RESUME_SUITE}" == "1" ]]; then
        echo "[resume] completed case skipped: ${run_id}"
        continue
      fi
      if [[ -e "${run_dir}" ]]; then
        echo "[error] incomplete or existing case cannot be overwritten: ${run_dir}" >&2
        echo "[hint] inspect it, choose a new SUITE_ID, or remove it explicitly after preserving evidence" >&2
        exit 2
      fi
      case_env=(
        env
        "ASCEND_RT_VISIBLE_DEVICES=${visible}"
        "LONGLIVE_SPARSE_METHOD=${method}"
      )
      if [[ -n "${checkpoint}" ]]; then
        case_env+=("LONGLIVE_GENERATOR_CKPT=${checkpoint}")
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

if [[ "${DRY_RUN}" == "0" ]]; then
  suite_python="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}/bin/python"
  [[ -x "${suite_python}" ]] || suite_python="python"
  "${suite_python}" scripts/evaluation/summarize_suite.py "${TASK}" \
    --suite-id "${SUITE_ID}"
fi
