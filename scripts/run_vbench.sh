#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Deployment settings. Model layout, data, frame count, and seeds come from YAML.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11}"
export GENERATION_ENV="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}"
export AISBENCH_ENV="${AISBENCH_ENV:-/mnt/share/r50063443/conda_envs/aisbench_npu}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/mnt/weight/vbench_models/}"
export CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29530}"

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
trap 'rm -f "${metadata_tmp}"' EXIT
"${GENERATION_ENV}/bin/python" scripts/resolve_inference_config.py vbench \
  --config "${CONFIG_PATH}" --preset "${PRESET}" >"${metadata_tmp}"

json_field() {
  "${GENERATION_ENV}/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1]))[sys.argv[2]]; print(value)' \
    "${metadata_tmp}" "$1"
}
run_tag="$(json_field run_tag)"
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
raw_root="${run_dir}/videos/raw"
prepared_dir="${run_dir}/videos/prepared"
mkdir -p "${raw_root}" "${prepared_dir}"
cp "${metadata_tmp}" "${run_dir}/manifest.json"

select_master_port() {
  "${GENERATION_ENV}/bin/python" - "$1" "$2" <<'PY'
import socket, sys
host, preferred = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind((host, preferred))
        print(preferred)
    except OSError:
        sock.bind((host, 0))
        print(sock.getsockname()[1])
PY
}

echo "[run] task=vbench preset=${PRESET} run_id=${run_id}"
echo "[run] devices=${ASCEND_RT_VISIBLE_DEVICES} layout=SP${sp_size}xDP${dp_size} prompts=${prompt_count}"
read -r -a seed_array <<<"${seeds}"
for sample_index in "${!seed_array[@]}"; do
  seed="${seed_array[${sample_index}]}"
  seed_dir="${raw_root}/seed_${seed}"
  raw_log="${run_dir}/seed_${seed}.log"
  resolved_config="${run_dir}/resolved_seed_${seed}.yaml"
  mkdir -p "${seed_dir}"
  completed="$(find "${seed_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
  if [[ "${completed}" -eq "${prompt_count}" ]]; then
    echo "[resume] seed=${seed} already complete (${completed}/${prompt_count})"
  else
    "${GENERATION_ENV}/bin/python" scripts/resolve_inference_config.py vbench \
      --config "${CONFIG_PATH}" --preset "${PRESET}" --seed "${seed}" \
      --output "${resolved_config}" --output-folder "${seed_dir}" >/dev/null
    seed_port="$(select_master_port "${MASTER_ADDR}" "$((MASTER_PORT + sample_index))")"
    echo "[generate] seed=${seed} ($((sample_index + 1))/${#seed_array[@]}) port=${seed_port}"
    LLV2_DEVICE=npu "${GENERATION_ENV}/bin/torchrun" \
      --nnodes=1 \
      --nproc_per_node="${nproc}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${seed_port}" \
      "${entrypoint}" \
      --config_path "${resolved_config}" \
      >"${raw_log}" 2>&1 || {
        tail -n 100 "${raw_log}" >&2
        exit 1
      }
  fi

  "${GENERATION_ENV}/bin/python" third_party/aisbench_adapter/prepare_vbench_videos.py \
    --benchmark standard \
    --src-dir "${seed_dir}" \
    --prompts-file "${naming_prompts}" \
    --dst-dir "${prepared_dir}" \
    --sample-index "${sample_index}" \
    --overwrite
done

echo "[evaluate] videos=${prepared_dir}"
aisbench_log="${run_dir}/aisbench.log"
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
  bash third_party/aisbench_adapter/run_vbench_eval.sh \
  2>&1 | tee "${aisbench_log}"

if grep -q '\[RUNNER-TASK-001\]' "${aisbench_log}"; then
  echo "[error] one or more AISBench tasks failed; inspect ${aisbench_log}" >&2
  exit 1
fi
echo "[done] run=${run_dir}"
