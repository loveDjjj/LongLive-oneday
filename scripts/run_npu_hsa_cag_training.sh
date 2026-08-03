#!/usr/bin/env bash
set -euo pipefail

# ---------- User-editable distributed/runtime settings ----------
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-12}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29600}"
export MAX_ITERS="${MAX_ITERS:-2000}"
export HSA_QUERY_BLOCK_BATCH="${HSA_QUERY_BLOCK_BATCH:-2}"
export LLV2_TRAIN_PROGRESS="${LLV2_TRAIN_PROGRESS:-1}"
export LLV2_DEVICE="npu"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# ---------- User-editable paths ----------
export LONGLIVE_ROOT="${LONGLIVE_ROOT:-/mnt/share/r50063443/LongLive-oneday}"
export GENERATION_ENV="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}"
export CONFIG_PATH="${CONFIG_PATH:-configs/train_dmd_hsa_cag_npu_bf16.yaml}"
export MODEL_ROOT="${MODEL_ROOT:-/mnt/share/weight/Wan2.2-TI2V-5B}"
export GENERATOR_CKPT="${GENERATOR_CKPT:-/mnt/share/weight/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt}"
export TRAIN_PROMPTS="${TRAIN_PROMPTS:-data/train/vidprom_filtered_extended/prompts_train.txt}"
export TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_longlive2_hsa_cag_npu_bf16}"
export DISABLE_WANDB="${DISABLE_WANDB:-1}"

cd "${LONGLIVE_ROOT}"

PYTHON="${GENERATION_ENV}/bin/python"
TORCHRUN="${GENERATION_ENV}/bin/torchrun"
LOG_DIR="logs/training/${TRAIN_RUN_NAME}"
WANDB_DIR="${LOG_DIR}/wandb"
mkdir -p "${LOG_DIR}" "${WANDB_DIR}"

IFS=',' read -r -a visible_devices <<< "${ASCEND_RT_VISIBLE_DEVICES}"
if (( ${#visible_devices[@]} != NPROC_PER_NODE )); then
    echo "[error] NPROC_PER_NODE=${NPROC_PER_NODE}, but ASCEND_RT_VISIBLE_DEVICES has ${#visible_devices[@]} devices" >&2
    exit 2
fi
for required in \
    "${PYTHON}" \
    "${TORCHRUN}" \
    "${CONFIG_PATH}" \
    "${GENERATOR_CKPT}" \
    "${TRAIN_PROMPTS}" \
    "${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth" \
    "${MODEL_ROOT}/google/umt5-xxl" \
    "${MODEL_ROOT}/Wan2.2_VAE.pth"; do
    if [[ ! -e "${required}" ]]; then
        echo "[error] required path does not exist: ${required}" >&2
        exit 2
    fi
done

PROMPT_COUNT="$(${PYTHON} -c 'import sys; print(sum(bool(x.strip()) for x in open(sys.argv[1], encoding="utf-8")))' "${TRAIN_PROMPTS}")"
EFFECTIVE_BATCH=$((NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))
echo "[run] config=${CONFIG_PATH}"
echo "[run] devices=${ASCEND_RT_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} effective_batch=${EFFECTIVE_BATCH}"
echo "[run] prompts=${PROMPT_COUNT} model_root=${MODEL_ROOT}"
echo "[run] generator_ckpt=${GENERATOR_CKPT}"
echo "[run] logs=${LOG_DIR}"

CONFIG_OVERRIDE="${LOG_DIR}/resolved_config.yaml"
"${PYTHON}" - "${CONFIG_PATH}" "${CONFIG_OVERRIDE}" <<'PY'
import os
import sys
from omegaconf import OmegaConf

source, output = sys.argv[1:]
config = OmegaConf.load(source)
config.model_kwargs.model_root = os.environ["MODEL_ROOT"]
config.real_model_kwargs.model_root = os.environ["MODEL_ROOT"]
config.fake_model_kwargs.model_root = os.environ["MODEL_ROOT"]
config.checkpoints.generator_ckpt = os.environ["GENERATOR_CKPT"]
config.data.data_path = os.environ["TRAIN_PROMPTS"]
config.training.gradient_accumulation_steps = int(os.environ["GRADIENT_ACCUMULATION_STEPS"])
config.training.max_iters = int(os.environ["MAX_ITERS"])
query_block_batch = int(os.environ["HSA_QUERY_BLOCK_BATCH"])
if query_block_batch <= 0:
    raise ValueError("HSA_QUERY_BLOCK_BATCH must be positive")
config.model_kwargs.sparse_config.query_block_batch = query_block_batch
OmegaConf.save(config, output)
PY

MASTER_PORT="$(${PYTHON} - "${MASTER_ADDR}" "${MASTER_PORT}" <<'PY'
import socket
import sys

host, preferred = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind((host, preferred))
        print(preferred)
    except OSError:
        sock.bind((host, 0))
        print(sock.getsockname()[1])
PY
)"
export MASTER_PORT
echo "[run] rendezvous=${MASTER_ADDR}:${MASTER_PORT} max_iters=${MAX_ITERS}"
echo "[run] hsa_query_block_batch=${HSA_QUERY_BLOCK_BATCH} progress=${LLV2_TRAIN_PROGRESS}"

extra_args=()
if [[ "${DISABLE_WANDB}" == "1" ]]; then
    extra_args+=(--disable-wandb)
fi

exec "${TORCHRUN}" \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train.py \
    --config_path "${CONFIG_OVERRIDE}" \
    --logdir "${LOG_DIR}" \
    --wandb-save-dir "${WANDB_DIR}" \
    "${extra_args[@]}"
