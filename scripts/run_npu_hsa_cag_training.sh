#!/usr/bin/env bash
set -euo pipefail

# ---------- User-editable distributed/runtime settings ----------
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export SP_SIZE="${SP_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export SHARDING_STRATEGY="${SHARDING_STRATEGY:-}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29600}"
export MAX_ITERS="${MAX_ITERS:-2000}"
export HSA_BACKEND="${HSA_BACKEND:-ascend_triton}"
export HSA_QUERY_BLOCK_BATCH="${HSA_QUERY_BLOCK_BATCH:-1}"
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
if (( NNODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo "[error] require NNODES > 0 and 0 <= NODE_RANK < NNODES; got NNODES=${NNODES}, NODE_RANK=${NODE_RANK}" >&2
    exit 2
fi
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if (( WORLD_SIZE % SP_SIZE != 0 )); then
    echo "[error] WORLD_SIZE=${WORLD_SIZE} must be divisible by SP_SIZE=${SP_SIZE}" >&2
    exit 2
fi
DP_SIZE=$((WORLD_SIZE / SP_SIZE))
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

if [[ "${HSA_BACKEND}" == "ascend_triton" ]]; then
    "${PYTHON}" - <<'PY'
from wan_5b.modules.sparse_attention_ascend import (
    ascend_triton_available,
    ascend_triton_unavailable_reason,
)

if not ascend_triton_available():
    raise RuntimeError(
        "HSA_BACKEND=ascend_triton is unavailable: "
        + ascend_triton_unavailable_reason()
    )
print("[run] Ascend Triton HSA backend is available")
PY
fi

PROMPT_COUNT="$(${PYTHON} -c 'import sys; print(sum(bool(x.strip()) for x in open(sys.argv[1], encoding="utf-8")))' "${TRAIN_PROMPTS}")"
EFFECTIVE_BATCH=$((DP_SIZE * GRADIENT_ACCUMULATION_STEPS))
echo "[run] config=${CONFIG_PATH}"
echo "[run] node=${NODE_RANK}/${NNODES} devices=${ASCEND_RT_VISIBLE_DEVICES} local_nproc=${NPROC_PER_NODE} world=${WORLD_SIZE} SP=${SP_SIZE} DP=${DP_SIZE} effective_batch=${EFFECTIVE_BATCH}"
echo "[run] prompts=${PROMPT_COUNT} model_root=${MODEL_ROOT}"
echo "[run] generator_ckpt=${GENERATOR_CKPT}"
echo "[run] logs=${LOG_DIR}"

CONFIG_OVERRIDE="${LOG_DIR}/resolved_config.yaml"
CONFIG_READY="${CONFIG_OVERRIDE}.ready"
if (( NODE_RANK == 0 )); then
    rm -f "${CONFIG_READY}"
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
config.infra.sequence_parallel_size = int(os.environ["SP_SIZE"])
if os.environ["SHARDING_STRATEGY"]:
    config.infra.sharding_strategy = os.environ["SHARDING_STRATEGY"]
config.model_kwargs.sparse_config.backend = os.environ["HSA_BACKEND"]
query_block_batch = int(os.environ["HSA_QUERY_BLOCK_BATCH"])
if query_block_batch <= 0:
    raise ValueError("HSA_QUERY_BLOCK_BATCH must be positive")
config.model_kwargs.sparse_config.query_block_batch = query_block_batch
OmegaConf.save(config, output)
PY
    touch "${CONFIG_READY}"
else
    echo "[run] waiting for rank-0 config: ${CONFIG_READY}"
    for ((attempt = 0; attempt < 600; attempt++)); do
        [[ -f "${CONFIG_READY}" ]] && break
        sleep 1
    done
    if [[ ! -f "${CONFIG_READY}" ]]; then
        echo "[error] timed out waiting for rank-0 config: ${CONFIG_READY}" >&2
        exit 2
    fi
fi

if (( NNODES == 1 )); then
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
elif (( NODE_RANK == 0 )); then
    "${PYTHON}" - "${MASTER_ADDR}" "${MASTER_PORT}" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind((host, port))
    except OSError as exc:
        raise SystemExit(
            f"[error] rendezvous {host}:{port} is unavailable; "
            "choose the same free MASTER_PORT on both nodes"
        ) from exc
PY
fi
export MASTER_PORT
echo "[run] rendezvous=${MASTER_ADDR}:${MASTER_PORT} max_iters=${MAX_ITERS}"
echo "[run] hsa_backend=${HSA_BACKEND} hsa_query_block_batch=${HSA_QUERY_BLOCK_BATCH} progress=${LLV2_TRAIN_PROGRESS}"
echo "[run] sharding_strategy=${SHARDING_STRATEGY:-config default}"
echo "[run] ascend_launch_blocking=${ASCEND_LAUNCH_BLOCKING:-0} task_queue_enable=${TASK_QUEUE_ENABLE:-default}"

extra_args=()
if [[ "${DISABLE_WANDB}" == "1" ]]; then
    extra_args+=(--disable-wandb)
fi

exec "${TORCHRUN}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train.py \
    --config_path "${CONFIG_OVERRIDE}" \
    --logdir "${LOG_DIR}" \
    --wandb-save-dir "${WANDB_DIR}" \
    "${extra_args[@]}"
