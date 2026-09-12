#!/usr/bin/env bash
set -euo pipefail

# ---------- 可覆盖的分布式与运行参数 ----------
export LLV2_DEVICE="${LLV2_DEVICE:-npu}"
export DRY_RUN="${DRY_RUN:-0}"
if [[ ! "${DRY_RUN}" =~ ^[01]$ ]]; then
    echo "[error] DRY_RUN must be 0 or 1" >&2
    exit 2
fi
case "${LLV2_DEVICE}" in
    cuda)
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
        TRAIN_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
        TRAIN_DEVICE_VARIABLE=CUDA_VISIBLE_DEVICES
        export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
        export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
        export SPARSE_BACKEND="${SPARSE_BACKEND:-${SLA_BACKEND:-portable}}"
        # 冻结底座按 BF16 加载以降低初始化内存；可训练参数仍使用 FP32 原始权重。
        export MODEL_LOAD_DTYPE="${MODEL_LOAD_DTYPE:-bfloat16}"
        export LLV2_DISTRIBUTED_BACKEND="${LLV2_DISTRIBUTED_BACKEND:-nccl}"
        ;;
    npu)
        export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
        TRAIN_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
        TRAIN_DEVICE_VARIABLE=ASCEND_RT_VISIBLE_DEVICES
        export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
        export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
        export SPARSE_BACKEND="${SPARSE_BACKEND:-${SLA_BACKEND:-ascend_triton}}"
        export MODEL_LOAD_DTYPE="${MODEL_LOAD_DTYPE:-float32}"
        export LLV2_DISTRIBUTED_BACKEND="${LLV2_DISTRIBUTED_BACKEND:-hccl}"
        export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
        export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
        ;;
    *) echo "[error] LLV2_DEVICE must be npu or cuda" >&2; exit 2 ;;
esac
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export LONGLIVE_SP_SIZE="${LONGLIVE_SP_SIZE:-${SP_SIZE:-4}}"
export SP_SIZE="${LONGLIVE_SP_SIZE}"
export SHARDING_STRATEGY="${SHARDING_STRATEGY:-}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
unset MASTER_PORT
export MAX_ITERS="${MAX_ITERS:-2000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export VIS_INTERVAL="${VIS_INTERVAL:-100}"
export MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-20}"
export SPARSE_METHOD="${SPARSE_METHOD:-sla_cag}"
export SPARSE_QUERY_BLOCK_BATCH="${SPARSE_QUERY_BLOCK_BATCH:-${SLA_QUERY_BLOCK_BATCH:-1}}"
export LONGLIVE_DENSE_PREFIX_CHUNKS="${LONGLIVE_DENSE_PREFIX_CHUNKS:-1}"
export GENERATOR_TRAIN_SCOPE="${GENERATOR_TRAIN_SCOPE:-}"
export GENERATOR_LR="${GENERATOR_LR:-}"
export LINEAR_LR="${LINEAR_LR:-}"
export LLV2_TRAIN_PROGRESS="${LLV2_TRAIN_PROGRESS:-1}"

# ---------- 可覆盖的路径参数 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export LONGLIVE_ROOT="${LONGLIVE_ROOT:-${REPO_ROOT}}"
if [[ "${LLV2_DEVICE}" == "cuda" ]]; then
    export GENERATION_ENV="${GENERATION_ENV:-${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}}"
else
    export GENERATION_ENV="${GENERATION_ENV:-/mnt/share/r50063443/conda_envs/longlive}"
fi
if [[ ! "${SPARSE_METHOD}" =~ ^(sla_cag|hsa_cag|hsa_sla_cag)$ ]]; then
    echo "[error] SPARSE_METHOD must be sla_cag, hsa_cag, or hsa_sla_cag" >&2
    exit 2
fi
export CONFIG_PATH="${CONFIG_PATH:-configs/train/${SPARSE_METHOD}.yaml}"
export MODEL_ROOT="${MODEL_ROOT:-/mnt/share/r50063443/Wan2.2-TI2V-5B}"
export GENERATOR_CKPT="${GENERATOR_CKPT:-/mnt/share/r50063443/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt}"
export TRAIN_PROMPTS="${TRAIN_PROMPTS:-data/train/vidprom_filtered_extended/prompts_train.txt}"
export TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-$(date +%Y%m%d_%H%M%S)_longlive2_${SPARSE_METHOD}_${LLV2_DEVICE}_bf16}"
export DISABLE_WANDB="${DISABLE_WANDB:-1}"
export VALIDATE_LINEAR_CHECKPOINT="${VALIDATE_LINEAR_CHECKPOINT:-1}"

cd "${LONGLIVE_ROOT}"

PYTHON="${GENERATION_ENV}/bin/python"
TORCHRUN="${GENERATION_ENV}/bin/torchrun"
ARTIFACT_DIR="runs/training/${TRAIN_RUN_NAME}"
LOG_DIR="logs/training/${TRAIN_RUN_NAME}"
WANDB_DIR="${ARTIFACT_DIR}/wandb"
TEXT_LOG="${LOG_DIR}/node_${NODE_RANK}.log"
METRICS_LOG="${LOG_DIR}/metrics.jsonl"
if [[ "${DRY_RUN}" == "1" && -e "${ARTIFACT_DIR}" ]]; then
    echo "[error] dry-run refuses to overwrite existing directory: ${ARTIFACT_DIR}" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "1" && -e "${ARTIFACT_DIR}" ]] && (( NNODES == 1 )); then
    resumable_checkpoint=false
    for checkpoint in "${ARTIFACT_DIR}"/checkpoints/step_*/train_state.pt; do
        if [[ -f "${checkpoint}" ]]; then
            resumable_checkpoint=true
            break
        fi
    done
    if [[ "${resumable_checkpoint}" != "true" ]]; then
        echo "[error] existing training directory has no train_state.pt; inspect it and use a new TRAIN_RUN_NAME: ${ARTIFACT_DIR}" >&2
        exit 2
    fi
fi
mkdir -p "${ARTIFACT_DIR}" "${LOG_DIR}" "${WANDB_DIR}"
if [[ "${DRY_RUN}" != "1" ]]; then
    exec > >(tee -a "${TEXT_LOG}") 2>&1
fi
printf '\n[launch] time=%s node=%s/%s run=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${NODE_RANK}" "${NNODES}" "${TRAIN_RUN_NAME}"

IFS=',' read -r -a visible_devices <<< "${TRAIN_VISIBLE_DEVICES}"
if (( ${#visible_devices[@]} != NPROC_PER_NODE )); then
    echo "[error] NPROC_PER_NODE=${NPROC_PER_NODE}, but ${TRAIN_DEVICE_VARIABLE} has ${#visible_devices[@]} devices" >&2
    exit 2
fi
if (( NNODES <= 0 || NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo "[error] require NNODES > 0 and 0 <= NODE_RANK < NNODES; got NNODES=${NNODES}, NODE_RANK=${NODE_RANK}" >&2
    exit 2
fi
if (( MAX_ITERS <= 0 || SAVE_INTERVAL <= 0 || VIS_INTERVAL < 0 || MAX_CHECKPOINTS <= 0 )); then
    echo "[error] require MAX_ITERS, SAVE_INTERVAL, MAX_CHECKPOINTS > 0 and VIS_INTERVAL >= 0" >&2
    exit 2
fi
if [[ ! "${VALIDATE_LINEAR_CHECKPOINT}" =~ ^[01]$ ]]; then
    echo "[error] VALIDATE_LINEAR_CHECKPOINT must be 0 or 1" >&2
    exit 2
fi
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if (( SP_SIZE <= 0 || GRADIENT_ACCUMULATION_STEPS <= 0 )); then
    echo "[error] LONGLIVE_SP_SIZE and GRADIENT_ACCUMULATION_STEPS must be positive" >&2
    exit 2
fi
if (( WORLD_SIZE % SP_SIZE != 0 )); then
    echo "[error] WORLD_SIZE=${WORLD_SIZE} must be divisible by LONGLIVE_SP_SIZE=${SP_SIZE}" >&2
    exit 2
fi
if (( 24 % SP_SIZE != 0 || 8 % SP_SIZE != 0 )); then
    echo "[error] LONGLIVE_SP_SIZE=${SP_SIZE} must divide both 24 attention heads and 8 latent frames per block" >&2
    exit 2
fi
DP_SIZE=$((WORLD_SIZE / SP_SIZE))
required_paths=("${PYTHON}" "${CONFIG_PATH}")
if [[ "${DRY_RUN}" != "1" ]]; then
    required_paths+=(
        "${TORCHRUN}" "${GENERATOR_CKPT}" "${TRAIN_PROMPTS}"
        "${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"
        "${MODEL_ROOT}/google/umt5-xxl" "${MODEL_ROOT}/Wan2.2_VAE.pth"
    )
fi
for required in "${required_paths[@]}"; do
    if [[ ! -e "${required}" ]]; then
        echo "[error] required path does not exist: ${required}" >&2
        exit 2
    fi
done

if [[ "${DRY_RUN}" != "1" && "${SPARSE_BACKEND}" == "ascend_triton" ]]; then
    "${PYTHON}" - <<'PY'
from wan_5b.modules.sla_attention_ascend import (
    ascend_triton_available,
    ascend_triton_unavailable_reason,
)

if not ascend_triton_available():
    raise RuntimeError(
        "SPARSE_BACKEND=ascend_triton is unavailable: "
        + ascend_triton_unavailable_reason()
    )
print("[run] Ascend Triton sparse backend is available")
PY
fi

PROMPT_COUNT=unavailable
if [[ -f "${TRAIN_PROMPTS}" ]]; then
    PROMPT_COUNT="$(${PYTHON} -c 'import sys; print(sum(bool(x.strip()) for x in open(sys.argv[1], encoding="utf-8")))' "${TRAIN_PROMPTS}")"
fi
EFFECTIVE_BATCH=$((DP_SIZE * GRADIENT_ACCUMULATION_STEPS))
echo "[run] config=${CONFIG_PATH}"
echo "[run] device=${LLV2_DEVICE} node=${NODE_RANK}/${NNODES} devices=${TRAIN_VISIBLE_DEVICES} local_nproc=${NPROC_PER_NODE} world=${WORLD_SIZE} SP=${SP_SIZE} DP=${DP_SIZE} effective_batch=${EFFECTIVE_BATCH}"
echo "[run] prompts=${PROMPT_COUNT} model_root=${MODEL_ROOT}"
echo "[run] generator_ckpt=${GENERATOR_CKPT}"
echo "[run] artifacts=${ARTIFACT_DIR}"
echo "[run] logs=${LOG_DIR}"

CONFIG_OVERRIDE="${ARTIFACT_DIR}/config.resolved.yaml"
CONFIG_READY="${CONFIG_OVERRIDE}.ready"
RENDEZVOUS_FILE="${ARTIFACT_DIR}/rendezvous.env"
RENDEZVOUS_READY="${RENDEZVOUS_FILE}.ready"
if (( NODE_RANK == 0 )); then
    rm -f "${CONFIG_READY}"
    if (( NNODES > 1 )); then
        rm -f "${RENDEZVOUS_READY}"
    fi
    "${PYTHON}" - "${CONFIG_PATH}" "${CONFIG_OVERRIDE}" "${REPO_ROOT}" <<'PY'
import os
import sys
from omegaconf import OmegaConf

source, output, code_root = sys.argv[1:]
sys.path.insert(0, code_root)
from utils.config import validate_training_resume_contract
config = OmegaConf.load(source)
config.model_kwargs.model_root = os.environ["MODEL_ROOT"]
config.real_model_kwargs.model_root = os.environ["MODEL_ROOT"]
config.fake_model_kwargs.model_root = os.environ["MODEL_ROOT"]
config.checkpoints.generator_ckpt = os.environ["GENERATOR_CKPT"]
config.data.data_path = os.environ["TRAIN_PROMPTS"]
config.training.gradient_accumulation_steps = int(os.environ["GRADIENT_ACCUMULATION_STEPS"])
config.training.max_iters = int(os.environ["MAX_ITERS"])
config.training.log_iters = int(os.environ["SAVE_INTERVAL"])
config.training.max_checkpoints = int(os.environ["MAX_CHECKPOINTS"])
if os.environ["GENERATOR_TRAIN_SCOPE"]:
    config.training.generator_train_scope = os.environ["GENERATOR_TRAIN_SCOPE"]
if os.environ["GENERATOR_LR"]:
    config.training.lr = float(os.environ["GENERATOR_LR"])
if os.environ["LINEAR_LR"]:
    config.training.lr_linear = float(os.environ["LINEAR_LR"])
config.evaluation.interval = int(os.environ["VIS_INTERVAL"])
config.infra.sequence_parallel_size = int(os.environ["SP_SIZE"])
device_type = os.environ["LLV2_DEVICE"]
load_dtype = os.environ["MODEL_LOAD_DTYPE"]
if load_dtype not in {"float32", "bfloat16"}:
    raise ValueError("MODEL_LOAD_DTYPE must be float32 or bfloat16")
config.infra.device_type = device_type
config.infra.distributed_backend = os.environ["LLV2_DISTRIBUTED_BACKEND"]
config.infra.model_load_dtype = load_dtype
config.infra.trainable_parameter_dtype = "float32"
config.infra.compute_dtype = "bfloat16" if config.infra.mixed_precision else "float32"
if os.environ["SHARDING_STRATEGY"]:
    config.infra.sharding_strategy = os.environ["SHARDING_STRATEGY"]
config.model_kwargs.sparse_config.method = os.environ["SPARSE_METHOD"]
config.model_kwargs.sparse_config.backend = os.environ["SPARSE_BACKEND"]
query_block_batch = int(os.environ["SPARSE_QUERY_BLOCK_BATCH"])
if query_block_batch <= 0:
    raise ValueError("SPARSE_QUERY_BLOCK_BATCH must be positive")
config.model_kwargs.sparse_config.query_block_batch = query_block_batch
dense_prefix_chunks = int(os.environ["LONGLIVE_DENSE_PREFIX_CHUNKS"])
if dense_prefix_chunks < 1:
    raise ValueError("LONGLIVE_DENSE_PREFIX_CHUNKS must be at least 1")
config.model_kwargs.sparse_config.dense_prefix_chunks = dense_prefix_chunks
if os.path.exists(output):
    validate_training_resume_contract(OmegaConf.load(output), config)
temporary_output = f"{output}.tmp.{os.getpid()}"
OmegaConf.save(config, temporary_output)
os.replace(temporary_output, output)
PY
    cp "${CONFIG_PATH}" "${ARTIFACT_DIR}/config.source.yaml"
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

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] resolved_config=${CONFIG_OVERRIDE} device=${LLV2_DEVICE} backend=${SPARSE_BACKEND} load_dtype=${MODEL_LOAD_DTYPE}"
    echo "[dry-run] world=${WORLD_SIZE} SP=${SP_SIZE} DP=${DP_SIZE} effective_batch=${EFFECTIVE_BATCH}; torchrun not started"
    exit 0
fi

torchrun_rendezvous_args=(--nnodes="${NNODES}")
if (( NNODES == 1 )); then
    torchrun_rendezvous_args+=(--standalone)
    echo "[run] rendezvous=standalone max_iters=${MAX_ITERS}"
else
    if (( NODE_RANK == 0 )); then
        MASTER_PORT="$(${PYTHON} - "${MASTER_ADDR}" <<'PY'
import socket
import sys

host = sys.argv[1]
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((host, 0))
    print(sock.getsockname()[1])
PY
        )"
        RENDEZVOUS_TMP="${RENDEZVOUS_FILE}.tmp.$$"
        printf 'MASTER_PORT=%q\n' "${MASTER_PORT}" >"${RENDEZVOUS_TMP}"
        mv "${RENDEZVOUS_TMP}" "${RENDEZVOUS_FILE}"
        touch "${RENDEZVOUS_READY}"
    else
        echo "[run] waiting for rank-0 rendezvous: ${RENDEZVOUS_READY}"
        for ((attempt = 0; attempt < 600; attempt++)); do
            [[ -f "${RENDEZVOUS_READY}" ]] && break
            sleep 1
        done
        if [[ ! -f "${RENDEZVOUS_READY}" ]]; then
            echo "[error] timed out waiting for rank-0 rendezvous: ${RENDEZVOUS_READY}" >&2
            exit 2
        fi
        # shellcheck disable=SC1090
        source "${RENDEZVOUS_FILE}"
    fi
    export MASTER_PORT
    torchrun_rendezvous_args+=(
        --node_rank="${NODE_RANK}"
        --master_addr="${MASTER_ADDR}"
        --master_port="${MASTER_PORT}"
    )
    echo "[run] rendezvous=${MASTER_ADDR}:${MASTER_PORT} source=dynamic max_iters=${MAX_ITERS}"
fi
echo "[run] save_interval=${SAVE_INTERVAL} vis_interval=${VIS_INTERVAL} max_checkpoints=${MAX_CHECKPOINTS}"
echo "[run] sparse_method=${SPARSE_METHOD} sparse_backend=${SPARSE_BACKEND} sparse_query_block_batch=${SPARSE_QUERY_BLOCK_BATCH} progress=${LLV2_TRAIN_PROGRESS}"
echo "[run] generator_train_scope=${GENERATOR_TRAIN_SCOPE:-config default} generator_lr=${GENERATOR_LR:-config default} linear_lr=${LINEAR_LR:-config default}"
echo "[run] sharding_strategy=${SHARDING_STRATEGY:-config default}"
echo "[run] ascend_launch_blocking=${ASCEND_LAUNCH_BLOCKING:-0} task_queue_enable=${TASK_QUEUE_ENABLE:-default}"

if (( NODE_RANK == 0 )); then
    "${PYTHON}" scripts/training/write_run_manifest.py \
        --output "${ARTIFACT_DIR}/manifest.json" \
        --run-id "${TRAIN_RUN_NAME}" \
        --kind training \
        --config "${CONFIG_OVERRIDE}" \
        --data "${TRAIN_PROMPTS}" \
        --world-size "${WORLD_SIZE}" \
        --sp-size "${SP_SIZE}" \
        --dp-size "${DP_SIZE}" \
        --effective-batch "${EFFECTIVE_BATCH}"
fi

extra_args=()
if [[ "${DISABLE_WANDB}" == "1" ]]; then
    extra_args+=(--disable-wandb)
fi

"${TORCHRUN}" \
    "${torchrun_rendezvous_args[@]}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train.py \
    --config_path "${CONFIG_OVERRIDE}" \
    --output-dir "${ARTIFACT_DIR}" \
    --metrics-path "${METRICS_LOG}" \
    --wandb-save-dir "${WANDB_DIR}" \
    "${extra_args[@]}"

if [[ ( "${SPARSE_METHOD}" == "sla_cag" || "${SPARSE_METHOD}" == "hsa_sla_cag" ) \
    && "${VALIDATE_LINEAR_CHECKPOINT}" == "1" ]] \
    && (( NODE_RANK == 0 )); then
    FINAL_CHECKPOINT_DIR="${ARTIFACT_DIR}/checkpoints/step_$(printf '%07d' "${MAX_ITERS}")"
    echo "[validate] SLA final checkpoint=${FINAL_CHECKPOINT_DIR}"
    "${PYTHON}" scripts/checkpoints/validate_linear_checkpoint.py \
        "${FINAL_CHECKPOINT_DIR}/train_state.pt" \
        --expected-method "${SPARSE_METHOD}" \
        --expected-step "${MAX_ITERS}" \
        --require-resume-state \
        --json-output "${FINAL_CHECKPOINT_DIR}/validation.json"
    GENERATOR_SIDECAR="${FINAL_CHECKPOINT_DIR}/generator_linear.pt"
    if [[ -f "${FINAL_CHECKPOINT_DIR}/generator_adapter.pt" ]]; then
        GENERATOR_SIDECAR="${FINAL_CHECKPOINT_DIR}/generator_adapter.pt"
    fi
    "${PYTHON}" scripts/checkpoints/validate_linear_checkpoint.py \
        "${GENERATOR_SIDECAR}" \
        --expected-method "${SPARSE_METHOD}" \
        --expected-step "${MAX_ITERS}"
fi
