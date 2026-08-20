# 昇腾环境安装与测试

本文是当前环境与测试的统一入口。训练参数见[训练指南](training.md)，推理、性能和质量评测见[推理与评测指南](inference_and_evaluation.md)。

## 1. 环境基线

当前维护的 BF16 环境为：

```text
架构：aarch64
Python：3.11
PyTorch：2.9.0+cpu
torch_npu：2.9.0
CANN：8.5.x
MindIE-SD：3.0.0
triton-ascend：3.2.0（训练算子）
```

使用 `torch_npu` 时，`torch.__version__` 以 `+cpu` 结尾是正常现象，应通过 `torch.npu.is_available()` 判断 NPU 是否可用。不要在该环境中混装 CUDA PyTorch、GPU Triton、FlashAttention、TransformerEngine、FourOverSix 或 NVFP4 包。

服务器默认路径：

```text
/mnt/share/r50063443/LongLive-oneday
/mnt/share/r50063443/conda_envs/longlive
/mnt/share/r50063443/conda_envs/aisbench_npu
/mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
/mnt/share/r50063443/vbench_models
/mnt/share/r50063443/Wan2.2-TI2V-5B
/mnt/share/r50063443/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt
```

Wan2.2 模型目录至少应包含：

```text
models_t5_umt5-xxl-enc-bf16.pth
google/umt5-xxl/
Wan2.2_VAE.pth
```

## 2. 激活并检查环境

```bash
conda activate /mnt/share/r50063443/conda_envs/longlive
source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
cd /mnt/share/r50063443/LongLive-oneday
npu-smi info
```

```bash
python - <<'PY'
import importlib.metadata
import platform
import torch
import torch_npu

print("machine:", platform.machine())
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu_available:", torch.npu.is_available())
print("npu_count:", torch.npu.device_count())
for package in ("mindiesd", "triton-ascend"):
    try:
        print(package, importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        print(package, "not installed")
PY
```

先安装匹配的 PyTorch、`torch_npu` 和 CANN，再安装仓库依赖：

```bash
pip install -r requirements_npu_bf16.txt
```

需要重装 Ascend Triton 时执行：

```bash
pip uninstall -y triton triton-ascend triton_ascend
pip install triton-ascend==3.2.0 \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
```

MindIE-SD 可能提示其自带的 `SparseLinearAttention` 需要 Triton-Ascend 3.2.1。该警告不能证明本仓库训练算子可用或不可用，训练准入以第 5 节的完整前向与反向测试为准。

## 3. 不占用 NPU 的检查

```bash
python -m compileall -q \
  train.py inference_sp.py pipeline trainer utils wan_5b scripts tests
find scripts -type f -name '*.sh' -exec bash -n {} +
PYTHONPATH=. pytest -q
```

检查 YAML 能否解析：

```bash
python - <<'PY'
from omegaconf import OmegaConf

for path in (
    "configs/train/hsa_cag.yaml",
    "configs/train/sla_cag.yaml",
    "configs/train/hsa_sla_cag.yaml",
    "configs/inference/msprof.yaml",
    "configs/inference/vbench.yaml",
):
    OmegaConf.load(path)
    print(path, "ok")
PY
```

检查性能与 VBench 矩阵展开结果：

```bash
DRY_RUN=1 PERF_DEVICES=0,1,2,3,4 \
bash scripts/evaluation/run_performance_matrix.sh

DRY_RUN=1 VBENCH_PRESETS=longlive2_standard_20pct \
DENSE_GENERATOR_CKPT=/mnt/share/r50063443/LongLive/checkpoints/dense.pt \
HSA_CAG_GENERATOR_CKPT=/mnt/share/r50063443/LongLive/checkpoints/hsa.pt \
SLA_CAG_GENERATOR_CKPT=/mnt/share/r50063443/LongLive/checkpoints/sla.pt \
HSA_SLA_CAG_GENERATOR_CKPT=/mnt/share/r50063443/LongLive/checkpoints/hybrid.pt \
bash scripts/evaluation/run_vbench_matrix.sh
```

这些检查只能验证代码、配置和调度逻辑，不能证明 NPU 数值正确性或性能收益。

## 4. MindIE-SD 推理算子

先验证小尺寸矩形 RainFusion：

```bash
test_id="npu-acceptance-$(date +%Y%m%d-%H%M%S)"
mkdir -p "logs/tests/${test_id}"
ASCEND_RT_VISIBLE_DEVICES=15 \
python tests/npu/mindiesd_sla_kernel_smoke.py \
  --device npu:0 --dtype bf16 --audit-unselected-kv \
  | tee "logs/tests/${test_id}/mindiesd-rainfusion-smoke.txt"
```

输出除数值误差通过外，还必须包含 `unselected_kv_audit=passed`。该审计只扰动 LUT 未选中的 K/V blocks：稀疏 softmax 输出必须保持不变，dense 对照必须发生变化。

再测试三种稀疏方法的真实 SP4、32 秒尾部形状：

```bash
for method in hsa_cag sla_cag hsa_sla_cag; do
  ASCEND_RT_VISIBLE_DEVICES=15 \
  python tests/npu/benchmark_sparse_attention.py \
    --method "${method}" --backend mindiesd --device npu:0 \
    --latent-frames 192 --warmup 5 --iterations 20 \
    | tee "logs/tests/${test_id}/${method}-mindiesd-32s.txt"
done
```

重点检查 `selected/total`、有效稀疏率、路由耗时、kernel 耗时、完整稀疏注意力耗时和相对 dense 加速。微基准加速只是必要条件，发布结论仍以 DiT-only 整网测试为准。

`mindiesd_bsa` 目前属于实验后端。若 `libopapi.so` 缺少 `aclnnBlockSparseAttentionV2`，重复加载同一 CANN 环境不会解决问题；只有完整 CANN/MindIE-SD 软件栈提供并注册该算子后，才能继续验证 BSA。当前默认推理后端保持 RainFusion。

## 5. Ascend Triton 训练算子

先执行小尺寸数值与反向对照：

```bash
ASCEND_RT_VISIBLE_DEVICES=15 PYTHONUNBUFFERED=1 timeout 20m \
python tests/npu/ascend_sla_kernel_smoke.py --device npu:0
```

期望最后输出：

```text
Ascend Triton SLA sparse smoke test passed
```

再验证混合方法在真实训练形状下的 Q/K/V 与投影反向：

```bash
ASCEND_RT_VISIBLE_DEVICES=15 \
python tests/npu/benchmark_sparse_attention.py \
  --method hsa_sla_cag --backend ascend_triton --device npu:0 \
  --latent-frames 32 --warmup 1 --iterations 1 \
  --check-training-backward \
  | tee "logs/tests/${test_id}/hybrid-training-backward.txt"
```

只有输出包含 `training_backward=passed`，才能启动多卡混合训练。`--check-linear-backward` 只验证本地线性投影可求导，不能替代稀疏 softmax 的 Q/K/V 反向检查。

## 6. 分布式训练烟测

准备数据后运行一个 optimizer step：

```bash
bash scripts/data/prepare_training_data.sh

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 SAVE_INTERVAL=1 MAX_CHECKPOINTS=1 VIS_INTERVAL=0 \
TRAIN_RUN_NAME=hsa_sla_cag_12card_smoke \
bash scripts/training/run_hsa_sla_cag.sh
```

SLA+CAG 或混合 `lora_plus_linear` 训练烟测需同时满足：

- Generator loss、Critic loss 和梯度范数均为有限值。
- `linear_grad_tensors/with_gradient/nonzero/finite` 均为 60。
- SLA+CAG 与混合主方案生成 `step_0000001/train_state.pt` 和 `generator_adapter.pt`；linear-only 对照生成 `generator_linear.pt`。
- 日志出现两次 `linear_checkpoint=passed`。
- 生成 `step_0000001/validation.json`。

## 7. 推理验收顺序

1. 先运行 5 秒与 32 秒 `dit_only`，要求稀疏方法的 p50 小于相同 checkpoint 的 dense p50。
2. 32 秒至少执行 1 次预热和 3 次有效测量，禁止用单次结果下结论。
3. 再运行 `sync_vae` 和 `async_vae`，确认端到端关键路径是否被 VAE 覆盖。
4. 短视频稳定后再扩展到 64 秒。
5. VBench 对比必须固定 checkpoint、提示词、seed、帧数、分辨率和采样参数。
6. 无 profiler 延迟稳定后再使用 msprof 定位算子。

具体命令与指标解释见[推理与评测指南](inference_and_evaluation.md)。

## 8. 常见故障

异步 NPU 报错时，可仅对单步复现临时开启阻塞：

```bash
ASCEND_LAUNCH_BLOCKING=1 MAX_ITERS=1 VIS_INTERVAL=0 \
bash scripts/training/run_hsa_sla_cag.sh
unset ASCEND_LAUNCH_BLOCKING
```

优先查找所有 rank 中最早出现的算子错误。某个 rank 先失败后出现的 broken pipe、semaphore warning 和 HCCL timeout 通常只是后续现象。

分布式启动至少满足：

```text
可见 NPU 数量 == NPROC_PER_NODE
WORLD_SIZE % SP_SIZE == 0
24 个 attention head % SP_SIZE == 0
每个 chunk 的 8 个 latent 帧 % SP_SIZE == 0
```

当前支持 `SP_SIZE=1/2/4/8`。多节点必须共享代码、模型、数据和运行目录，并使用一致的 `MASTER_ADDR` 与 `TRAIN_RUN_NAME`。先启动 rank 0，由系统动态分配 rendezvous 端口并写入共享 run 目录；其他节点自动读取，不设置 `MASTER_PORT`。
