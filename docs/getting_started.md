# 昇腾环境安装与测试

本文只描述当前维护的昇腾 BF16 路径：环境安装、依赖检查、代码测试、Ascend Triton HSA kernel 烟测和单步训练烟测。训练参数与数据见 [HSA+CAG 训练指南](hsa_cag_training.md)，VBench 与 msprof 见 [推理与评测指南](inference_and_evaluation.md)。

## 1. 已验证的软件栈

必须把 CANN、PyTorch、`torch_npu` 和 Ascend Triton 作为一组管理。当前已验证组合为：

```text
架构：aarch64
Python：3.11
PyTorch：2.9.0+cpu
torch_npu：2.9.0
CANN：8.5 系列
triton-ascend：3.2.0
```

`torch.__version__` 中的 `+cpu` 在 `torch_npu` 环境里是正常的，设备是否可用以 `torch.npu.is_available()` 为准。不要在同一环境安装 CUDA PyTorch、官方 GPU Triton、FlashAttention、TransformerEngine、FourOverSix 或 NVFP4 依赖。

## 2. 目录和权重前置条件

默认脚本假设：

```text
/mnt/share/r50063443/LongLive-oneday                 # 仓库
/mnt/share/r50063443/conda_envs/longlive             # 生成/训练环境
/mnt/share/weight/Wan2.2-TI2V-5B                     # Wan 模型根目录
/mnt/share/weight/LongLive/checkpoints/longlive2_5b/
  longlive2_merged_generator.pt                      # LongLive 基础 Generator
```

Wan 模型根目录至少应包含：

```text
models_t5_umt5-xxl-enc-bf16.pth
google/umt5-xxl/
Wan2.2_VAE.pth
```

路径不同不需要改源码。训练通过 `LONGLIVE_ROOT`、`GENERATION_ENV`、`MODEL_ROOT`、`GENERATOR_CKPT` 覆盖；推理通过 `GENERATION_ENV`、`LONGLIVE_MODEL_ROOT`、`LONGLIVE_GENERATOR_CKPT` 覆盖。

## 3. 激活 CANN 和 Python 环境

使用与当前 `torch_npu` 匹配的 CANN 8.5，不要在未验证的情况下切到其他 CANN 主版本：

```bash
conda activate /mnt/share/r50063443/conda_envs/longlive
source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
cd /mnt/share/r50063443/LongLive-oneday
```

检查设备和版本：

```bash
npu-smi info

python - <<'PY'
import platform
import torch
import torch_npu

print("machine:", platform.machine())
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
PY
```

## 4. 安装项目依赖

先按昇腾官方兼容矩阵安装 `torch`、`torchvision` 和 `torch_npu`，再安装仓库依赖：

```bash
pip install -r requirements_npu_bf16.txt
```

若 Triton 曾经混装，清理后安装固定版本：

```bash
pip uninstall -y triton triton-ascend triton_ascend
rm -rf "${CONDA_PREFIX}/lib/python3.11/site-packages/triton" \
       "${CONDA_PREFIX}/lib/python3.11/site-packages/triton-"*.dist-info \
       "${CONDA_PREFIX}/lib/python3.11/site-packages/triton_ascend-"*.dist-info
pip install triton-ascend==3.2.0 \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple \
  --trusted-host triton-ascend.osinfra.cn
```

只有服务器证书链确实需要时才使用 `--trusted-host`；更规范的处理是安装正确 CA 证书。确认环境里只有一套 Triton：

```bash
python - <<'PY'
import triton
print("triton:", triton.__version__)
print("module:", triton.__file__)
PY
pip list | grep -i triton
```

## 5. 不占用 NPU 的代码检查

以下检查适合提交前运行，不代表 NPU 端到端已经通过：

```bash
python -m compileall -q \
  train.py inference_sp.py model pipeline trainer utils wan_5b scripts tests
find scripts -type f -name '*.sh' -exec bash -n {} +
PYTHONPATH=. python -m pytest -q tests
```

配置解析检查：

```bash
python - <<'PY'
from omegaconf import OmegaConf

for path in (
    "configs/train/hsa_cag.yaml",
    "configs/inference/vbench.yaml",
    "configs/inference/msprof.yaml",
):
    OmegaConf.load(path)
    print(path, "ok")
PY
```

仅展开 VBench 配置、但不启动生成：

```bash
python scripts/evaluation/resolve_config.py vbench \
  --config configs/inference/vbench.yaml \
  --preset longlive2_standard_5pct \
  --seed 0 \
  --output /tmp/longlive_vbench_resolved.yaml
```

加入 `LONGLIVE_SPARSE_METHOD=hsa_cag` 后，输出的
`model_kwargs.sparse_config` 必须包含 `enabled: true`、`backend: mindiesd`
和 `block_q/block_k: 128`。

## 6. Ascend Triton 训练 kernel 烟测

正式后端必须先通过前向和反向正确性测试。首次运行会编译 kernel，可能需要几十秒：

```bash
ASCEND_RT_VISIBLE_DEVICES=15 PYTHONUNBUFFERED=1 timeout 20m \
python tests/npu/ascend_hsa_kernel_smoke.py --device npu:0
```

脚本会显示以下阶段：

```text
[smoke] checking Ascend Triton availability
[smoke] launching Triton forward (first run compiles the kernel)
forward max_abs=... mean_abs=...
[smoke] launching Triton backward (first run compiles backward kernels)
dq max_abs=... mean_abs=...
dk max_abs=... mean_abs=...
dv max_abs=... mean_abs=...
Ascend Triton HSA smoke test passed
```

代码阈值是前向 `max_abs <= 0.05, mean_abs <= 0.01`，反向 `max_abs <= 0.08, mean_abs <= 0.015`。只检查前向可运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=15 PYTHONUNBUFFERED=1 timeout 20m \
python tests/npu/ascend_hsa_kernel_smoke.py --device npu:0 --forward-only
```

## 7. 单步训练环境烟测

先按训练文档准备提示词，再用 12 卡 `SP4 x DP3` 跑一个 optimizer step：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 TRAIN_RUN_NAME=hsa_cag_smoke \
VIS_INTERVAL=0 \
bash scripts/training/run_hsa_cag.sh
```

成功标准不是只完成模型加载，而是终端出现 `step 1` 的 loss/grad norm，且生成：

```text
runs/training/hsa_cag_smoke/config.source.yaml
runs/training/hsa_cag_smoke/config.resolved.yaml
runs/training/hsa_cag_smoke/manifest.json
logs/training/hsa_cag_smoke/node_0.log
logs/training/hsa_cag_smoke/metrics.jsonl
```

`SAVE_INTERVAL` 默认是 10，所以单步烟测不会产生 checkpoint；需要验证保存时显式设置 `SAVE_INTERVAL=1 MAX_CHECKPOINTS=1`。

## 8. VBench 和 msprof 环境检查

VBench 还需要独立 AISBench 环境和本地 VBench 模型缓存：

```bash
test -x /path/to/longlive-env/bin/torchrun
test -x /path/to/aisbench-env/bin/ais_bench
test -d /path/to/vbench_models
```

msprof 运行前检查：

```bash
msprof --help
msprof-analyze --help
```

完整启动命令和输出目录见推理文档。不要用一次 msprof 采集同时代表无插桩的生产吞吐，msprof 本身会引入开销。

## 9. 常见故障

### 9.1 异步 NPU 报错定位

Python 栈可能只指向下一次同步位置。仅在复现单步错误时启用：

```bash
ASCEND_LAUNCH_BLOCKING=1 MAX_ITERS=1 VIS_INTERVAL=0 \
bash scripts/training/run_hsa_cag.sh
```

定位后取消，避免严重拖慢训练：

```bash
unset ASCEND_LAUNCH_BLOCKING
```

错误码 `107020` 常表示设备任务或跨 rank 等待超时。应先检查所有 `node_<rank>.log` 中最早出现的算子错误；后续 BrokenPipe、semaphore 泄漏和其他 rank 超时通常是连带结果。

### 9.2 Ascend Triton 不可用

训练后端不可用时，检查 `triton.__file__`、CANN 环境、`torch_npu` 版本和
`wan_5b.modules.sparse_attention_ascend.ascend_triton_unavailable_reason()`。
训练使用 `ascend_triton`，不会静默退回 portable。

正式稀疏推理使用 MindIE-SD 3.0.0 的 `RainFusionAttention`。虽然 3.0.0
通用安装页把 CANN 8.0.0 列为包级基线，但本路径依赖的
`aclnnRainFusionAttention` 从 ops-transformer 8.5.0 起提供；因此实际最低
使用 CANN 8.5.0，并确保安装匹配版本的 ops 包。华为提供的 MindIE-SD 3.0.0
镜像采用 CANN 8.5.1 + torch/torch_npu 2.9.0，这是新建环境的推荐组合。
不要安装当前 `dev` 分支；它要求 CANN 9.0.1。安装稳定版：

```bash
pip install --trusted-host ascend.devcloud.huaweicloud.com \
  -i https://ascend.devcloud.huaweicloud.com/pypi/simple/ mindiesd==3.0.0
```

安装后检查：

```bash
python - <<'PY'
from wan_5b.modules.sparse_attention_mindiesd import (
    mindiesd_available, mindiesd_unavailable_reason,
)
print("available:", mindiesd_available())
print("detail:", mindiesd_unavailable_reason())
PY
```

`available: false` 时不要运行 HSA VBench；启动入口也会在加载 5B 模型前失败。

安装完成后先验证 BF16、矩形 Q/KV 和完整 LUT 与 dense attention 数值一致：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python tests/npu/mindiesd_hsa_kernel_smoke.py --device npu:0 --dtype bf16
```

再运行真实 SP4 尾部形状基准；只有 `full_ms` 小于 dense 延迟才进入 VBench：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python tests/npu/benchmark_hsa_inference.py --device npu:0
```

### 9.3 分布式启动失败

确认可见设备数等于 `NPROC_PER_NODE`，`WORLD_SIZE` 能整除 `SP_SIZE`，且 `SP_SIZE` 同时整除 24 个 attention head 和每块 8 个 latent 帧。当前合法 SP 为 `1/2/4/8`。
