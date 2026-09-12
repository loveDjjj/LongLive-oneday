# 当前测试清单

本轮验证同一套入口在单机 **4 张 H100 80GB** 上的 BF16 推理与稀疏后训练。CUDA 数值、反向、四卡 FSDP、完整训练、恢复、视频与 VBench 全部**待服务器验证**；NPU 回归也**待服务器验证**。本地主机测试不能替代硬件验收。环境安装见[环境安装与测试](setup_and_validation.md)，方法契约见[训练指南](training.md)和[推理与评测指南](inference_and_evaluation.md)。

本地回归：macOS/CPU PyTorch 2.13.0 下 **332 passed，2 CUDA tests skipped，3 subtests passed**；Shell/YAML/编译/空白及本地Markdown链接检查通过。CPU BF16 portable烟测的三方法Q/K/V/补偿梯度和未选KV审计通过；隔离环境的真实PEFT小模型CPU前后向通过。上述结果不包含CUDA编译、NCCL或FSDP硬件执行。

## 1. 环境与路径

在仓库根目录执行，先按环境文档安装 CUDA 生成环境。将模型路径改为服务器实际目录；不得把训练增量 `train_state.pt` 当成完整 Generator。

在当前 `feat/unified-sparse-attention` 分支更新服务器代码：

```bash
git pull --ff-only origin feat/unified-sparse-attention
```

```bash
conda activate longlive_cuda
export LLV2_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GENERATION_ENV="${CONDA_PREFIX}"
export LONGLIVE_MODEL_ROOT=/path/to/Wan2.2-TI2V-5B
export LONGLIVE_GENERATOR_CKPT=/path/to/longlive2_merged_generator.pt
export MODEL_ROOT="${LONGLIVE_MODEL_ROOT}"
export GENERATOR_CKPT="${LONGLIVE_GENERATOR_CKPT}"
export TRAIN_PROMPTS="${PWD}/data/train/vidprom_filtered_extended/prompts_train.txt"
export test_id="h100-$(date +%Y%m%d-%H%M%S)"
mkdir -p "logs/tests/${test_id}"
set -o pipefail

nvidia-smi
nvidia-smi topo -m
python - <<'PY'
import torch
print('torch=', torch.__version__, 'cuda=', torch.version.cuda)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
assert torch.cuda.is_bf16_supported()
for index in range(4):
    prop = torch.cuda.get_device_properties(index)
    print(index, prop.name, 'GiB=', prop.total_memory / 2**30)
PY
```

记录驱动、GPU互联、每卡容量和主机RAM。准备训练提示词后再执行第5节；不自动下载模型或训练数据。

## 2. 主机回归与配置展开

此处通过隐藏CUDA只执行主机检查；GPU tests被跳过是预期行为。

```bash
LLV2_DEVICE=auto CUDA_VISIBLE_DEVICES="" python -m pytest -q
python -m compileall -q train.py inference_sp.py pipeline model trainer utils wan_5b scripts tests
git diff --check

DRY_RUN=1 LONGLIVE_SP_SIZE=4 BENCHMARK_MODE=dit_only \
bash scripts/evaluation/run_benchmark.sh 5s

DRY_RUN=1 LONGLIVE_SPARSE_METHOD=hsa_sla_cag LONGLIVE_SPARSE_BACKEND=cuda_flex \
bash scripts/evaluation/run_vbench.sh longlive2_standard_5pct

DRY_RUN=1 METHODS=dense,hsa_cag,sla_cag,hsa_sla_cag DURATIONS=5s \
MODES=dit_only,sync_vae LONGLIVE_SP_SIZE=4 PERF_DEVICES=0,1,2,3 \
bash scripts/evaluation/run_performance_matrix.sh

DRY_RUN=1 TRAIN_RUN_NAME="${test_id}-train-dry" \
bash scripts/training/run_hsa_sla_cag.sh
```

预期：主机测试通过；推理dry-run在唯一临时目录保存resolved/manifest，包含CUDA设备与SP4/DP1且不要求CANN；性能矩阵展开8个case；训练resolved位于 `runs/training/${test_id}-train-dry/`，使用SP4、累积8、BF16底座、FP32训练参数、`portable` 后端。同名训练dry-run必须拒绝覆盖，不能将其目录作为正式训练运行名。

## 3. 单卡数值与完整反向

先小形状后真实SP4形状；首次FlexAttention编译可能耗时数分钟。本节不发布性能数据。

```bash
python tests/cuda/sparse_attention_smoke.py --device cuda:0 --backend portable \
  | tee "logs/tests/${test_id}/portable-small.txt"
python tests/cuda/sparse_attention_smoke.py --device cuda:0 --backend cuda_flex \
  | tee "logs/tests/${test_id}/cuda-flex-small.txt"
python tests/cuda/sparse_attention_smoke.py --device cuda:0 --backend cuda_flex --real-shape \
  | tee "logs/tests/${test_id}/cuda-flex-real.txt"
```

预期：小形状与独立dense mask参考的输出和Q/K/V梯度误差通过；真实 `Q=7040/KV=28160/H=6/D=128` 与selected-KV参考一致；出现 `softmax_backward=passed`、`unselected_kv_audit=passed` 和三种方法各自的 `training_backward=passed`。SLA/混合方法同时检查投影weight/bias梯度；未选KV扰动不变性只约束sparse softmax，不约束全KV线性补偿。

## 4. 四卡通信与FSDP小模型

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  tests/cuda/distributed_smoke.py \
  | tee "logs/tests/${test_id}/distributed.txt"
```

预期：`ulysses_backward=passed mixed_dtype_fsdp=passed optimizer_resume=passed`，验证真实NCCL all-to-all数据/梯度交换、BF16底座+FP32 LoRA/原始补偿层的FSDP、optimizer更新和小checkpoint恢复。此测试不加载5B，不能替代整网显存验收。

## 5. 完整训练与恢复

第3–4节通过后，运行混合方法完整一步。保留32帧、窗口32、128-token block、默认训练范围与稀疏预算。

```bash
NPROC_PER_NODE=4 LONGLIVE_SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
SPARSE_BACKEND=cuda_flex MAX_ITERS=1 SAVE_INTERVAL=1 MAX_CHECKPOINTS=2 VIS_INTERVAL=0 \
TRAIN_RUN_NAME="${test_id}-hybrid-smoke" \
bash scripts/training/run_hsa_sla_cag.sh

NPROC_PER_NODE=4 LONGLIVE_SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
SPARSE_BACKEND=cuda_flex MAX_ITERS=2 SAVE_INTERVAL=1 MAX_CHECKPOINTS=2 VIS_INTERVAL=0 \
TRAIN_RUN_NAME="${test_id}-hybrid-smoke" \
bash scripts/training/run_hsa_sla_cag.sh
```

预期：第一条完成Generator/Critic更新，loss/梯度有限，60个原始线性张量梯度有效且checkpoint验证通过。第二条从step1恢复到step2，不能从0重跑。记录所有rank峰值显存和主机加载峰值。SLA/HSA各自验收时分别更换入口为 `run_sla_cag.sh` / `run_hsa_cag.sh`，使用新方法专属运行名，不共享恢复目录。

排查kernel时可用新的运行名和 `SPARSE_BACKEND=portable`；参考后端可能很慢或占用更多显存。排查底座加载精度时，以新运行名设置 `MODEL_LOAD_DTYPE=float32`。不要直接更改已开始运行的dtype/backend。正式SP4×DP1训练将累积设为8以保持有效batch8；不得根据一次step推断训练速度。

## 6. 四卡推理与无profiler性能

先5秒DiT-only，后同步VAE；固定checkpoint、提示词、seed、设备和采样参数。单项benchmark默认每个DP replica预热1次、有效测量3次。

```bash
LONGLIVE_SP_SIZE=4 LONGLIVE_SPARSE_METHOD=dense BENCHMARK_MODE=dit_only \
RUN_ID="${test_id}-dense-dit-5s" bash scripts/evaluation/run_benchmark.sh 5s

LONGLIVE_SP_SIZE=4 LONGLIVE_SPARSE_METHOD=hsa_sla_cag \
LONGLIVE_SPARSE_BACKEND=cuda_flex BENCHMARK_MODE=dit_only \
RUN_ID="${test_id}-hybrid-dit-5s" bash scripts/evaluation/run_benchmark.sh 5s

LONGLIVE_SP_SIZE=4 LONGLIVE_SPARSE_METHOD=dense BENCHMARK_MODE=sync_vae \
RUN_ID="${test_id}-dense-sync-5s" bash scripts/evaluation/run_benchmark.sh 5s
```

预期：保存latent/可播放视频、resolved、manifest和summary；无CANN/MindIE依赖。不得用5卡的 `SP4+async_vae` 启动4卡机器。短样本通过后运行32秒矩阵，分别报告DiT和端到端p50：

```bash
METHODS=dense,hsa_cag,sla_cag,hsa_sla_cag DURATIONS=32s \
MODES=dit_only,sync_vae LONGLIVE_SP_SIZE=4 PERF_DEVICES=0,1,2,3 \
LONGLIVE_SPARSE_BACKEND=cuda_flex TASK=benchmark SUITE_ID="${test_id}-performance" \
bash scripts/evaluation/run_performance_matrix.sh
```

## 7. CUDA VBench 5%质量

按环境文档准备独立官方评估环境、源码和缓存，再回到生成环境执行。替换三处实际路径；四卡用于生成，官方evaluator逐维在逻辑 `cuda:0` 上运行。

```bash
export VBENCH_ENV=/path/to/longlive_vbench_cuda
export VBENCH_SOURCE_DIR=/path/to/VBench
export VBENCH_CACHE_DIR=/path/to/vbench_models

LONGLIVE_SP_SIZE=4 LONGLIVE_DP_SIZE=1 LONGLIVE_SPARSE_METHOD=dense \
RUN_ID="${test_id}-vbench-dense" \
bash scripts/evaluation/run_vbench.sh longlive2_standard_5pct
```

预期：官方16维、quality/semantic/total和原始逐视频JSON齐全，记录官方VBench源码SHA。后训练质量须比较基础权重dense、训练后同权重dense与匹配sparse三组。继续同run时显式设置 `RESUME_RUN=1`；配置/设备不匹配必须拒绝恢复。

## 8. 结果回填

```text
test_id=
commit=
GPU型号/每卡容量/互联/主机RAM=
torch/torchvision/CUDA/driver=
小形状数值与QKV backward=
真实形状数值与QKV backward=
未选KV审计=
四卡通信/FSDP/optimizer恢复=
完整step1/step2恢复/60个线性张量梯度=
训练和加载峰值内存=
resolved_config_path=
dense/sparse DiT p50与同步VAE端到端p50=
VBench源码SHA/16维/官方汇总=
NPU回归状态=
异常及日志路径=
```
