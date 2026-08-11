# LongLive2.0 昇腾 SLA+CAG

本仓库维护 LongLive2.0-5B 在昇腾 NPU 上的 BF16 稀疏后训练、推理和评测流程。当前唯一训练目标是提示词驱动的 DMD 后训练：Generator 使用 SLA+CAG 稀疏注意力，Real Teacher 和 Fake Critic 使用稠密注意力，Generator 与 Critic 均通过 LoRA 更新。

## 当前范围

- 模型：`Wan2.2-TI2V-5B` 与 LongLive2.0 合并后的 Generator。
- 训练：纯提示词、32 latent 帧、每块 8 帧、4 步 backward simulation。
- 并行：FSDP + Ulysses SP + DP，支持单机和多机。
- 稀疏后端：训练使用可反向的 `ascend_triton`；推理使用 MindIE-SD
  `RainFusionAttention` 融合算子；`portable` 仅用于正确性对照。
- 推理：BF16 Ulysses SP，支持稠密和 SLA+CAG 对照；SLA 对全部滚动 KV
  执行全局 128-token block Top-K，并强制保留 sink/recent blocks。
- 评测：无 profiler 重复性能基准、VBench Standard 与 msprof。

不再维护源仓库的 AR、I2V、NVFP4、FourOverSix、CUDA/Hopper 和非 SP 推理入口。配置若启用这些能力会在启动阶段直接失败。

## 目录规范

```text
configs/
  train/sla_cag.yaml               # 唯一训练源配置
  inference/                        # VBench 和 msprof 紧凑预设
scripts/                            # 按 data/training/evaluation/checkpoints 分组
data/train/                         # 训练提示词与数据 manifest
runs/                               # checkpoint、配置、manifest、视频和评测产物
logs/                               # 仅保存文本日志和 JSONL 指标
docs/                               # 中文操作文档
```

一次训练运行的产物如下：

```text
runs/training/<run-name>/
  config.source.yaml
  config.resolved.yaml
  manifest.json
  checkpoints/step_0000010/train_state.pt

logs/training/<run-name>/
  node_0.log
  metrics.jsonl
```

## 快速开始

1. 按 [环境安装指南](docs/getting_started.md) 配置 CANN、PyTorch、`torch_npu` 和 Ascend Triton。
2. 准备提示词：

```bash
bash scripts/data/prepare_training_data.sh
```

3. 验证 Ascend Triton 前向和反向：

```bash
ASCEND_RT_VISIBLE_DEVICES=15 PYTHONUNBUFFERED=1 \
python tests/npu/ascend_sla_kernel_smoke.py --device npu:0
```

4. 运行 12 卡单步训练烟测：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 TRAIN_RUN_NAME=sla_cag_smoke \
VIS_INTERVAL=0 \
bash scripts/training/run_sla_cag.sh
```

5. 正式训练时复用同一脚本并明确设置步数和梯度累积。完整命令、批量含义、恢复规则和双节点示例见 [SLA+CAG 训练指南](docs/sla_cag_training.md)。

## 推理与评测

训练 checkpoint 保存的是 LoRA 和恢复状态。先合并 Generator LoRA：

```bash
python scripts/checkpoints/merge_lora.py \
  --config_path configs/train/sla_cag.yaml \
  --generator_ckpt /path/to/longlive2_merged_generator.pt \
  --lora_ckpt runs/training/<run-name>/checkpoints/step_0002000/train_state.pt \
  --output_path runs/merged/longlive2_sla_cag_2k.pt \
  --device npu:0
```

再使用同一权重运行稠密对照和稀疏实验：

```bash
LONGLIVE_GENERATOR_CKPT=runs/merged/longlive2_sla_cag_2k.pt \
RUN_ID=sla_cag_dense \
bash scripts/evaluation/run_vbench.sh longlive2_standard_20pct

LONGLIVE_GENERATOR_CKPT=runs/merged/longlive2_sla_cag_2k.pt \
LONGLIVE_SPARSE_METHOD=sla_cag RUN_ID=sla_cag_sparse \
bash scripts/evaluation/run_vbench.sh longlive2_standard_20pct
```

先运行无 profiler 的三次性能对照：

```bash
LONGLIVE_GENERATOR_CKPT=runs/merged/longlive2_sla_cag_2k.pt \
LONGLIVE_SPARSE_METHOD=dense RUN_ID=dense-32s-3run \
bash scripts/evaluation/run_benchmark.sh 32s

LONGLIVE_GENERATOR_CKPT=runs/merged/longlive2_sla_cag_2k.pt \
LONGLIVE_SPARSE_METHOD=sla_cag \
LONGLIVE_SLA_BACKEND=mindiesd RUN_ID=sla-32s-3run \
bash scripts/evaluation/run_benchmark.sh 32s
```

确认端到端收益后再采集 msprof：

```bash
LONGLIVE_GENERATOR_CKPT=runs/merged/longlive2_sla_cag_2k.pt \
LONGLIVE_SPARSE_METHOD=sla_cag LONGLIVE_SLA_BACKEND=mindiesd \
bash scripts/evaluation/run_msprof.sh 32s
```

推理预设、设备数量和输出格式见 [推理与评测指南](docs/inference_and_evaluation.md)。
当前 SP4 实测 SLA 将 32 秒 latent-only DiT p50 从 `48.656 s` 降至 `36.240 s`
（`1.343x`），但 dedicated VAE 仍主导约 127 秒的完整生成临界路径，因此默认发布
配置继续保持 Dense，直到 VAE 优化后重新通过完整端到端验收。

## 文档

- [环境安装、依赖检查与烟测](docs/getting_started.md)
- [训练、数据集、日志、checkpoint 与恢复](docs/sla_cag_training.md)
- [推理、VBench、msprof 与评测产物](docs/inference_and_evaluation.md)
- [昇腾稀疏注意力上游调研与技术选型](docs/ascend_sparse_attention_design.md)

## 上游项目

本项目基于 NVIDIA LongLive2.0 代码演进，并保留原项目适用的许可证和版权声明。论文与上游实现请参考 [NVlabs/LongLive](https://github.com/NVlabs/LongLive)。
