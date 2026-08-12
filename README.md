# LongLive2.0 昇腾稀疏注意力

本仓库维护 LongLive2.0-5B 在昇腾 NPU 上的 BF16 训练、推理和评测，并在同一套模型与评测入口中支持四种注意力方法：

- `dense`：LongLive2.0 原生滚动 KV 稠密注意力。
- `hsa_cag`：参考 Light Forcing，先按 latent 帧筛选历史，再执行块稀疏注意力与 CAG。
- `sla_cag`：对滚动 KV 全局执行 Smooth-K 块路由，并加入线性补偿与 CAG。
- `hsa_sla_cag`：先用 HSA 筛选候选帧，再用 SLA 选择 token block，以 CAG 控制预算，并使用全 KV 线性补偿。

三种稀疏方法使用独立训练配置。`hsa_cag` 和 `sla_cag` 训练 Generator LoRA；`hsa_sla_cag` 冻结 Generator 主干，只训练 30 层 `sla_linear` 投影，DMD Fake Critic 仍训练 LoRA。

## 支持范围

- 基础模型：Wan2.2-TI2V-5B 与 LongLive2.0 合并 Generator。
- 训练：提示词驱动 DMD，32 个 latent 帧，每个 AR chunk 8 帧，4 步采样，FSDP + Ulysses SP + DP。
- 训练稀疏算子：`ascend_triton`，必须支持 Q/K/V 反向。
- 推理稀疏算子：MindIE-SD `RainFusionAttention`。
- 推理模式：仅 DiT、同步 VAE、独立 NPU 异步 VAE。
- 性能时长：5 秒、32 秒和 64 秒。
- 评测：无 profiler 延迟、VBench Standard、msprof 与 VAE-only 分析。

当前昇腾版本不维护 CUDA/Hopper、NVFP4、FourOverSix 和原生非 SP 部署路径。

## 快速开始

安装环境并执行分层验收：

```bash
conda activate /mnt/share/r50063443/conda_envs/longlive
source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
cd /mnt/share/r50063443/LongLive-oneday
PYTHONPATH=. pytest -q
```

准备训练提示词：

```bash
bash scripts/data/prepare_training_data.sh
```

启动多卡训练前，先验证混合方法的完整 Q/K/V 反向：

```bash
ASCEND_RT_VISIBLE_DEVICES=15 \
python tests/npu/benchmark_sparse_attention.py \
  --method hsa_sla_cag --backend ascend_triton --device npu:0 \
  --latent-frames 32 --warmup 1 --iterations 1 \
  --check-training-backward
```

输出必须包含 `training_backward=passed`。

三种训练入口：

```bash
bash scripts/training/run_hsa_cag.sh
bash scripts/training/run_sla_cag.sh
bash scripts/training/run_hsa_sla_cag.sh
```

运行单个 32 秒 DiT-only 性能用例：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
LONGLIVE_SPARSE_METHOD=hsa_sla_cag BENCHMARK_MODE=dit_only \
RUN_ID=hybrid-dit-32s \
bash scripts/evaluation/run_benchmark.sh 32s
```

检查完整性能矩阵，不占用 NPU：

```bash
DRY_RUN=1 PERF_DEVICES=0,1,2,3,4 \
bash scripts/evaluation/run_performance_matrix.sh
```

## 文档

- [环境安装与测试](docs/setup_and_validation.md)
- [训练指南](docs/training.md)
- [推理与评测指南](docs/inference_and_evaluation.md)
- [参考论文索引](reference/README.md)

## 运行产物

```text
runs/training/<run-id>/       配置、manifest 和 checkpoint
runs/performance/<run-id>/    无 profiler 视频或 latent 与延迟汇总
runs/msprof/dit/<run-id>/     DiT/生成过程 profiler 数据
runs/msprof/vae/<run-id>/     VAE-only profiler 数据
runs/vbench/<run-id>/         生成视频与 AISBench 结果
runs/suites/<suite-id>/       可比较的 CSV/JSON 矩阵汇总
logs/<task>/<run-id>/         与 runs 任务层级对应的文本日志
logs/tests/<test-id>/         smoke、microbenchmark 和临时测试日志
```

本项目基于 [NVlabs/LongLive](https://github.com/NVlabs/LongLive) 开发，并保留适用的上游许可证和版权声明。
