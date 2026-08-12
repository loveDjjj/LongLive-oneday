# LongLive2.0 昇腾稀疏训练指南

本文是当前训练流程的统一说明，覆盖 `hsa_cag`、`sla_cag` 和 `hsa_sla_cag` 三种方法。环境和算子准入见[环境安装与测试](setup_and_validation.md)，推理、性能与质量评测见[推理与评测指南](inference_and_evaluation.md)。

## 1. 训练目标与边界

当前训练是 LongLive2.0-5B 的提示词驱动稀疏 DMD 后训练：

1. Generator 从噪声在线生成 32 个 latent 帧。
2. 32 帧按每个 AR chunk 8 帧划分为 4 个因果块，使用 4 步采样。
3. Dense Real Teacher 和 Dense Fake Critic 对 Generator 中间状态计算 score。
4. Generator 使用指定稀疏方法，Teacher 与 Critic 保持 dense，避免监督目标同时引入稀疏偏差。
5. 当前不读取真实视频，不维护 I2V、teacher forcing、NVFP4 或非 causal 训练路径。

三种配置的核心差异：

| 方法 | Generator 训练范围 | 默认目标/基准稀疏率 | 主要路由 |
| --- | --- | --- | --- |
| `hsa_cag` | LoRA | 0.85 / 0.95 | 帧级 HSA，保留 6 帧，当前 chunk 稠密 |
| `sla_cag` | LoRA | 0.90 / 0.93 | 对滚动 KV 全局执行 Smooth-K block Top-K |
| `hsa_sla_cag` | 仅 `sla_linear` | 0.90 / 0.93 | HSA 选 8 个候选帧，SLA 再选 block |

Fake Critic 在三种方法中均使用 LoRA。混合方法的 Generator 主干冻结，只训练 30 层 `sla_linear` 的 weight/bias，共 60 个张量。

当前 SLA+CAG 与混合方法统一使用 `0.90/0.93` CAG 预算。旧 SLA checkpoint 若使用 `0.95/0.97` 训练，其训练分布没有因配置更新而改变：可以先用新预算做算子和 DiT 性能探索，但正式质量结论必须重新训练或继续微调，并在 manifest 中记录实际训练预算。

## 2. 稀疏行为

LongLive2.0 最多保留 32 个 latent 帧的滚动 KV。每帧在 patch embedding 后对应 `22 x 40 = 880` 个 token；SP4 下 attention 侧每个 rank 看到完整序列和 `24 / 4 = 6` 个 head。

### 2.1 HSA+CAG

- HSA 先在 latent 帧层面选择历史 KV，默认保留 1 个 sink 帧、2 个相邻帧和动态选择帧，共 6 帧。
- 当前 8 帧 chunk 默认保持稠密，因此长序列理论稀疏上限会受到当前块占比限制。
- CAG 根据 AR rollout 位置改变稀疏预算；没有足够历史时回退 dense。
- 正式配置使用 40-token block，与 Light Forcing 风格的帧优先路由保持一致。

### 2.2 SLA+CAG

- 不先裁剪 latent 帧，而是对当前滚动 KV 的 128-token blocks 全局打分并 Top-K。
- 强制保留 sink 与最近帧对应的 block。
- 默认 `dense_current=false`，避免当前 8/32 帧将理论稀疏率限制在 75%。
- 线性注意力分支使用完整 KV 统计量，补偿稀疏 softmax 丢失的信息。

### 2.3 HSA+SLA+CAG

- HSA 先保留 sink 1 帧、recent 1 帧，并按重要度补足到 8 个候选帧。
- SLA 只在候选帧的 blocks 内执行 Smooth-K Top-K，而不是将候选帧全部计算。
- CAG 控制每个 AR chunk 的最终 block 预算，线性补偿仍使用完整 KV。
- SP4、32 秒尾部形状为 `Q=7040`、`KV=28160`，当前配置通常选择约 20 至 22 个/220 个 KV blocks，即约 90% 的有效稀疏率；最终值以运行日志为准。

路由索引本身不可微，但选中 block 的 Q/K/V 必须保持梯度。训练只支持通过完整反向测试的 `ascend_triton` 后端；MindIE-SD RainFusion/BSA 是推理前向算子。

## 3. 训练数据

训练输入是 UTF-8 文本，每个非空行是一条提示词。运行：

```bash
bash scripts/data/prepare_training_data.sh
```

也可指定本地源文件：

```bash
SOURCE_FILE=/path/to/vidprom_filtered_extended.txt \
bash scripts/data/prepare_training_data.sh
```

只有明确允许联网时才设置：

```bash
ALLOW_DOWNLOAD=1 bash scripts/data/prepare_training_data.sh
```

脚本会规范化、去重，并排除与完整 VBench Standard/Augmented 重合的提示词。默认产物：

```text
data/train/vidprom_filtered_extended/
├── prompts_train.txt
└── manifest.json
```

详细数量、SHA256 与数据许可见[data/train/README.md](../data/train/README.md)。

## 4. 并行布局与有效 batch

```text
world_size = NNODES x NPROC_PER_NODE
DP = world_size / SP_SIZE
有效 batch = DP x batch_size x GRADIENT_ACCUMULATION_STEPS
```

当前 `batch_size=1`。模型有 24 个 attention head，每个 chunk 有 8 个 latent 帧，因此合法的 `SP_SIZE` 为 `1/2/4/8`。例如 12 卡使用 SP4 x DP3、梯度累积 16 时，有效 batch 为 48。

一个 step 表示一次 optimizer 更新，不是一条提示词或一个时间块。增加 SP 主要改变单样本分片和通信，增加 DP 才直接提高样本吞吐。跨节点 SP 会增加 HCCL 开销，推荐让每个 SP 组位于单一节点内。

## 5. 启动前准入

所有多卡训练先执行主机测试和训练 kernel 的完整反向检查：

```bash
PYTHONPATH=. pytest -q

ASCEND_RT_VISIBLE_DEVICES=15 \
python tests/npu/benchmark_sparse_attention.py \
  --method hsa_sla_cag --backend ascend_triton --device npu:0 \
  --latent-frames 32 --warmup 1 --iterations 1 \
  --check-training-backward
```

输出必须包含 `training_backward=passed`。`--check-linear-backward` 只证明补偿投影可求导，不能证明多层训练需要的稀疏 softmax Q/K/V 反向可用。

## 6. 配置与启动变量

三份源配置：

```text
configs/train/hsa_cag.yaml
configs/train/sla_cag.yaml
configs/train/hsa_sla_cag.yaml
```

三个入口都委托给 `scripts/training/run_sparse_cag.sh`。启动器将环境变量覆盖写入 `runs/training/<run-name>/config.resolved.yaml`，训练进程只读取 resolved 配置。

常用变量：

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES` | 当前节点可见 NPU | `0,...,15` |
| `NPROC_PER_NODE` | 每节点 worker 数 | 16 |
| `NNODES` / `NODE_RANK` | 节点数 / 当前节点编号 | 1 / 0 |
| `MASTER_ADDR` | 多节点 rendezvous 所在的 rank 0 地址；单节点忽略 | `127.0.0.1` |
| `SP_SIZE` | Ulysses SP 大小 | 4 |
| `GRADIENT_ACCUMULATION_STEPS` | 梯度累积次数 | 16 |
| `MAX_ITERS` | 训练结束目标 step | 2000 |
| `SAVE_INTERVAL` | checkpoint 间隔 | 10 |
| `VIS_INTERVAL` | 训练内验证间隔，0 关闭 | 100 |
| `MAX_CHECKPOINTS` | 最多保留 checkpoint 数 | 20 |
| `SPARSE_BACKEND` | 训练稀疏后端 | `ascend_triton` |
| `TRAIN_RUN_NAME` | 运行目录与自动恢复标识 | 时间戳名称 |
| `DISABLE_WANDB` | 1 表示禁用 W&B | 1 |

路径可通过 `LONGLIVE_ROOT`、`GENERATION_ENV`、`MODEL_ROOT`、`GENERATOR_CKPT`、`TRAIN_PROMPTS` 和 `CONFIG_PATH` 覆盖。

## 7. 训练命令

12 卡单步混合烟测：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 SAVE_INTERVAL=1 MAX_CHECKPOINTS=1 VIS_INTERVAL=0 \
TRAIN_RUN_NAME=hsa_sla_cag_12card_smoke \
bash scripts/training/run_hsa_sla_cag.sh
```

12 卡、1000 step 混合训练：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=16 \
MAX_ITERS=1000 SAVE_INTERVAL=10 MAX_CHECKPOINTS=20 VIS_INTERVAL=100 \
TRAIN_RUN_NAME=hsa_sla_cag_12card_1k \
bash scripts/training/run_hsa_sla_cag.sh
```

原生 HSA 或 SLA 只需更换入口与运行名：

```bash
bash scripts/training/run_hsa_cag.sh
bash scripts/training/run_sla_cag.sh
```

双节点 24 卡示例，两个节点使用相同共享路径、运行名和 rank 0 地址。先启动节点 0；节点 0 由系统动态分配空闲端口并写入共享 run 目录，节点 1 自动读取，不设置 `MASTER_PORT`。节点 0：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=12 \
MASTER_ADDR=10.0.0.10 \
SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=8 MAX_ITERS=1000 \
TRAIN_RUN_NAME=hsa_sla_cag_24card_1k \
bash scripts/training/run_hsa_sla_cag.sh
```

节点 1 使用同一命令，仅改为 `NODE_RANK=1`。

## 8. 日志与梯度验收

```text
runs/training/<run-name>/
├── config.source.yaml
├── config.resolved.yaml
├── manifest.json
├── checkpoints/step_*/
└── wandb/

logs/training/<run-name>/
├── node_<rank>.log
└── metrics.jsonl
```

混合训练每次 Generator 更新都会记录：

```text
linear_grad_tensors
linear_grad_with_gradient
linear_grad_nonzero
linear_grad_finite
linear_grad_l2_min
linear_grad_l2_max
```

前四项都应为 60，L2 范数应为有限非零值。缺失、全零或非有限梯度会在 optimizer step 前失败。原生 HSA/SLA 则重点检查 Generator LoRA、Critic LoRA 的梯度范数和 loss 是否有限。

## 9. Checkpoint 与恢复

完整恢复包为 `checkpoints/step_*/train_state.pt`，包含 Generator 可训练参数、Critic LoRA、optimizer、scheduler、RNG 和数据游标等状态。使用相同 `TRAIN_RUN_NAME` 会自动扫描并恢复最大 step。

`MAX_ITERS` 是最终目标 step。例如已恢复到 100，设置 `MAX_ITERS=1000` 表示继续到 1000，而不是再训练 1000 步。不同稀疏方法的 checkpoint 不能混用。

混合方法还会写出 `generator_linear.pt`。正常结束时，即使最终 step 不能整除 `SAVE_INTERVAL`，也会补存最终 checkpoint。默认由 node rank 0 验证完整包与 sidecar，并生成 `validation.json`。

手工验证：

```bash
python scripts/checkpoints/validate_linear_checkpoint.py \
  runs/training/hsa_sla_cag_12card_1k/checkpoints/step_0001000/train_state.pt \
  --expected-step 1000 --require-resume-state \
  --json-output runs/training/hsa_sla_cag_12card_1k/checkpoints/step_0001000/validation.json

python scripts/checkpoints/validate_linear_checkpoint.py \
  runs/training/hsa_sla_cag_12card_1k/checkpoints/step_0001000/generator_linear.pt \
  --expected-step 1000
```

## 10. 导出推理权重

原生 HSA/SLA checkpoint 需要把 Generator LoRA 合并到训练时使用的 LongLive2.0 基础 Generator：

```bash
python scripts/checkpoints/merge_lora.py \
  --config_path configs/train/sla_cag.yaml \
  --generator_ckpt /mnt/share/weight/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --lora_ckpt runs/training/sla_cag_12card_2k/checkpoints/step_0002000/train_state.pt \
  --output_path runs/merged/longlive2_sla_cag_2k.pt \
  --device npu:0
```

混合方法使用同一脚本和对应配置；脚本会识别 `linear_only` 训练范围并应用 `generator_linear` sidecar。不要将训练增量合并到原始 Wan2.2 权重，必须使用训练时的 `longlive2_merged_generator.pt`。

导出后先运行同 checkpoint 的 dense 与对应 sparse DiT-only 对照，再运行 VBench。完整流程见[推理与评测指南](inference_and_evaluation.md)。
