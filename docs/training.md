# LongLive2.0 稀疏训练指南

本文是当前训练流程的统一说明，覆盖 `hsa_cag`、`sla_cag` 和 `hsa_sla_cag` 三种方法。环境和算子准入见[环境安装与测试](setup_and_validation.md)，推理、性能与质量评测见[推理与评测指南](inference_and_evaluation.md)。

本文既有命令默认面向昇腾；新增 4 张 H100 的 CUDA 适配见第 11 节。CUDA 完整训练仍**待服务器验证**。

## 1. 训练目标与边界

当前训练是 LongLive2.0-5B 的提示词驱动稀疏 DMD 后训练：

1. Generator 从噪声在线生成 32 个 latent 帧。
2. 32 帧按每个 AR chunk 8 帧划分为 4 个因果块，使用 4 步采样。
3. Dense Real Teacher 和 Dense Fake Critic 对 Generator 中间状态计算 score。
4. Generator 使用指定稀疏方法，Teacher 与 Critic 保持 dense，避免监督目标同时引入稀疏偏差。
5. 当前不读取真实视频，不维护 I2V、teacher forcing、NVFP4 或非 causal 训练路径。

三种方法的核心差异：

| 方法 | Generator 训练范围 | 默认目标/基准稀疏率 | 主要路由 |
| --- | --- | --- | --- |
| `hsa_cag` | LoRA | 0.85 / 0.95 | history-only HSA frame router + 全局 block Top-K |
| `sla_cag` | 主干 LoRA + 原始 `sla_linear` | 0.85 / 0.95 | 对完整 resident KV 全局执行 Smooth-K block Top-K |
| `hsa_sla_cag` 主方案 | 主干 LoRA + 原始 `sla_linear` | 0.85 / 0.95 | 共享 history-only HSA router，再执行 SLA global block Top-K |
| `hsa_sla_cag` 对照 | 仅原始 `sla_linear` | 0.85 / 0.95 | 路由与主方案相同，隔离补偿层自身能力 |

Fake Critic 在三种方法中均使用 LoRA。SLA+CAG 与混合主方案的 `sla_linear` 都不包装 LoRA，而是直接训练每层 weight/bias，共 60 个原始张量；Transformer block 内的其余 Linear（包括 self/cross attention 和 FFN）使用 rank 128 LoRA。两种方法的主干 LoRA 与补偿层分别使用 `2e-6` 和 `2e-5` 学习率，只比较路由差异。linear-only 对照冻结 Generator 主干，仅训练混合方法相同的 60 个补偿层张量。

三种方法统一使用 `0.85/0.95` CAG 预算；SLA+CAG 与 HSA+SLA+CAG 主方案统一使用 `lora_plus_linear` 参数化，HSA+CAG 维持 LoRA。SLA+CAG 尚无需要兼容的旧训练产物，正式训练只维护该统一契约。

## 2. 稀疏行为

LongLive2.0 最多保留 32 个 latent 帧的滚动 KV。每帧在 patch embedding 后对应 `22 x 40 = 880` 个 token；SP4 下 attention 侧每个 rank 看到完整序列和 `24 / 4 = 6` 个 head。

### 2.1 HSA+CAG

- HSA 只在 history 中选择 protected、near 4 帧和 dynamic 4 帧；current 8 帧始终进入候选 frame pool。
- current blocks 不再 dense append，而是和候选 history blocks 一起参加全局 CAG Top-K。
- CAG 根据 AR rollout 位置改变稀疏预算；默认 `dense_prefix_chunks=1`，仅首个没有 history 的 AR chunk 回退 dense。需要让前 N 个 chunk 保持 dense 时，可通过 YAML 或 `LONGLIVE_DENSE_PREFIX_CHUNKS` 覆盖；该计数按整段 rollout chunk 编号计算，不在 multi-shot 边界自动重置。history 不足时使用全部可用 history 后继续进入 block sparse stage。
- 训练和推理统一使用 128-token block，适配 MindIE-SD RainFusion 和 Ascend Triton 的共同执行契约。

### 2.2 SLA+CAG

- 不先裁剪 latent 帧，而是对当前滚动 KV 的 128-token blocks 全局打分并 Top-K。
- 强制保留 sink 与最近帧对应的 block。
- `dense_current_blocks=false`，表示 current blocks 参加全局 Top-K；`hard_keep_*_frames` 只表示 SLA block-stage safety anchor，且占用最终 CAG K。
- 线性注意力分支使用完整 KV 统计量，补偿稀疏 softmax 丢失的信息。

### 2.3 HSA+SLA+CAG

- HSA 调用与 HSA+CAG 相同的 history-only router：protected + near 4 + dynamic 4，再加入 current8；global sink / shot sink 只让前 2 帧进入候选池。
- SLA 只在候选帧的 blocks 内执行 Smooth-K Top-K，而不是将候选帧全部计算；current blocks 不保证全部选中。
- CAG 控制每个 AR chunk 的最终 block 预算，线性补偿仍使用完整 KV。
- SP4、32 秒尾部形状为 `Q=7040`、`KV=28160`，当前配置通常选择约 20 至 22 个/220 个 KV blocks，即约 90% 的有效稀疏率；最终值以运行日志为准。

路由索引本身不可微，但选中 block 的 Q/K/V 必须保持梯度。昇腾正式训练使用通过完整反向测试的 `ascend_triton`；CUDA 新增 `portable` 参考后端和显式 `cuda_flex` 后端，均需完成目标 GPU 准入。MindIE-SD RainFusion/BSA 是 NPU 推理前向算子。

## 3. 训练数据

训练输入是 UTF-8 文本，每个非空行是一条提示词。运行：

```bash
bash scripts/data/prepare_training_data.sh
```

也可指定本地源文件：

```bash
SOURCE_FILE=/mnt/share/r50063443/LongLive/data/vidprom_filtered_extended.txt \
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
DP = world_size / LONGLIVE_SP_SIZE
有效 batch = DP x batch_size x GRADIENT_ACCUMULATION_STEPS
```

当前 `batch_size=1`。模型有 24 个 attention head，每个 chunk 有 8 个 latent 帧，因此合法的 `LONGLIVE_SP_SIZE` 为 `1/2/4/8`。SLA+CAG 和 HSA+SLA+CAG 的 16 卡正式配置均使用 SP8 x DP2、梯度累积 4，有效 batch 为 8，并执行 200 次 optimizer update。

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

四个入口都委托给 `scripts/training/run_sparse_cag.sh`。`run_hsa_sla_cag.sh` 是 LoRA+linear 主方案，`run_hsa_sla_cag_linear_only.sh` 是 linear-only 对照。启动器将环境变量覆盖写入 `runs/training/<run-name>/config.resolved.yaml`，训练进程只读取 resolved 配置。

常用变量：

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES` | 当前节点可见 NPU | `0,...,15` |
| `NPROC_PER_NODE` | 每节点 worker 数 | 16 |
| `NNODES` / `NODE_RANK` | 节点数 / 当前节点编号 | 1 / 0 |
| `MASTER_ADDR` | 多节点 rendezvous 所在的 rank 0 地址；单节点忽略 | `127.0.0.1` |
| `LONGLIVE_SP_SIZE` | Ulysses SP 大小；SLA/混合方法入口默认 8；旧 `SP_SIZE` 仅作兼容别名 | 4 |
| `GRADIENT_ACCUMULATION_STEPS` | 梯度累积次数；SLA/混合方法入口默认 4 | 16 |
| `MAX_ITERS` | 训练结束目标 step；SLA/混合方法入口默认 200 | 2000 |
| `SAVE_INTERVAL` | checkpoint 间隔；SLA/混合方法入口默认 20 | 10 |
| `VIS_INTERVAL` | 训练内验证间隔，0 关闭 | 100 |
| `MAX_CHECKPOINTS` | 最多保留 checkpoint 数；SLA/混合方法入口默认 5 | 20 |
| `SPARSE_BACKEND` | 训练稀疏后端 | `ascend_triton` |
| `LONGLIVE_DENSE_PREFIX_CHUNKS` | 训练前缀 dense 的 AR chunk 数 | 1 |
| `GENERATOR_TRAIN_SCOPE` | Generator 范围；通常由具体入口设置 | 配置文件 |
| `GENERATOR_LR` / `LINEAR_LR` | 主干 LoRA / 原始补偿层学习率 | 配置文件 |
| `TRAIN_RUN_NAME` | 运行目录与自动恢复标识 | 时间戳名称 |
| `DISABLE_WANDB` | 1 表示禁用 W&B | 1 |

路径可通过 `LONGLIVE_ROOT`、`GENERATION_ENV`、`MODEL_ROOT`、`GENERATOR_CKPT`、`TRAIN_PROMPTS` 和 `CONFIG_PATH` 覆盖。

## 7. 训练命令

12 卡单步混合主方案烟测：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 LONGLIVE_SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 SAVE_INTERVAL=1 MAX_CHECKPOINTS=1 VIS_INTERVAL=0 \
TRAIN_RUN_NAME=hsa_sla_cag_12card_smoke \
bash scripts/training/run_hsa_sla_cag.sh
```

相同数据、稀疏率和并行布局的 linear-only 对照：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 LONGLIVE_SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 SAVE_INTERVAL=1 MAX_CHECKPOINTS=1 VIS_INTERVAL=0 \
TRAIN_RUN_NAME=hsa_sla_cag_linear_only_12card_smoke \
bash scripts/training/run_hsa_sla_cag_linear_only.sh
```

16 卡、200 step 混合正式训练：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NPROC_PER_NODE=16 LONGLIVE_SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
MAX_ITERS=200 SAVE_INTERVAL=20 MAX_CHECKPOINTS=5 VIS_INTERVAL=100 \
TRAIN_RUN_NAME=hsa_sla_cag_16card_200step \
bash scripts/training/run_hsa_sla_cag.sh
```

相同资源配置的 SLA+CAG 正式训练：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NPROC_PER_NODE=16 LONGLIVE_SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
MAX_ITERS=200 SAVE_INTERVAL=20 MAX_CHECKPOINTS=5 VIS_INTERVAL=100 \
TRAIN_RUN_NAME=sla_cag_16card_200step \
bash scripts/training/run_sla_cag.sh
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
LONGLIVE_SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=8 MAX_ITERS=200 \
TRAIN_RUN_NAME=hsa_sla_cag_24card_200step \
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

SLA+CAG 与混合训练每次 Generator 更新都会记录：

```text
linear_grad_tensors
linear_grad_with_gradient
linear_grad_nonzero
linear_grad_finite
linear_grad_l2_min
linear_grad_l2_max
```

前四项都应为 60，L2 范数应为有限非零值。缺失、全零或非有限梯度会在 optimizer step 前失败。HSA+CAG 则重点检查 Generator LoRA、Critic LoRA 的梯度范数和 loss 是否有限。

## 9. Checkpoint 与恢复

完整恢复包为 `checkpoints/step_*/train_state.pt`，包含 Generator 可训练参数、Critic LoRA、optimizer、scheduler、RNG 和数据游标等状态。使用相同 `TRAIN_RUN_NAME` 会自动扫描并恢复最大 step。

`MAX_ITERS` 是最终目标 step。例如已恢复到 100，设置 `MAX_ITERS=200` 表示继续到 200，而不是再训练 200 步。不同稀疏方法的 checkpoint 不能混用。

SLA+CAG 与混合主方案都会写出轻量 `generator_adapter.pt`，同时包含 Generator LoRA 与原始 `sla_linear`；linear-only 对照写出 `generator_linear.pt`。完整恢复始终使用 `train_state.pt`。不同方法或 scope 的参数集合和优化器状态不能交叉恢复。SLA+CAG 与混合方法正常结束时都会自动校验最终完整 checkpoint 和 sidecar；即使最终 step 不能整除 `SAVE_INTERVAL`，也会补存最终 checkpoint。

手工验证：

```bash
python scripts/checkpoints/validate_linear_checkpoint.py \
  runs/training/hsa_sla_cag_16card_200step/checkpoints/step_0000200/train_state.pt \
  --expected-step 200 --require-resume-state \
  --json-output runs/training/hsa_sla_cag_16card_200step/checkpoints/step_0000200/validation.json

python scripts/checkpoints/validate_linear_checkpoint.py \
  runs/training/hsa_sla_cag_16card_200step/checkpoints/step_0000200/generator_adapter.pt \
  --expected-step 200
```

## 10. 导出推理权重

SLA+CAG 使用 `generator_adapter.pt`：先把主干 LoRA 合并到训练时使用的 LongLive2.0 基础 Generator，再加载原始 `sla_linear`：

```bash
python scripts/checkpoints/merge_lora.py \
  --config_path configs/train/sla_cag.yaml \
  --generator_ckpt /mnt/share/r50063443/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --lora_ckpt runs/training/sla_cag_16card_200step/checkpoints/step_0000200/generator_adapter.pt \
  --output_path runs/merged/longlive2_sla_cag_200step.pt \
  --device npu:0
```

混合主方案使用同一脚本和 `generator_adapter.pt`，先合并主干 LoRA，再加载原始 `sla_linear`；linear-only 对照使用 `generator_linear.pt`。不要将训练增量合并到原始 Wan2.2 权重，必须使用训练时的 `longlive2_merged_generator.pt`。

主方案导出：

```bash
python scripts/checkpoints/merge_lora.py \
  --config_path configs/train/hsa_sla_cag.yaml \
  --generator_ckpt /mnt/share/r50063443/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --lora_ckpt runs/training/hsa_sla_cag_16card_200step/checkpoints/step_0000200/generator_adapter.pt \
  --output_path runs/merged/longlive2_hsa_sla_cag_lora_plus_linear_200step.pt \
  --device npu:0
```

linear-only 对照使用同一源配置，但必须显式覆盖训练范围：

```bash
python scripts/checkpoints/merge_lora.py \
  --config_path configs/train/hsa_sla_cag.yaml \
  --generator_train_scope linear_only \
  --generator_ckpt /mnt/share/r50063443/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --lora_ckpt runs/training/hsa_sla_cag_linear_only_12card_200step/checkpoints/step_0000200/generator_linear.pt \
  --output_path runs/merged/longlive2_hsa_sla_cag_linear_only_200step.pt \
  --device npu:0
```

导出后先运行同 checkpoint 的 dense 与对应 sparse DiT-only 对照，再运行 VBench。完整流程见[推理与评测指南](inference_and_evaluation.md)。

## 11. 单机 4 张 H100

CUDA 共用上述四个训练入口和三份 YAML，通过环境变量生成独立 resolved 配置，不复制模型或改变路由预算。安装见[环境安装与测试](setup_and_validation.md)；本轮 dry-run、四卡小模型、完整 step 与恢复命令见[当前测试清单](current_test.md)。

| 变量 | CUDA 默认值与作用 |
| --- | --- |
| `LLV2_DEVICE` | 显式设为 `cuda`；未设置时启动器仍选择 NPU。 |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3`；`NPROC_PER_NODE=4`，数量必须匹配。 |
| `LONGLIVE_SP_SIZE` | 4，形成 SP4×DP1；SP2×DP2 需显式设为 2。 |
| `GRADIENT_ACCUMULATION_STEPS` | 8；SP4×DP1、batch1 时有效 batch8。SP2×DP2 要保持 batch8 则设为 4。 |
| `SPARSE_BACKEND` | `portable`；完成算子准入后可显式设为 `cuda_flex`。 |
| `MODEL_LOAD_DTYPE` | `bfloat16`，减少底座加载内存；可设 `float32` 排查加载差异。 |
| `GENERATION_ENV` | 当前激活环境根目录；可显式覆盖。 |
| `DRY_RUN` | 1 时无需真实模型/提示词，保存 resolved 后退出；拒绝覆盖已有 run。 |

CUDA 与 NPU 保持相同 32 latent 帧、每 chunk8 帧、四步采样、128-token block、`0.85/0.95` CAG 与训练范围。Generator 和 Critic 实际 LoRA 各包含约 3.22 亿参数；`sla_linear` 共约 49.5 万参数。CUDA 的冻结底座可保存为 BF16，但可训练参数在恢复 checkpoint 前就提升为 FP32，避免小学习率更新和恢复值被 BF16 舍入。FSDP 将 FP32 adapter/补偿叶模块与 BF16 底座分别包装；此包装可能增加通信，必须实测。

只有 Generator 启用 Ulysses SP；Teacher/Critic 仍处理完整序列。四卡单节点 FSDP 对模型权重分片不等于 Teacher/Critic 激活也缩小四倍。`portable` 会重复 gather KV，不能承诺默认配置一定不 OOM。先以累积1完成完整 Generator+Critic 更新，再增加累积保持正式有效 batch；不要以缩短32帧窗口替代显存问题定位。

恢复使用同一训练方法、scope、加载 dtype 与并行配置，以 `MAX_ITERS` 指定最终目标 step。启动器在覆盖 resolved 前核对设备、加载精度、方法/后端、SP、训练范围、窗口、数据形状和基础 checkpoint 路径；不一致时保留原证据并拒绝恢复。单机已有目录但没有完整 `train_state.pt` 时也会拒绝重新启动。首次迁移优先使用基础合并 Generator 开始新 run；跨 NPU/CUDA 的 optimizer、RNG 和数据游标恢复不在当前验收承诺内。权重导出仍用现有 `scripts/checkpoints/merge_lora.py`，CUDA 设备为 `cuda:0`。
