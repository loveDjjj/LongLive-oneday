# SLA+CAG 训练指南

本文是当前唯一训练流程的权威说明，覆盖训练目标、数据集、并行布局、启动命令、终端输出、日志、checkpoint、恢复和 LoRA 合并。环境安装与 kernel 测试见 [环境安装与测试](getting_started.md)，质量和性能评测见 [推理与评测指南](inference_and_evaluation.md)。

## 1. 训练范围

当前训练是 LongLive2.0-5B 的提示词驱动稀疏 DMD 后训练，不读取真实视频，也不维护 AR、I2V、teacher forcing、NVFP4 或非 causal 分支：

1. SLA+CAG Generator 从噪声在线生成 32 个 latent 帧。
2. 32 帧按每块 8 帧分成 4 个因果时间块，使用 4 步采样。
3. Dense Real Teacher 和 Dense Fake Critic 对 Generator 中间状态计算 score。
4. Critic 每个 optimizer step 更新；Generator 默认每 5 步更新一次。
5. Generator 与 Fake Critic 只更新 LoRA，基础参数冻结；Real Teacher 不更新。

四个时间块属于同一个视频样本，和 DP 数量无关。Generator 使用稀疏自注意力，Teacher/Critic 保持稠密，避免监督目标同时引入稀疏偏差。

## 2. SLA+CAG 行为

- CAG 根据 rollout 位置调整历史 KV 稀疏率；没有历史 KV 的第一块保持稠密。
- SLA 用每个 Q/K block 的代表向量做全局 Top-K，覆盖历史和当前 KV。
- Sink 与最近帧对应的 blocks 强制保留；默认 `dense_current=false`，避免当前
  8 帧稠密造成约 25% 的理论稀疏率上限。
- 线性注意力分支补偿稀疏 softmax 丢失的信息，其输出投影从零初始化并参与 LoRA 后训练。
- 正式训练使用 `ascend_triton`，kernel 直接消费 block LUT 并支持前向和反向。
- 路由索引不可微，但被选中的 Q/K/V 保持梯度传播。

latent 空间网格是 `44 x 80`，patch embedding 后为 `22 x 40=880` token/帧。
Ulysses all-to-all 收集完整序列并切分 head，因此 attention 侧每个 rank 都看到
`8 x 880=7040` 个 query token，只是 head 数变为 `24 / SP_SIZE`。
`block_q` 和 `block_k` 必须整除 7040，正式配置使用 128。

## 3. 训练数据集

### 3.1 数据契约

训练只接受 UTF-8 文本：

```text
一条非空行 = 一个训练样本
```

空行会被忽略。`PromptDataset` 为每条提示词返回同一提示词组成的四块时间序列：

```python
{
    "idx": 17,
    "prompt": "A paper boat floats along a narrow stream.",
    "prompts": ["A paper boat floats along a narrow stream."] * 4,
}
```

这里的 4 是 `32 latent frames / 8 frames per block`，不是四条独立样本。

### 3.2 准备 VidProM 衍生提示词

默认脚本离线运行，优先复用已经通过数量与 SHA256 校验的文件：

```bash
bash scripts/data/prepare_training_data.sh
```

指定本地源文件：

```bash
SOURCE_FILE=/path/to/vidprom_filtered_extended.txt \
bash scripts/data/prepare_training_data.sh
```

只有明确允许联网时才启用下载：

```bash
ALLOW_DOWNLOAD=1 bash scripts/data/prepare_training_data.sh
```

输入目标是 248221 条非空提示词：

```text
SHA256 7896742f468bc8aef9e4547424d1ce0a951acdb2a82233790155401a99bf5aa5
```

准备脚本会按规范化文本去重，并排除与完整 VBench Standard/Augmented 重合的提示词。输出：

```text
data/train/vidprom_filtered_extended/
├── prompts_train.txt       # 248217 条
└── manifest.json           # 来源、hash、去重数和评测集排除数
```

输出 SHA256：

```text
c5ca345c5cb83db295dee0dda0f06530032e5ea2fe0e83c6fe686a4111b02623
```

VidProM 衍生数据按 CC BY-NC 4.0 的适用范围使用。训练启动器只校验文件存在和非空行数量；标准数据准备脚本额外校验固定 SHA256。

## 4. SP、DP、step 和样本数

```text
world_size = NNODES x NPROC_PER_NODE
DP = world_size / SP_SIZE
有效 batch = DP x batch_size x GRADIENT_ACCUMULATION_STEPS
```

当前 `batch_size` 固定为 1。模型有 24 个 attention head，每个时间块有 8 个 latent 帧，因此合法 `SP_SIZE` 为 `1/2/4/8`。

| 卡数 | 推荐布局 | 累积为 1 时每 step 的提示词数 |
| ---: | --- | ---: |
| 1 | SP1 x DP1 | 1 |
| 12 | SP4 x DP3 | 3 |
| 16 | SP4 x DP4 或 SP8 x DP2 | 4 或 2 |
| 24，双节点各 12 卡 | SP4 x DP6 | 6 |
| 32，双节点各 16 卡 | SP8 x DP4 或 SP4 x DP8 | 4 或 8 |

一个 `step` 是一次 optimizer 更新，不是一条样本或一个时间块。12 卡 SP4、DP3、累积 16 时每步抽样 48 条提示词；2000 步约 96000 次抽样，相当于 248217 条训练集的约 0.387 epoch。

增加 SP 主要改变单样本的分片和通信；增加 DP 主要提高总吞吐。跨节点 SP 会增加 HCCL 开销，推荐让每个 SP 组完整位于单节点。

## 5. 配置和启动变量

唯一源配置：

```text
configs/train/sla_cag.yaml
```

启动器把环境覆盖写到 `runs/training/<run-name>/config.resolved.yaml`，训练进程只读取 resolved 配置。

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES` | 当前节点可见 NPU | `0,...,15` |
| `NPROC_PER_NODE` | 每节点 worker 数 | 16 |
| `NNODES` / `NODE_RANK` | 节点数 / 当前节点编号 | 1 / 0 |
| `MASTER_ADDR` / `MASTER_PORT` | rendezvous 地址 | `127.0.0.1:29600` |
| `SP_SIZE` | Ulysses SP 大小 | 4 |
| `GRADIENT_ACCUMULATION_STEPS` | 梯度累积次数 | 16 |
| `MAX_ITERS` | 训练结束时的目标 step | 2000 |
| `SAVE_INTERVAL` | checkpoint 间隔 | 10 |
| `VIS_INTERVAL` | 训练内验证间隔，0 关闭 | 100 |
| `MAX_CHECKPOINTS` | 保留的 checkpoint 数 | 20 |
| `SLA_BACKEND` | 稀疏后端 | `ascend_triton` |
| `SLA_QUERY_BLOCK_BATCH` | portable Query block 合并数 | 1 |
| `SHARDING_STRATEGY` | 可选 FSDP 策略覆盖 | 空，使用 YAML |
| `TRAIN_RUN_NAME` | 运行目录与自动恢复标识 | 时间戳名称 |
| `DISABLE_WANDB` | 1 禁用 W&B | 1 |
| `LLV2_TRAIN_PROGRESS` | 1 显示 tqdm 进度 | 1 |

路径变量：`LONGLIVE_ROOT`、`GENERATION_ENV`、`MODEL_ROOT`、`GENERATOR_CKPT`、`TRAIN_PROMPTS`、`CONFIG_PATH`。启动器会在分配模型前校验 Python、torchrun、模型文件、Generator checkpoint 和提示词文件。

## 6. 常用训练命令

### 6.1 单卡功能检查

5B Generator、Teacher、Critic 可能超过单卡 HBM；该命令主要检查初始化与小环境：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 SP_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 SAVE_INTERVAL=1 MAX_CHECKPOINTS=1 \
VIS_INTERVAL=0 TRAIN_RUN_NAME=sla_cag_1card_smoke \
bash scripts/training/run_sla_cag.sh
```

### 6.2 12 卡单步烟测

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 SAVE_INTERVAL=1 MAX_CHECKPOINTS=1 \
VIS_INTERVAL=0 TRAIN_RUN_NAME=sla_cag_12card_smoke \
bash scripts/training/run_sla_cag.sh
```

### 6.3 12 卡正式训练

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=16 \
MAX_ITERS=2000 SAVE_INTERVAL=10 MAX_CHECKPOINTS=20 \
VIS_INTERVAL=100 TRAIN_RUN_NAME=sla_cag_12card_2k \
bash scripts/training/run_sla_cag.sh
```

有效 batch 是 48。累积 16 会执行 16 个 micro-batch，墙钟时间近似随累积增加；改变有效 batch 后应重新评估学习率和总样本数。

### 6.4 双节点 24 卡

两台机器使用相同代码、环境、共享路径、`TRAIN_RUN_NAME`、`MASTER_ADDR` 和 `MASTER_PORT`。

节点 0：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=12 \
MASTER_ADDR=10.0.0.10 MASTER_PORT=29600 \
SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=8 MAX_ITERS=2000 \
TRAIN_RUN_NAME=sla_cag_24card_2k \
bash scripts/training/run_sla_cag.sh
```

节点 1 运行同一命令，只将 `NODE_RANK=1`。该布局为 SP4 x DP6，有效 batch 为 48。

### 6.5 双节点 32 卡

每节点 16 卡可用 SP8，使 SP 组不跨节点：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NNODES=2 NODE_RANK=<0或1> NPROC_PER_NODE=16 \
MASTER_ADDR=10.0.0.10 MASTER_PORT=29600 \
SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=12 MAX_ITERS=2000 \
TRAIN_RUN_NAME=sla_cag_32card_2k \
bash scripts/training/run_sla_cag.sh
```

该布局为 SP8 x DP4，有效 batch 为 48。

## 7. 终端输出和指标

启动阶段应看到实际布局、路径和后端：

```text
[launch] time=... node=0/1 run=sla_cag_12card_2k
[run] Ascend Triton SLA backend is available
[run] node=0/1 devices=... world=12 SP=4 DP=3 effective_batch=48
[run] prompts=248217 model_root=...
[run] generator_ckpt=...
[run] save_interval=10 vis_interval=100 max_checkpoints=20
```

训练 tqdm 的 postfix 会依次显示 text encoding、rollout/loss、backward、optimizer。Generator 更新步输出：

```text
step 1, per iteration time ..., generator_loss ..., generator_grad_norm ...,
dmdtrain_gradient_norm ..., critic_loss ..., critic_grad_norm ...
```

其他步只输出 critic 字段。检查 loss/grad norm 是否有限、迭代时间是否突增，以及所有 rank 是否停在同一阶段。

`logs/training/<run-name>/metrics.jsonl` 每行是独立 JSON：

```json
{"critic_grad_norm": 0.0, "critic_loss": 0.0, "iteration_seconds": 0.0, "step": 1}
```

Generator 更新步额外包含 `generator_loss`、`generator_grad_norm`、`dmdtrain_gradient_norm`。第一步 `iteration_seconds` 固定为 0；不要把 JSONL 改成多行 JSON 数组。W&B 默认禁用，设置 `DISABLE_WANDB=0` 才会上报。

## 8. 保存数据和目录

```text
runs/training/<run-name>/
├── config.source.yaml
├── config.resolved.yaml
├── manifest.json
├── wandb/
└── checkpoints/
    └── step_0000100/
        └── train_state.pt

logs/training/<run-name>/
├── node_0.log
├── node_1.log                 # 多节点时存在
└── metrics.jsonl
```

`manifest.json` 记录 Git commit/dirty 状态、Python 和 torch/torch_npu/triton 版本、world/SP/DP、有效 batch、配置和数据 SHA256 以及关键环境变量。`node_<rank>.log` 追加保存 launcher 和本节点所有 worker 输出。

## 9. Checkpoint 格式和恢复

当前 `train_state.pt` 格式版本为 4。原生 HSA/SLA 的 Generator 使用
`generator_lora`；hybrid `linear_only` 使用 `generator_linear`：

```python
{
    "generator_lora": ...,
    "critic_lora": ...,
    "generator_optimizer": ...,
    "critic_optimizer": ...,
    "step": ...,
    "checkpoint_format_version": 4,
    "generator_train_scope": "lora",  # hybrid 为 linear_only
    "generator_trainable_parameters": ...,
    "sparse_method": "sla_cag",
    "world_size": ...,
    "sequence_parallel_size": ...,
    "data_parallel_size": ...,
    "batch_size": 1,
    "gradient_accumulation_steps": ...,
    "global_samples_consumed": ...,
    "rng_states": ...,
}
```

复用同一 `TRAIN_RUN_NAME` 会扫描 `checkpoints/step_*/train_state.pt` 并按最大 step
自动恢复。恢复状态必须显式包含 `sparse_method: sla_cag`；旧 HSA 或无方法标记的
checkpoint 会被拒绝，防止混用路由、投影和 optimizer 状态。稠密基础 Generator
不受此限制，加载时缺失的 `sla_linear` 参数会按零初始化补齐。

Hybrid 每次保存完整 `train_state.pt` 时，还会原子写入同目录下的小型
`generator_linear.pt`。前者包含 critic、optimizer、RNG 和数据游标，用于恢复训练；
后者只包含 30 层线性投影及必要元数据，用于快速验收和导出。训练后先执行：

```bash
python scripts/checkpoints/validate_linear_checkpoint.py \
  runs/training/hsa_sla_cag_12card_1k/checkpoints/step_0001000/train_state.pt \
  --expected-step 1000 --require-resume-state \
  --json-output runs/training/hsa_sla_cag_12card_1k/checkpoints/step_0001000/validation.json

python scripts/checkpoints/validate_linear_checkpoint.py \
  runs/training/hsa_sla_cag_12card_1k/checkpoints/step_0001000/generator_linear.pt \
  --expected-step 1000
```

默认检查 30 层 weight/bias 配对、形状、有限值和非零更新，并拒绝其他算法或
训练 scope。仅验证零初始化测试文件时才使用 `--allow-zero`。

`MAX_ITERS` 是最终目标 step。例如 checkpoint 已到 100，设置 `MAX_ITERS=2000` 会继续到 2000，不是额外训练 2000 步。改变 SP/DP/卡数后 optimizer 可重分片，但数据分配和 RNG 不保证逐位一致。

## 10. 合并 Generator LoRA

训练 checkpoint 不是完整模型。VBench/msprof 使用前，将 `generator_lora` 合并到训练时使用的同一个 LongLive 基础 Generator：

```bash
python scripts/checkpoints/merge_lora.py \
  --config_path configs/train/sla_cag.yaml \
  --generator_ckpt /mnt/share/weight/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --lora_ckpt runs/training/sla_cag_12card_2k/checkpoints/step_0002000/train_state.pt \
  --output_path runs/merged/longlive2_sla_cag_2k.pt \
  --device npu:0
```

脚本只合并 `generator_lora`；`critic_lora` 只用于训练恢复。输出是 `{"generator": state_dict, ...}` 的完整 BF16 checkpoint。不要把 LoRA 合并到原始 Wan 权重，必须使用训练配置中的 `longlive2_merged_generator.pt`。

## 11. 验收和故障检查

最低验收：

- Ascend Triton SLA 前向/反向 smoke test 通过。
- resolved 配置中的 SP、累积、路径和 `backend` 符合预期。
- loss 与 grad norm 无 NaN/Inf。
- checkpoint 能恢复 step、LoRA、optimizer 和数据游标。
- 使用同一合并权重完成 dense/SLA VBench 对照和 msprof 性能采集。

训练卡住时先检查所有节点日志中最早的错误。`ASCEND_LAUNCH_BLOCKING=1` 仅用于单步定位；长期开启会显著降低速度。若不同节点共享目录不可见，非 rank-0 节点会等待 `config.resolved.yaml.ready` 最多 600 秒后失败。
