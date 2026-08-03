# 昇腾 NPU 上的 LongLive2 HSA+CAG

本分支新增了与 Light Forcing 类似的稀疏后训练和推理路径，同时保留 LongLive 现有的因果 KV Cache、Ulysses SP、检查点格式和异步 VAE 实现。

## 已实现功能

- CAG Chunk 稀疏率：第一个 Chunk 保持 Dense，后续 Chunk 的历史 KV 稀疏率按 `base - beta / sqrt(frames)` 增长，并满足配置的平均历史稀疏率目标。
- HSA 帧级路由：保留 Sink 帧、最近帧，以及与当前 Query 最相关的中间历史帧。
- HSA Block 级路由：在选中的历史帧内继续筛选 Token Block，当前 Chunk 始终保持完整可见。
- 昇腾执行后端：先 Gather 选中的 K/V，再调用 PyTorch SDPA；支持 Query Block 批处理。路由选择本身不可微，但梯度可以通过被选中的 Q/K/V 传播。
- 训练：仅使用 Prompt 的 DMD 后训练，包含稀疏 Generator、Dense Real Teacher 和 Fake Critic。
- 推理：普通推理和 Ulysses SP 的 Cached Attention 路径均支持 HSA+CAG。

当前 NPU 后端优先保证正确性和可训练性，并不是 Light Forcing 中融合后的 Triton/FA4 Kernel。它会真实减少参与注意力计算的历史 K/V Token，但最终加速效果仍需通过 msprof 验证，不能仅根据理论稀疏率判断。

本实现相对原版 Light Forcing 做了以下适配：

- 原论文使用因果 Wan2.1 Student，并使用非因果 Wan2.1-14B 提供 Score；本项目使用全因果 Wan2.2-TI2V-5B Teacher/Critic，以兼容现有 LongLive2 检查点和训练代码。
- 保留项目原生的 8 帧 Latent Chunk 和 `44x80` Token Grid。
- 默认平均历史稀疏率和末期基础稀疏率分别设为 `0.85`、`0.95`，比论文中的 `0.88`、`0.98` 更保守，便于先验证 NPU 稳定性和质量。

## 训练数据

训练只需要 Prompt，不需要读取真实视频或预先保存的 VAE Latent。Student 从纯噪声开始在线生成中间状态，DMD Loss 比较 Real Teacher 与 Fake Critic 在这些状态上的 Score。

准备数据：

```bash
bash scripts/prepare_hsa_training_data.sh
```

脚本默认离线运行。如果已存在并通过 SHA256 和数量校验的 `prompts_train.txt`，会直接复用且不会发起任何网络请求；否则会查找本地 `source_prompts.txt` 或 `vidprom_filtered_extended.txt`，完成去重并移除与 VBench Prompt 规范化后完全相同的样本。也可以通过 `SOURCE_FILE=/path/to/vidprom_filtered_extended.txt` 指定本地文件。只有显式设置 `ALLOW_DOWNLOAD=1` 时才会访问 Hugging Face。预期得到 `248217` 条训练 Prompt。数据来源和过滤规则见 `data/train/README.md`。

## 启动训练

先运行 10 Iteration 的 Smoke Test。这里的 12 个进程是 FSDP Worker，不使用推理阶段的 SP/DP 布局：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 \
GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=10 \
TRAIN_RUN_NAME=hsa_cag_smoke \
bash scripts/run_npu_hsa_cag_training.sh
```

Smoke Test 通过后启动正式训练：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 \
GRADIENT_ACCUMULATION_STEPS=6 \
MAX_ITERS=2000 \
TRAIN_RUN_NAME=hsa_cag_2k \
bash scripts/run_npu_hsa_cag_training.sh
```

复用同一个 `TRAIN_RUN_NAME` 会从该目录中最新的检查点继续训练。新版 LoRA 检查点会完整保存 Generator/Critic LoRA、两套 AdamW Optimizer State、训练 Step、全局样本游标和各 Rank 的随机数状态。相同卡数、单卡 Batch Size 和梯度累积配置下可以无损恢复；旧版仅包含 LoRA 权重和 Step 的检查点仍可加载，但会输出 AdamW State 缺失警告。

改变卡数后，FSDP 会将完整 Optimizer State 重新分片，因此可以继续训练；但样本到 Rank 的分配和各 Rank 随机数流会改变，不能保证与原布局逐样本、逐位一致。启动脚本还会记录配置文件、解析后的配置、完整日志和实际使用端口。有效全局 Batch Size 为：

```text
NPROC_PER_NODE * batch_size * GRADIENT_ACCUMULATION_STEPS
```

## 合并与评测

训练检查点同时包含 Generator LoRA 和 Critic LoRA。评测前只需将 Generator LoRA 合并到 LongLive2 Generator：

```bash
/mnt/share/r50063443/conda_envs/longlive/bin/python \
  scripts/merge_lora_generator.py \
  --config_path configs/train_dmd_hsa_cag_npu_bf16.yaml \
  --generator_ckpt /mnt/share/weight/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --lora_ckpt logs/training/hsa_cag_2k/checkpoint_model_002000/model.pt \
  --output_path checkpoints/longlive2_hsa_cag_2k_merged.pt \
  --device npu:0
```

使用同一个合并后的检查点分别运行 Dense 对照组和 Sparse 实验组：

```bash
LONGLIVE_GENERATOR_CKPT=checkpoints/longlive2_hsa_cag_2k_merged.pt \
RUN_ID=hsa_cag_dense_control \
bash scripts/run_vbench.sh longlive2_standard_20pct

LONGLIVE_GENERATOR_CKPT=checkpoints/longlive2_hsa_cag_2k_merged.pt \
LONGLIVE_SPARSE_METHOD=hsa_cag \
RUN_ID=hsa_cag_sparse \
bash scripts/run_vbench.sh longlive2_standard_20pct
```

使用相同稀疏设置进行 32 秒 msprof 测试：

```bash
LONGLIVE_GENERATOR_CKPT=checkpoints/longlive2_hsa_cag_2k_merged.pt \
LONGLIVE_SPARSE_METHOD=hsa_cag \
bash scripts/run_msprof.sh 32s
```

重点比较以下指标：

- VBench Total、Quality 和 Semantic 分数。
- 无 Profiling 时的生成延迟与 FPS。
- 峰值 HBM 占用。
- FlashAttention/SDPA 算子耗时。
- HCCL 通信耗时。

建议的第一阶段验收标准是：VBench Total 下降小于 `0.5`，同时在多次无 Profiling 测试中获得可复现的端到端加速。

## 参数调优

主要参数位于配置文件的 `sparse_config` 和 `sparsity.options`：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `sparsity` | 后续 Chunk 的平均历史 KV 稀疏率目标 | `0.85` |
| `sparsity_base` | CAG 后期的基础稀疏率 | `0.95` |
| `block_q`、`block_k` | 路由和注意力使用的 Token Block 大小 | `40` |
| `keep_frames` | 进入第二阶段路由的历史帧数量 | `6` |
| `keep_sink` | 始终可被选中的最早历史帧数量 | `1` |
| `keep_near` | 始终可被选中的最近历史帧数量 | `2` |
| `dense_current` | 当前 Chunk 是否保持 Dense | `true` |
| `query_block_batch` | 单次 SDPA 合并处理的 Query Block 数量 | `2` |

在 `44x80` Grid 下，每个 Latent Frame 包含 `880` 个 Token，因此 Block 大小必须能整除 `880`。增大 `query_block_batch` 可以减少 Kernel Launch 次数，但会增加临时 Gather Tensor 的显存占用。建议在昇腾上从 `2` 开始，并在修改稀疏率之前通过 msprof 对比 `1/2/4`。

## 范围与限制

- HSA+CAG 支持零训练直接推理。DMD 后训练用于补偿稀疏注意力带来的分布偏移，是推荐步骤，但不是启用稀疏推理的必要条件。
- 当前 Chunk 保持 Dense，因此端到端 FLOPs 存在下限；稀疏主要作用于历史 KV。
- 当前使用 Mean-pooled Q/K 和 Top-K 完成路由，路由索引不可微。
- 训练阶段使用 FSDP 数据并行，不使用 Ulysses Sequence Parallel；推理阶段支持 Ulysses HSA。
- 如果 msprof 显示 Gather 和 SDPA 成为主要瓶颈，下一步应实现昇腾自定义融合 Block-Sparse 算子。
