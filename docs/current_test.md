# 当前测试清单

本文件只记录当前一轮需要在昇腾服务器执行的临时命令。下一轮测试任务必须覆盖重写本文件；长期稳定命令应迁移到[环境安装与测试](setup_and_validation.md)。

## 已通过准入

提交 `ddc7298` 已完成 HSA+SLA+CAG `lora_plus_linear` 的 4 卡单步训练：

- Wan2.2 分片 checkpoint 成功加载，不再残留 meta tensor；
- Generator LoRA、Fake Critic LoRA 和 30 层 `sla_linear` 完成反向与 optimizer update；
- 60/60 个 `sla_linear` 张量均有非零、有限梯度；
- `train_state.pt` 和 `generator_adapter.pt` 均保存并通过校验。

## 16 卡 1000 步正式训练

下面使用单节点 `SP8 × DP2`、每卡 micro-batch 1、梯度累积 4，有效 batch 为 `2 × 1 × 4 = 8`。`MAX_ITERS=1000` 表示 1000 次 optimizer update；每次 update 累积 4 个 micro-batch。Generator 与 Critic 的更新比例为 1:1，因此全程各执行 4000 次 backward。

```bash
cd /mnt/share/r50063443/LongLive-oneday
git pull origin feat/unified-sparse-attention

source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
conda activate /mnt/share/r50063443/conda_envs/longlive

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NPROC_PER_NODE=16 SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
MAX_ITERS=1000 SAVE_INTERVAL=20 MAX_CHECKPOINTS=5 VIS_INTERVAL=100 \
TRAIN_RUN_NAME=hsa_sla_cag_lora_plus_linear_16card_1k \
bash scripts/training/run_hsa_sla_cag.sh
```

运行语义：

- 每 4 个 Generator backward 和 4 个 Critic backward 执行一次对应 optimizer update；
- 每 20 次 optimizer update 保存一次 checkpoint，即 step 20、40、...、1000；
- 最多保留最近 5 个 checkpoint；同一 `TRAIN_RUN_NAME` 重启会从最新完整 checkpoint 恢复；
- 每 100 step 执行一次训练内 latent 验证；如只关注训练吞吐，可设置 `VIS_INTERVAL=0`。

启动后检查 resolved 配置：

```bash
run_dir=runs/training/hsa_sla_cag_lora_plus_linear_16card_1k
grep -E 'sequence_parallel_size|gradient_accumulation_steps|max_iters|log_iters|generator_train_scope|lr:|lr_linear' \
  "${run_dir}/config.resolved.yaml"
tail -f logs/training/hsa_sla_cag_lora_plus_linear_16card_1k/node_0.log
```

预期启动摘要包含 `world=16 SP=8 DP=2 effective_batch=8`。首个 step 后，`metrics.jsonl` 中四个 `linear_grad_*` 计数应继续保持 60。
