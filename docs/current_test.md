# 当前测试清单

本文件只记录当前一轮需要在昇腾服务器执行的临时命令。下一轮测试任务必须覆盖重写本文件；长期稳定命令应迁移到[环境安装与测试](setup_and_validation.md)。

## 测试目标

验证 SLA+CAG 在 16 卡 `SP8 × DP2` 下的 LoRA 训练线路。正式参数与 HSA+SLA+CAG 对齐：梯度累积 4、1000 次 optimizer update、每 20 step 保存、最多保留 5 个 checkpoint、每 100 step 保存一次验证 latent。

SLA+CAG 的 Generator 和 Fake Critic 均训练 rank 128 LoRA；Generator 每步更新。它不使用混合方案的 `lora_plus_linear` scope，也不生成 `generator_adapter.pt`，完整训练恢复使用 `train_state.pt`。

## 1. 16 卡单步准入

先使用独立 run 验证 SP8 下模型加载、SLA 反向、两套 LoRA optimizer 和 checkpoint 保存：

```bash
cd /mnt/share/r50063443/LongLive-oneday
git pull origin feat/unified-sparse-attention

source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
conda activate /mnt/share/r50063443/conda_envs/longlive

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NPROC_PER_NODE=16 SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 SAVE_INTERVAL=1 MAX_CHECKPOINTS=1 VIS_INTERVAL=0 \
TRAIN_RUN_NAME=sla_cag_lora_16card_sp8_smoke \
bash scripts/training/run_sla_cag.sh
```

预期启动摘要包含 `world=16 SP=8 DP=2 effective_batch=2`，训练完成后存在：

```text
runs/training/sla_cag_lora_16card_sp8_smoke/checkpoints/step_0000001/train_state.pt
```

日志中的 `generator_loss`、`generator_grad_norm`、`critic_loss` 和 `critic_grad_norm` 必须有限。该真实 NPU 准入在结果回填前视为待验证。

## 2. 16 卡 1000 步正式训练

单步准入通过后启动正式 run：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NPROC_PER_NODE=16 SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
MAX_ITERS=1000 SAVE_INTERVAL=20 MAX_CHECKPOINTS=5 VIS_INTERVAL=100 \
TRAIN_RUN_NAME=sla_cag_lora_16card_1k \
bash scripts/training/run_sla_cag.sh
```

运行语义：

- 有效 batch 为 `DP2 × batch1 × accumulation4 = 8`；
- 每 4 个 Generator backward 和 4 个 Critic backward 执行一次 optimizer update；
- 全程 Generator 与 Critic 各执行 4000 次 backward、1000 次 optimizer update；
- checkpoint 写在 step 20、40、...、1000，最多保留最近 5 个；
- step 100、200、...、1000 保存训练内验证 latent，不执行 VAE 或 VBench。

启动后检查：

```bash
run_dir=runs/training/sla_cag_lora_16card_1k
grep -E 'sequence_parallel_size|gradient_accumulation_steps|max_iters|log_iters|max_checkpoints|dfake_gen_update_ratio|generator_train_scope' \
  "${run_dir}/config.resolved.yaml"
tail -f logs/training/sla_cag_lora_16card_1k/node_0.log
```

预期 resolved 配置为 `SP8`、累积 4、1000 step、保存间隔 20、保留 5 个、更新比例 1，且 `generator_train_scope: lora`。
