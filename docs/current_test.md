# 当前测试清单

本文件只记录当前一轮需要在昇腾服务器执行的临时命令。下一轮测试任务必须覆盖重写本文件；长期稳定命令应迁移到[环境安装与测试](setup_and_validation.md)。

## 测试目标

1. 在 16 卡 `SP8 × DP2` 下验证 SLA+CAG 已统一为“主干 LoRA + 原始 `sla_linear`”，并以 `VIS_INTERVAL=1` 确认 step 1 验证后能继续完成 step 2。
2. 正式运行 `HSA+SLA+CAG` 与 `SLA+CAG` 各 200 次 optimizer update。

两种正式训练均保持梯度累积 4、有效 batch 8、每 20 step 保存、最多保留 5 个 checkpoint，并在 step 100 和 200 保存验证 latent。

## 1. VIS_INTERVAL=1 两步准入

先拉取最新代码并进入环境：

```bash
cd /mnt/share/r50063443/LongLive-oneday
git pull origin feat/unified-sparse-attention

source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
conda activate /mnt/share/r50063443/conda_envs/longlive
```

使用 SLA+CAG 运行 2 step。累积保持 4，以覆盖正式训练相同的 LoRA、原始补偿层、FSDP backward 和 optimizer 路径：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NPROC_PER_NODE=16 SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
MAX_ITERS=2 SAVE_INTERVAL=1 MAX_CHECKPOINTS=2 VIS_INTERVAL=1 \
TRAIN_RUN_NAME=sla_cag_lora_plus_linear_vis1_16card_smoke \
bash scripts/training/run_sla_cag.sh
```

训练结束后执行：

```bash
run_dir=runs/training/sla_cag_lora_plus_linear_vis1_16card_smoke
log_file=logs/training/sla_cag_lora_plus_linear_vis1_16card_smoke/metrics.jsonl

grep -E 'sequence_parallel_size|gradient_accumulation_steps|max_iters|log_iters|max_checkpoints|generator_train_scope|lr_linear' \
  "${run_dir}/config.resolved.yaml"
test -f "${run_dir}/checkpoints/step_0000002/train_state.pt"
test -f "${run_dir}/checkpoints/step_0000002/generator_adapter.pt"
test "$(find "${run_dir}/vis/step_0000001" -name 'latents_*.pt' | wc -l)" -eq 16
test "$(find "${run_dir}/vis/step_0000002" -name 'latents_*.pt' | wc -l)" -eq 16
test "$(wc -l < "${log_file}")" -eq 2
tail -n 2 "${log_file}"
grep -E 'linear_checkpoint=passed' "logs/training/sla_cag_lora_plus_linear_vis1_16card_smoke/node_0.log"
```

预期 resolved 配置包含 `SP8`、累积 4、`max_iters: 2`、`generator_train_scope: lora_plus_linear`、`lr_linear: 2e-5`。每条 metrics 的 `linear_grad_tensors/with_gradient/nonzero/finite` 均应为 60；日志应出现 adapter sidecar 的两次 `linear_checkpoint=passed`。两个验证目录应各有 16 个 latent，metrics 必须同时出现 step 1 和 step 2，loss、梯度范数均有限。这同时证明统一训练契约和 `VIS_INTERVAL` 恢复路径有效。该真实 NPU 准入在结果回填前视为待验证。

## 2. HSA+SLA+CAG 正式 200 step

准入通过后运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NPROC_PER_NODE=16 SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
MAX_ITERS=200 SAVE_INTERVAL=20 MAX_CHECKPOINTS=5 VIS_INTERVAL=100 \
TRAIN_RUN_NAME=hsa_sla_cag_16card_200step \
bash scripts/training/run_hsa_sla_cag.sh
```

预期有效 batch 为 `DP2 × batch1 × accumulation4 = 8`；Generator 和 Critic 各执行 800 次 backward、200 次 optimizer update。checkpoint 写在 step 20、40、...、200，最终只保留 step 120 至 200 的 5 份；验证 latent 写在 step 100 和 200。

## 3. SLA+CAG 正式 200 step

混合算法和 SLA 使用不同 run 名，可以串行运行；有独立的另一组 16 卡时也可并行运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NPROC_PER_NODE=16 SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
MAX_ITERS=200 SAVE_INTERVAL=20 MAX_CHECKPOINTS=5 VIS_INTERVAL=100 \
TRAIN_RUN_NAME=sla_cag_16card_200step \
bash scripts/training/run_sla_cag.sh
```

SLA 的并行布局、有效 batch、参数化、保存和验证节奏与混合算法相同。预期 `generator_train_scope: lora_plus_linear`，完整恢复文件为 `checkpoints/step_0000200/train_state.pt`，推理增量为同目录的 `generator_adapter.pt`。
