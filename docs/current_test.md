# 当前测试清单

本轮只验证 PEFT 0.19.1 与旧版 Transformers 组合下的 LoRA checkpoint 加载修复，并从 SLA+CAG step 200 切换到双节点 32 卡继续训练。长期稳定命令见[环境安装与测试](setup_and_validation.md)。

## 1. 更新与主机回归测试

两个节点都执行：

```bash
cd /mnt/a800_share/r50063443/LongLive-oneday
git pull origin feat/unified-sparse-attention

conda activate /mnt/a800_share/r50063443/conda_envs/longlive
source /mnt/a800_share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh

python - <<'PY'
import peft
import transformers
print("peft", peft.__version__)
print("transformers", transformers.__version__)
PY

python -m pytest -q \
  tests/test_lora_utils.py \
  tests/test_training_resume.py \
  tests/test_inference_checkpoint_contract.py
```

预期测试全部通过。服务器可继续使用当前 `peft==0.19.1`，不要求临时升级 Transformers。

## 2. 续训前检查

共享目录必须保留完整 step 200 恢复包：

```bash
cd /mnt/a800_share/r50063443/LongLive-oneday

export TRAIN_RUN_NAME=sla_cag_16card_1000step
export STEP200="runs/training/${TRAIN_RUN_NAME}/checkpoints/step_0000200"

test -f "${STEP200}/train_state.pt"
test -f "${STEP200}/generator_adapter.pt"

python scripts/checkpoints/validate_linear_checkpoint.py \
  "${STEP200}/train_state.pt" \
  --expected-method sla_cag \
  --expected-step 200 \
  --require-resume-state
```

必须输出 `linear_checkpoint=passed`。正式续训前另行归档 step 200，避免其被 `MAX_CHECKPOINTS=5` 的滚动清理删除。

## 3. 单节点 16 卡立即重试

先用当前已占用的 16 卡确认真实 PEFT checkpoint 可以越过 LoRA 加载并进入 step 200。该命令会直接继续正式训练，不是只加载后退出：

```bash
cd /mnt/a800_share/r50063443/LongLive-oneday
conda activate /mnt/a800_share/r50063443/conda_envs/longlive
source /mnt/a800_share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NPROC_PER_NODE=16 SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
MAX_ITERS=1000 SAVE_INTERVAL=50 MAX_CHECKPOINTS=5 VIS_INTERVAL=100 \
TRAIN_RUN_NAME=sla_cag_16card_1000step \
bash scripts/training/run_sla_cag.sh
```

单节点布局保持原训练的 `SP8 × DP2 × accumulation4`，有效 batch 仍为 8，因此可以恢复逐 rank RNG，并保持最接近原 200-step 训练的连续性。

## 4. 双节点 32 卡续训

两台机器必须访问同一个仓库和 `runs/training/sla_cag_16card_1000step/`。将下面的 `<rank0-ip>` 替换为节点 0 在双节点训练网络中的实际 IP；不要设置 `MASTER_PORT`。

节点 0 先启动：

```bash
cd /mnt/a800_share/r50063443/LongLive-oneday
conda activate /mnt/a800_share/r50063443/conda_envs/longlive
source /mnt/a800_share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=16 \
MASTER_ADDR=<rank0-ip> \
SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=2 \
MAX_ITERS=1000 SAVE_INTERVAL=20 MAX_CHECKPOINTS=5 VIS_INTERVAL=100 \
TRAIN_RUN_NAME=sla_cag_16card_1000step \
bash scripts/training/run_sla_cag.sh
```

节点 1 再启动：

```bash
cd /mnt/a800_share/r50063443/LongLive-oneday
conda activate /mnt/a800_share/r50063443/conda_envs/longlive
source /mnt/a800_share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
NNODES=2 NODE_RANK=1 NPROC_PER_NODE=16 \
MASTER_ADDR=<rank0-ip> \
SP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=2 \
MAX_ITERS=1000 SAVE_INTERVAL=20 MAX_CHECKPOINTS=5 VIS_INTERVAL=100 \
TRAIN_RUN_NAME=sla_cag_16card_1000step \
bash scripts/training/run_sla_cag.sh
```

## 5. 必须确认的恢复日志

修复后的启动日志必须依次包含：

```text
Auto resume: Found LoRA checkpoint at .../step_0000200/train_state.pt
Loading LoRA generator weights: 600 keys in checkpoint
Resuming LoRA training from step 200
Restored generator and critic AdamW state from LoRA checkpoint
Warning: training layout changed from world=16, SP=8, DP=2, batch=1, accumulation=4 to world=32, SP=8, DP=4, batch=1, accumulation=2
```

最后一条布局变更警告只在双节点 32 卡续训时出现。world size 改变后，原 checkpoint 的 16 份逐 rank RNG 无法映射到 32 个 rank，因此还会出现 RNG fallback 警告。这意味着双节点续训不保证 bitwise 一致，但有效 batch 仍保持为 `DP4 × accumulation2 = 8`。

确认训练进入 `step 200: accumulation 1/2`，并至少完成一次 optimizer update。若仍失败，保留两个节点的：

```text
logs/training/sla_cag_16card_1000step/node_0.log
logs/training/sla_cag_16card_1000step/node_1.log
runs/training/sla_cag_16card_1000step/config.resolved.yaml
runs/training/sla_cag_16card_1000step/manifest.json
```
