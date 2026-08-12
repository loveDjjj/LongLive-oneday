# 当前测试清单

本文件只记录当前提交需要在昇腾服务器执行的临时测试。下一轮测试任务必须覆盖重写本文件；长期稳定命令应迁移到[环境安装与测试](setup_and_validation.md)。

## 测试目标

验证 HSA+SLA+CAG 的两种训练范围：

- 主方案 `lora_plus_linear`：Generator attention 主干 LoRA 与 30 层原始 `sla_linear` 联合训练；
- 对照方案 `linear_only`：Generator 主干冻结，只训练相同的 60 个 `sla_linear` 张量。

两次 smoke 必须使用不同 `TRAIN_RUN_NAME`，checkpoint 不可交叉恢复。

## 1. 更新与主机回归

```bash
cd /mnt/share/r50063443/LongLive-oneday
git pull origin feat/unified-sparse-attention

conda activate /mnt/share/r50063443/conda_envs/longlive
PYTHONPATH=. pytest -q \
  tests/test_training_config_contract.py \
  tests/test_training_resume.py \
  tests/test_inference_checkpoint_contract.py \
  tests/test_linear_checkpoint_validation.py \
  tests/test_shell_launchers.py
```

## 2. 训练算子完整反向

```bash
source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=4 \
python tests/npu/benchmark_sparse_attention.py \
  --method hsa_sla_cag --backend ascend_triton --device npu:0 \
  --latent-frames 32 --warmup 1 --iterations 1 \
  --check-training-backward
```

预期包含 `training_backward=passed`。

## 3. LoRA+linear 主方案单步训练

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 SAVE_INTERVAL=1 MAX_CHECKPOINTS=1 VIS_INTERVAL=0 \
TRAIN_RUN_NAME=hsa_sla_cag_lora_plus_linear_4card_smoke \
bash scripts/training/run_hsa_sla_cag.sh
```

检查：

```bash
run_dir=runs/training/hsa_sla_cag_lora_plus_linear_4card_smoke
step_dir="${run_dir}/checkpoints/step_0000001"

grep -E 'generator_train_scope|lr:|lr_linear' "${run_dir}/config.resolved.yaml"
grep -E 'linear_grad_(tensors|with_gradient|nonzero|finite)' \
  logs/training/hsa_sla_cag_lora_plus_linear_4card_smoke/metrics.jsonl

python - "${step_dir}/generator_adapter.pt" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert state["generator_train_scope"] == "lora_plus_linear"
assert state["generator_lora"]
assert len(state["generator_linear"]) == 60
print("lora_plus_linear_checkpoint=passed")
PY
```

预期 resolved scope 为 `lora_plus_linear`，学习率为 LoRA `2e-6`、linear `2e-5`，四个 linear 梯度计数均为 60，并输出 `lora_plus_linear_checkpoint=passed`。

## 4. linear-only 对照单步训练

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 SAVE_INTERVAL=1 MAX_CHECKPOINTS=1 VIS_INTERVAL=0 \
TRAIN_RUN_NAME=hsa_sla_cag_linear_only_4card_smoke \
bash scripts/training/run_hsa_sla_cag_linear_only.sh
```

检查：

```bash
run_dir=runs/training/hsa_sla_cag_linear_only_4card_smoke
step_dir="${run_dir}/checkpoints/step_0000001"

grep 'generator_train_scope' "${run_dir}/config.resolved.yaml"
python scripts/checkpoints/validate_linear_checkpoint.py \
  "${step_dir}/generator_linear.pt" --expected-step 1
```

预期 resolved scope 为 `linear_only`，并输出 `linear_checkpoint=passed`。主机测试不能替代这两次真实 NPU 训练，服务器结果回填前均视为待验证。
