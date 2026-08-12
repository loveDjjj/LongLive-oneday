# 当前测试清单

本文件只记录当前提交需要在昇腾服务器执行的临时测试。下一轮测试任务必须覆盖重写本文件；长期稳定命令应迁移到[环境安装与测试](setup_and_validation.md)。

## 测试目标

验证 Wan2.2 分片 checkpoint 的低内存加载修复，以及 HSA+SLA+CAG 主方案 `lora_plus_linear` 的 4 卡单步训练。此前算子前向、反向已经通过，本轮不重复运行算子测试。

故障根因是原生 Wan checkpoint 不包含新增的 `sla_linear`。Diffusers/Accelerate 在 meta device 上构造 5B 模型时，这些缺失参数无法被 checkpoint 物化，最终 `.to(npu)` 报 `Cannot copy out of meta tensor`。修复后只延迟创建 `sla_linear`，其余 5B 参数仍使用分片低内存加载。

## 1. 更新与主机回归

```bash
cd /mnt/share/r50063443/LongLive-oneday
git pull origin feat/unified-sparse-attention

conda activate /mnt/share/r50063443/conda_envs/longlive
PYTHONPATH=. pytest -q \
  tests/test_sparse_linear_materialization.py \
  tests/test_training_config_contract.py \
  tests/test_training_resume.py \
  tests/test_inference_checkpoint_contract.py
```

## 2. LoRA+linear 主方案单步训练

使用新 run 名保留此前失败目录，不允许覆盖 `hsa_sla_cag_lora_plus_linear_4card_smoke`。

```bash
source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 SP_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=1 SAVE_INTERVAL=1 MAX_CHECKPOINTS=1 VIS_INTERVAL=0 \
TRAIN_RUN_NAME=hsa_sla_cag_lora_plus_linear_loadfix_4card_smoke \
bash scripts/training/run_hsa_sla_cag.sh
```

加载阶段预期不再出现 `Cannot copy out of meta tensor`。训练完成后检查：

```bash
run_dir=runs/training/hsa_sla_cag_lora_plus_linear_loadfix_4card_smoke
step_dir="${run_dir}/checkpoints/step_0000001"

grep -E 'generator_train_scope|lr:|lr_linear' "${run_dir}/config.resolved.yaml"
grep -E 'linear_grad_(tensors|with_gradient|nonzero|finite)' \
  logs/training/hsa_sla_cag_lora_plus_linear_loadfix_4card_smoke/metrics.jsonl

python - "${step_dir}/generator_adapter.pt" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert state["generator_train_scope"] == "lora_plus_linear"
assert state["generator_lora"]
assert len(state["generator_linear"]) == 60
print("lora_plus_linear_checkpoint=passed")
PY
```

预期 resolved scope 为 `lora_plus_linear`，LoRA 学习率为 `2e-6`、linear 学习率为 `2e-5`，四个 linear 梯度计数均为 60，并输出 `lora_plus_linear_checkpoint=passed`。在服务器结果回填前，真实 5B 权重加载和单步训练仍视为待验证。
