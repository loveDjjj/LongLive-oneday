# 当前测试清单

本文件只记录当前一轮需要在昇腾服务器执行的临时命令。下一轮测试任务必须覆盖重写本文件；长期稳定命令应迁移到[环境安装与测试](setup_and_validation.md)。

## 测试目标

完成 200 step SLA+CAG `lora_plus_linear` checkpoint 的验收、合并和 VBench Standard 5% 评测，并与已有基础权重及 200-step HSA+SLA+CAG 结果组成训练方法与推理路由消融。

已有结果无需重跑：

```text
基础权重：dense / hsa_cag / sla_cag / hsa_sla_cag
Hybrid-trained：dense / hsa_sla_cag
```

本轮必须新增 4 组：

```text
SLA-trained + dense
SLA-trained + sla_cag
SLA-trained + hsa_sla_cag
Hybrid-trained + sla_cag
```

新增结果与已有结果合并后，形成两种 200-step 训练权重 × `dense/sla_cag/hsa_sla_cag` 的完整 `2×3` 消融。所有组固定使用 `longlive2_standard_5pct`：43 条提示词、5 个 seed、每组 215 个视频、125 pixel 帧、24 FPS、4 步采样、`SP2 × DP8`。

## 1. 环境与公共路径

```bash
cd /mnt/a800_share/r50063443/LongLive-oneday
git pull origin feat/unified-sparse-attention
git rev-parse --short HEAD

conda activate /mnt/a800_share/r50063443/conda_envs/longlive
source /mnt/a800_share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export GENERATION_ENV=/mnt/a800_share/r50063443/conda_envs/longlive
export AISBENCH_ENV=/mnt/a800_share/r50063443/conda_envs/aisbench_npu
export VBENCH_CACHE_DIR=/mnt/a800_share/r50063443/vbench_models
export CANN_ENV_SCRIPT=/mnt/a800_share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
export LONGLIVE_MODEL_ROOT=/mnt/a800_share/r50063443/Wan2.2-TI2V-5B
export BASE_CKPT=/mnt/a800_share/r50063443/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt

export SLA_STEP_DIR=runs/training/sla_cag_16card_200step/checkpoints/step_0000200
export SLA_TRAINED_CKPT=runs/merged/longlive2_sla_cag_200step.pt
export HYBRID_TRAINED_CKPT=runs/merged/longlive2_hsa_sla_cag_200step.pt

export TEST_ID="sla-200step-vbench-$(date +%Y%m%d-%H%M%S)"
mkdir -p "logs/tests/${TEST_ID}" runs/merged

test -f "${BASE_CKPT}"
test -f "${HYBRID_TRAINED_CKPT}"
```

如果实际训练 run 名不同，只覆盖 `SLA_STEP_DIR`，不要移动或重命名原始训练目录。

## 2. 验收 SLA+CAG 训练 checkpoint

先确认完整恢复文件、推理 sidecar 和训练结束时的自动校验结果存在：

```bash
test -f "${SLA_STEP_DIR}/train_state.pt"
test -f "${SLA_STEP_DIR}/generator_adapter.pt"
test -f "${SLA_STEP_DIR}/validation.json"
cat "${SLA_STEP_DIR}/validation.json"

python scripts/checkpoints/validate_linear_checkpoint.py \
  "${SLA_STEP_DIR}/train_state.pt" \
  --expected-method sla_cag \
  --expected-step 200 \
  --require-resume-state \
  --json-output "logs/tests/${TEST_ID}/sla-train-state-validation.json" \
  | tee "logs/tests/${TEST_ID}/sla-train-state-validation.txt"

python scripts/checkpoints/validate_linear_checkpoint.py \
  "${SLA_STEP_DIR}/generator_adapter.pt" \
  --expected-method sla_cag \
  --expected-step 200 \
  --json-output "logs/tests/${TEST_ID}/sla-adapter-validation.json" \
  | tee "logs/tests/${TEST_ID}/sla-adapter-validation.txt"
```

两条命令都必须输出 `linear_checkpoint=passed`。预期 `generator_train_scope=lora_plus_linear`、30 层、60 个有限且非零的原始 `sla_linear` 张量；`train_state.pt` 还必须通过 optimizer、RNG、数据游标和并行布局恢复状态校验。

## 3. 合并 SLA+CAG 推理权重

必须以 LongLive2.0 合并 Generator 为基础权重，将 SLA 训练得到的 Generator LoRA 合并，并加载同一 sidecar 中的原始 `sla_linear`：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python scripts/checkpoints/merge_lora.py \
  --config_path configs/train/sla_cag.yaml \
  --generator_ckpt "${BASE_CKPT}" \
  --lora_ckpt "${SLA_STEP_DIR}/generator_adapter.pt" \
  --output_path "${SLA_TRAINED_CKPT}" \
  --device npu:0 \
  | tee "logs/tests/${TEST_ID}/merge-sla-200step.txt"

test -s "${SLA_TRAINED_CKPT}"
sha256sum "${BASE_CKPT}" "${SLA_TRAINED_CKPT}" \
  | tee "logs/tests/${TEST_ID}/merged-checkpoint-sha256.txt"
```

合并日志必须包含：

```text
Loading LoRA checkpoint
Merging LoRA
Loading raw SLA linear checkpoint
Saved merged generator
```

使用内存映射检查合并包元数据和 30 层补偿参数，避免把整个 5B checkpoint 复制进主存：

```bash
python - "${SLA_TRAINED_CKPT}" <<'PY' | tee "logs/tests/${TEST_ID}/merged-sla-checkpoint-validation.txt"
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
assert checkpoint["checkpoint_format"] == "longlive_generator_merged_lora"
assert checkpoint["sparse_method"] == "sla_cag"
assert checkpoint["generator_train_scope"] == "lora_plus_linear"
assert checkpoint["merged_lora"] is True
state = checkpoint["generator"]
linear = {
    name: tensor
    for name, tensor in state.items()
    if ".sla_linear." in name
}
assert len(linear) == 60, len(linear)
assert all(torch.isfinite(tensor).all().item() for tensor in linear.values())
assert sum(torch.count_nonzero(tensor).item() for tensor in linear.values()) > 0
print(
    "merged_sla_checkpoint=passed",
    f"tensors={len(linear)}",
    f"source={checkpoint['source_lora_ckpt']}",
)
PY
```

不要将 `train_state.pt` 或 `generator_adapter.pt` 直接传给推理入口。

## 4. SLA-trained 三路由矩阵

同一份 SLA-trained checkpoint 分别运行 dense、SLA 和 Hybrid。该矩阵回答主干 LoRA 漂移、目标 SLA 路由的残余损失，以及 SLA 权重向 Hybrid 路由的迁移能力：

```bash
METHODS=dense,sla_cag,hsa_sla_cag \
VBENCH_PRESETS=longlive2_standard_5pct \
SUITE_ID=longlive2-sla-200step-crossroute-5pct \
bash scripts/evaluation/run_vbench_matrix.sh "${SLA_TRAINED_CKPT}"
```

预期产生：

```text
runs/vbench/longlive2-sla-200step-crossroute-5pct-longlive2_standard_5pct-dense/
runs/vbench/longlive2-sla-200step-crossroute-5pct-longlive2_standard_5pct-sla_cag/
runs/vbench/longlive2-sla-200step-crossroute-5pct-longlive2_standard_5pct-hsa_sla_cag/
runs/suites/longlive2-sla-200step-crossroute-5pct/vbench/
```

脚本支持断点续跑；保持同一个 `SUITE_ID` 重跑时，完整 case 会跳过，不完整 seed 会继续。

## 5. Hybrid-trained 补测 SLA 路由

已有 Hybrid-trained + dense/Hybrid，无需重复生成。只补测相同 Hybrid-trained checkpoint 下的 SLA 路由：

```bash
LONGLIVE_GENERATOR_CKPT="${HYBRID_TRAINED_CKPT}" \
LONGLIVE_SPARSE_METHOD=sla_cag \
RUN_ID=longlive2-hybrid-200step-sla_cag-5pct \
bash scripts/evaluation/run_vbench.sh longlive2_standard_5pct
```

该组与已有以下两组共同构成 Hybrid-trained 的三路由对照：

```text
longlive2-hybrid-200step-dense-5pct
longlive2-hybrid-200step-sla_cag-5pct
longlive2-hybrid-200step-hsa_sla_cag-5pct
```

## 6. 结果完整性检查

检查两种训练权重的 6 组结果：

```bash
for run_id in \
  longlive2-sla-200step-crossroute-5pct-longlive2_standard_5pct-dense \
  longlive2-sla-200step-crossroute-5pct-longlive2_standard_5pct-sla_cag \
  longlive2-sla-200step-crossroute-5pct-longlive2_standard_5pct-hsa_sla_cag \
  longlive2-hybrid-200step-dense-5pct \
  longlive2-hybrid-200step-sla_cag-5pct \
  longlive2-hybrid-200step-hsa_sla_cag-5pct; do
  test -f "runs/vbench/${run_id}/vbench_results.json"
  test -f "runs/vbench/${run_id}/aisbench_summary.md"
  echo "===== ${run_id} ====="
  cat "runs/vbench/${run_id}/aisbench_summary.md"
done
```

每组必须包含 16 个 VBench 官方维度及 `quality`、`semantic`、`total`。保留各 run 的 `manifest.json`、resolved YAML、AISBench 原始输出和视频，不要只保存终端表格。

## 7. 必须计算的质量差分

将新增结果与已有基础权重结果联合分析：

| 差分 | 研究问题 |
| --- | --- |
| SLA-trained dense - 基础 dense | SLA 主干 LoRA 是否造成分布漂移 |
| SLA-trained SLA - SLA-trained dense | SLA 训练后的运行时稀疏代价 |
| SLA-trained SLA - 基础 SLA | 200-step SLA 训练的恢复量 |
| Hybrid-trained Hybrid - Hybrid-trained dense | Hybrid 训练后的运行时稀疏代价；已有结果为 Total -3.12 分 |
| SLA-trained SLA - Hybrid-trained Hybrid | 两种方法各自适配后的最终部署质量；包含权重差异，不能解释为纯路由效应 |
| SLA-trained Hybrid - SLA-trained SLA | 固定 SLA 权重时 Hybrid 相对 SLA 的路由差异 |
| Hybrid-trained Hybrid - Hybrid-trained SLA | 固定 Hybrid 权重时 Hybrid 相对 SLA 的路由差异 |
| 上述两个路由差异之差 | 训练方法与推理路由是否存在明显交互或专用化 |

逐维重点检查 `Object Class`、`Multiple Objects`、`Spatial Relationship`、`Background Consistency`、`Subject Consistency` 和 `Imaging Quality`。小样本下的 Scene、Human Action 单项大幅波动不能单独作为算法改进证据。

## 8. 进入 20% 评测的门槛

本轮先完成 5% 消融，不立即重复所有方法的 20% 评测。满足以下条件后再选择候选进入统一 20% 或 Full VBench：

1. checkpoint、manifest、seed、提示词和 resolved 配置核对一致。
2. 候选 sparse 相对其同 checkpoint dense 的 Total 差距明显小于当前 Hybrid 的 3.12 分。
3. Multiple Objects、Spatial Relationship 和 Object Class 不再出现 10 分以上系统性下降，或已通过更低稀疏率消融确认可恢复。
4. 同 checkpoint 的 DiT-only 性能仍有明确收益；质量比较不能替代训练后权重性能重测。

如果 SLA-trained SLA 和 Hybrid-trained Hybrid 都未达到门槛，应先执行 `0.85/0.88/0.90` 稀疏率消融或延长训练，不应直接用 20% 评测扩大一个已明确失败的配置。
