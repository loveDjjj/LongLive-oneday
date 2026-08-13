# 当前测试清单

本文件只记录当前一轮需要在昇腾服务器执行的临时命令。下一轮测试任务必须覆盖重写本文件；长期稳定命令应迁移到[环境安装与测试](setup_and_validation.md)。

## 测试目标

使用 VBench Standard 5% 子集，对 LongLive2.0 基础权重和完成 200 step 训练的 HSA+SLA+CAG 权重进行质量回归。

必测 5 组：

1. 基础权重 + LongLive2.0 原生 `dense` attention。
2. 基础权重 + `sla_cag` attention。
3. 基础权重 + `hsa_cag` attention。
4. 基础权重 + `hsa_sla_cag` attention。
5. 训练后权重 + `hsa_sla_cag` attention。

另增加“训练后权重 + `dense` attention”诊断组，用于区分主干 LoRA 后训练造成的质量变化和运行时稀疏造成的质量变化。

本轮固定使用 `longlive2_standard_5pct`：43 条原始提示词、5 个 seed、每组 215 个视频、125 pixel 帧、24 FPS、4 步采样、`SP2 × DP8`。5% 子集只用于快速回归，不能作为完整 VBench 发布结果。

## 1. 环境与代码

```bash
cd /mnt/share/r50063443/LongLive-oneday
git pull origin feat/unified-sparse-attention
git rev-parse --short HEAD

source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
conda activate /mnt/share/r50063443/conda_envs/longlive
```

预期提交不低于 `d9c5597`。确认 16 张卡均可用，并设置本轮公共路径：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export BASE_CKPT=/mnt/share/weight/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt
export HYBRID_STEP_DIR=runs/training/hsa_sla_cag_16card_200step/checkpoints/step_0000200
export TRAINED_CKPT=runs/merged/longlive2_hsa_sla_cag_200step.pt

test -f "${BASE_CKPT}"
```

## 2. 验收训练 checkpoint

训练完成后确认完整恢复文件、推理 sidecar 和自动校验结果存在：

```bash
test -f "${HYBRID_STEP_DIR}/train_state.pt"
test -f "${HYBRID_STEP_DIR}/generator_adapter.pt"
test -f "${HYBRID_STEP_DIR}/validation.json"
cat "${HYBRID_STEP_DIR}/validation.json"

python scripts/checkpoints/validate_linear_checkpoint.py \
  "${HYBRID_STEP_DIR}/train_state.pt" \
  --expected-method hsa_sla_cag \
  --expected-step 200 \
  --require-resume-state

python scripts/checkpoints/validate_linear_checkpoint.py \
  "${HYBRID_STEP_DIR}/generator_adapter.pt" \
  --expected-method hsa_sla_cag \
  --expected-step 200
```

两条校验均须输出 `linear_checkpoint=passed`，且 checkpoint 应为 `generator_train_scope=lora_plus_linear`、30 层、60 个非零且有限的原始 `sla_linear` 张量。

## 3. 导出训练后完整 Generator

VBench 使用完整 Generator checkpoint。将训练后的主干 LoRA 合并到基础 LongLive2.0 Generator，并加载原始 `sla_linear`：

```bash
mkdir -p runs/merged

ASCEND_RT_VISIBLE_DEVICES=0 \
python scripts/checkpoints/merge_lora.py \
  --config_path configs/train/hsa_sla_cag.yaml \
  --generator_ckpt "${BASE_CKPT}" \
  --lora_ckpt "${HYBRID_STEP_DIR}/generator_adapter.pt" \
  --output_path "${TRAINED_CKPT}" \
  --device npu:0

test -f "${TRAINED_CKPT}"
```

不要把 `train_state.pt` 或 `generator_adapter.pt` 直接作为 `LONGLIVE_GENERATOR_CKPT`。

## 4. 基础权重四方法矩阵

四组使用同一个基础 checkpoint、提示词、seed、分辨率、帧数、采样参数和设备集合：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
METHODS=dense,hsa_cag,sla_cag,hsa_sla_cag \
VBENCH_PRESETS=longlive2_standard_5pct \
SUITE_ID=longlive2-base-5pct \
bash scripts/evaluation/run_vbench_matrix.sh "${BASE_CKPT}"
```

脚本支持断点续跑。再次使用相同 `SUITE_ID` 时，已有 `vbench_results.json` 的完整 case 会跳过；缺少视频的 seed 会继续生成。

预期产生：

```text
runs/vbench/longlive2-base-5pct-longlive2_standard_5pct-dense/
runs/vbench/longlive2-base-5pct-longlive2_standard_5pct-hsa_cag/
runs/vbench/longlive2-base-5pct-longlive2_standard_5pct-sla_cag/
runs/vbench/longlive2-base-5pct-longlive2_standard_5pct-hsa_sla_cag/
runs/suites/longlive2-base-5pct/vbench/
```

## 5. 训练后 HSA+SLA+CAG

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
LONGLIVE_GENERATOR_CKPT="${TRAINED_CKPT}" \
LONGLIVE_SPARSE_METHOD=hsa_sla_cag \
RUN_ID=longlive2-hybrid-200step-hsa_sla_cag-5pct \
bash scripts/evaluation/run_vbench.sh longlive2_standard_5pct
```

该组必须确认 resolved 配置使用训练后完整权重、`hsa_sla_cag` 和 MindIE-SD RainFusion 后端。

## 6. 训练后 dense 诊断组

该组不是用户提出的最低 5 组之一，但必须保留，才能判断 LoRA 后训练是否改变了 dense 生成质量：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
LONGLIVE_GENERATOR_CKPT="${TRAINED_CKPT}" \
LONGLIVE_SPARSE_METHOD=dense \
RUN_ID=longlive2-hybrid-200step-dense-5pct \
bash scripts/evaluation/run_vbench.sh longlive2_standard_5pct
```

## 7. 结果验收

检查所有 6 组结果：

```bash
for run_id in \
  longlive2-base-5pct-longlive2_standard_5pct-dense \
  longlive2-base-5pct-longlive2_standard_5pct-hsa_cag \
  longlive2-base-5pct-longlive2_standard_5pct-sla_cag \
  longlive2-base-5pct-longlive2_standard_5pct-hsa_sla_cag \
  longlive2-hybrid-200step-hsa_sla_cag-5pct \
  longlive2-hybrid-200step-dense-5pct; do
  test -f "runs/vbench/${run_id}/vbench_results.json"
  test -f "runs/vbench/${run_id}/aisbench_summary.md"
  echo "===== ${run_id} ====="
  cat "runs/vbench/${run_id}/aisbench_summary.md"
done
```

每组必须包含 16 个 VBench 官方维度，以及官方 `quality`、`semantic` 和 `total` 聚合分数。重点比较：

1. 基础 sparse 与基础 dense：未经后训练适配时三种稀疏路由的质量损失。
2. 训练后 hybrid 与基础 hybrid：200 step 后训练的补偿收益。
3. 训练后 hybrid 与训练后 dense：适配后的运行时稀疏质量损失。
4. 训练后 dense 与基础 dense：主干 LoRA 后训练本身造成的质量变化。

基础权重运行 SLA+CAG 或混合方法时，缺失的 `sla_linear` 会严格初始化为零；训练后完整权重包含非零原始补偿层。HSA 使用默认 `0.85/0.95` 预算，SLA 与混合方法使用 `0.90/0.93`，因此本轮比较的是各方法当前默认配置，而不是相同实际稀疏率下的纯路由消融。
