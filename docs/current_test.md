# 当前测试清单

本文件只记录当前提交需要在昇腾服务器执行的临时测试。下一轮测试任务必须覆盖重写本文件；长期稳定命令应迁移到[环境安装与测试](setup_and_validation.md)。

## 测试目标

验证统一稀疏分发器可以直接接收模型层预解析后的 `HSAAttentionConfig`，并恢复 HSA+CAG 的真实 SP1 推理。原失败目录保留为证据，不覆盖、不删除。

## 1. 更新与主机回归

```bash
cd /mnt/share/r50063443/LongLive-oneday
git pull origin feat/unified-sparse-attention

conda activate /mnt/share/r50063443/conda_envs/longlive
PYTHONPATH=. pytest -q tests/test_hsa_cag.py tests/test_sla_cag.py tests/test_hsa_sla_cag.py
```

预期全部通过，尤其是 `test_dispatcher_accepts_parsed_hsa_config_in_attention_path`。

## 2. HSA+CAG 真实推理回归

失败的 `base-weight-speed-20260812-165636-hsa_cag-5s-dit_only-sp1` 已是不完整 run。使用新 ID 运行一个 5 秒 SP1 case：

```bash
source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh

export LONGLIVE_GENERATOR_CKPT=/mnt/share/weight/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt
export SUITE_ID="hsa-dispatch-fix-$(date +%Y%m%d-%H%M%S)"

METHODS=hsa_cag DURATIONS=5s SP_SIZES=1 MODES=dit_only PERF_DEVICES=4 \
bash scripts/evaluation/run_performance_matrix.sh
```

预期不再出现 `HSAAttentionConfig object is not iterable`，完成 1 次预热和 3 次有效测量，并生成：

```text
runs/performance/${SUITE_ID}-hsa_cag-5s-dit_only-sp1/summary.json
runs/suites/${SUITE_ID}/performance/results.csv
```

## 3. 恢复完整性能矩阵

定向回归通过后使用新的 suite ID 重跑完整矩阵。不要复用含失败目录的旧 ID：

```bash
export SUITE_ID="base-weight-speed-fixed-$(date +%Y%m%d-%H%M%S)"
export METHODS=dense,hsa_cag,sla_cag,hsa_sla_cag
export DURATIONS=5s,32s,64s
export TASK=benchmark

SP_SIZES=1 MODES=dit_only PERF_DEVICES=4 \
bash scripts/evaluation/run_performance_matrix.sh
```

该命令先恢复当前中断的 SP1 DiT-only 阶段。后续 SP4、同步 VAE 和异步 VAE 命令仍按[推理与评测](inference_and_evaluation.md)中的矩阵流程执行，并在同一终端保持新的 `SUITE_ID` 不变。
