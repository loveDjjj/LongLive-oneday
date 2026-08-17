# 当前测试清单

本轮验证 HSA+SLA+CAG 已移除 `hard_keep_sink_frames` / `hard_keep_recent_frames`，并将 global sink / shot sink 各限制为前 2 帧进入候选池。服务器 NPU 项目当前均为**待服务器验证**，长期稳定命令见[环境安装与测试](setup_and_validation.md)。

## 1. 主机回归测试

```bash
cd /mnt/a800_share/r50063443/LongLive-oneday
conda activate /mnt/a800_share/r50063443/conda_envs/longlive

python -m pytest -q \
  tests/test_hsa_sla_cag.py \
  tests/test_inference_config_resolution.py \
  tests/test_training_config_contract.py

python -m compileall -q wan_5b utils tests
git diff --check
```

预期：测试全部通过；HSA+SLA+CAG resolved 配置中包含 `max_global_sink_frames=2`、`max_shot_sink_frames=2`，且不再出现 `hard_keep_sink_frames` / `hard_keep_recent_frames`。

## 2. 服务器待验证

待在 NPU 服务器上补跑 `tests/npu/benchmark_sparse_attention.py --method hsa_sla_cag` 和相关推理矩阵，确认新的候选帧上限在实际路由日志中生效。

## 3. 结果回填

完成服务器验证后，将以下信息回填到本文件顶部或提交说明中：

```text
test_id=
device=
HSA+SLA+CAG candidate sink frames=
HSA+SLA+CAG training backward=
异常与日志路径=
```
