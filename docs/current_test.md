# 当前测试清单

本轮验证 `hsa_cag` 新增 `hsa_history_mode: rolling | full`，其中 `full` 为 HSA 提供完整已完成历史 KV，同时保留 `protect_longlive_sink_frames`、`keep_near_history_frames` 和 current8 候选语义。服务器 NPU 项目当前均为**待服务器验证**，长期稳定命令见[环境安装与测试](setup_and_validation.md)。

## 1. 主机回归测试

```bash
cd /mnt/a800_share/r50063443/LongLive-oneday
conda activate /mnt/a800_share/r50063443/conda_envs/longlive

python -m pytest -q \
  tests/test_hsa_cag.py \
  tests/test_inference_config_resolution.py \
  tests/test_training_config_contract.py

python -m compileall -q wan_5b pipeline utils scripts tests
python -m compileall -q inference_sp.py
git diff --check
```

预期：测试全部通过；`hsa_cag` 默认 resolved 配置为 `hsa_history_mode=rolling`，设置 `LONGLIVE_HSA_HISTORY_MODE=full` 后 resolved 配置切换为 `full`；HSA frame router 在 40 帧 history 的单测中可以选中 rolling 32 帧窗口之前的历史帧。

## 2. 服务器待验证

```bash
cd /mnt/a800_share/r50063443/LongLive-oneday
conda activate /mnt/a800_share/r50063443/conda_envs/longlive

export LONGLIVE_SPARSE_METHOD=hsa_cag
export LONGLIVE_HSA_HISTORY_MODE=full
export LONGLIVE_HSA_FRAME_DEBUG=1
export LONGLIVE_HSA_FRAME_DEBUG_MAX_ROWS=16

python scripts/resolve_inference_config.py msprof \
  --config configs/inference/msprof.yaml \
  --preset 32s \
  --output /tmp/longlive-hsa-full-32s.yaml

torchrun --standalone --nproc_per_node=4 \
  inference_sp.py \
  --config /tmp/longlive-hsa-full-32s.yaml
```

预期：resolved 配置包含 `hsa_history_mode: full`；debug 日志出现 `[hsa-frame-route] mode=full`，并且在超过 32 帧后允许 `contains_pre_window_frame=true`，例如 current frame 已到 96 附近时可以选择 frame 17。多 prompt benchmark 中每条视频结束后会执行 `pipeline.clear_cache()` 和 `empty_cache()`，下一条 prompt 的 T5 text encoder 不应再被上一条视频的 full-history KV cache 顶爆。

## 3. 结果回填

完成服务器验证后，将以下信息回填到本文件顶部或提交说明中：

```text
test_id=
device=
hsa_history_mode=
是否出现 pre-window selected frame=
resolved_config_path=
多 prompt 是否通过=
异常与日志路径=
```
