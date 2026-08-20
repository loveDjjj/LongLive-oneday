# 当前测试清单

本轮验证评测和训练入口统一使用 `LONGLIVE_SP_SIZE`，矩阵入口可传逗号列表；新增 `dense_prefix_chunks` / `LONGLIVE_DENSE_PREFIX_CHUNKS` 控制前缀 AR chunk 走 dense。服务器 NPU 项目当前为**待服务器验证**；长期稳定命令见[环境安装与测试](setup_and_validation.md)。

## 1. 主机回归测试

```bash
cd /mnt/share/r50063443/LongLive-oneday
conda activate /mnt/share/r50063443/conda_envs/longlive

rg -n 'SP[_]SIZES=' docs scripts tests || true

python -m pytest -q \
  tests/test_inference_config_resolution.py \
  tests/test_evaluation_matrix.py \
  tests/test_shell_launchers.py \
  tests/test_hsa_cag.py \
  tests/test_sla_cag.py \
  tests/test_hsa_sla_cag.py \
  tests/test_training_config_contract.py

python - <<'PY'
from omegaconf import OmegaConf

for path in (
    "configs/train/hsa_cag.yaml",
    "configs/train/sla_cag.yaml",
    "configs/train/hsa_sla_cag.yaml",
    "configs/inference/msprof.yaml",
    "configs/inference/vbench.yaml",
):
    config = OmegaConf.load(path)
    print(path, "ok")
PY

DRY_RUN=1 METHODS=dense DURATIONS=5s MODES=dit_only \
LONGLIVE_SP_SIZE=1 PERF_DEVICES=10 \
bash scripts/evaluation/run_performance_matrix.sh

LONGLIVE_SP_SIZE=1 LONGLIVE_SPARSE_METHOD=sla_cag \
LONGLIVE_DENSE_PREFIX_CHUNKS=2 \
python scripts/evaluation/resolve_config.py benchmark \
  --config configs/inference/msprof.yaml \
  --preset 5s \
  --output /tmp/longlive-sp1-prefix2.yaml

python -m compileall -q \
  train.py inference_sp.py pipeline trainer utils wan_5b scripts tests
git diff --check
```

预期：不再出现面向用户的旧矩阵 SP 变量示例；相关 pytest 全部通过；YAML 均可解析；矩阵 dry-run 使用 `sp=1` 且只需要 1 张 NPU；resolved 配置包含 `sp_size: 1` 和 `dense_prefix_chunks: 2`；编译检查与空白检查通过。

## 2. 服务器待验证

单卡 SP1 DiT-only 使用：

```bash
cd /mnt/share/r50063443/LongLive-oneday
conda activate /mnt/share/r50063443/conda_envs/longlive

ASCEND_RT_VISIBLE_DEVICES=10 \
LONGLIVE_SP_SIZE=1 \
LONGLIVE_GENERATOR_CKPT=/mnt/share/r50063443/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt \
LONGLIVE_SPARSE_METHOD=sla_cag \
LONGLIVE_DENSE_PREFIX_CHUNKS=1 \
BENCHMARK_MODE=dit_only \
RUN_ID=sla_cag-sp1-dit-only-5s \
bash scripts/evaluation/run_benchmark.sh 5s
```

如需让前 2 个 AR chunk 保持 dense，将 `LONGLIVE_DENSE_PREFIX_CHUNKS=2`。当前实现按整段视频 chunk 编号计算，不会在 multi-shot 边界自动重置。

## 3. 结果回填

完成服务器验证后，将以下信息回填到本文件顶部或提交说明中：

```text
test_id=
device=
LONGLIVE_SP_SIZE=
LONGLIVE_DENSE_PREFIX_CHUNKS=
resolved_config_path=
是否解析到预期 sp_size 和 dense_prefix_chunks=
异常与日志路径=
```
