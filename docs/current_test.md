# 当前测试清单

本轮验证仓库不再固化个人服务器绝对路径，配置和启动脚本改用 `/path/to/...` 占位默认值，并继续支持通过环境变量覆盖真实部署路径。服务器 NPU 项目当前为**待服务器验证**；长期稳定命令见[环境安装与测试](setup_and_validation.md)。

## 1. 主机回归测试

```bash
cd /path/to/LongLive-oneday
conda activate /path/to/conda_envs/longlive

rg -n '/mnt/a800[_]share/r50063[0-9][0-9]3|/mnt/share/weight/LongLive/checkpoints/longlive2[_]5b|LongLive/checkpoints/longlive2[_]5b' . || true

python -m pytest -q \
  tests/test_inference_config_resolution.py \
  tests/test_training_config_contract.py \
  tests/test_evaluation_matrix.py

python - <<'PY'
from omegaconf import OmegaConf

for path in (
    "configs/train/hsa_cag.yaml",
    "configs/train/sla_cag.yaml",
    "configs/train/hsa_sla_cag.yaml",
    "configs/inference/msprof.yaml",
    "configs/inference/vbench.yaml",
):
    OmegaConf.load(path)
    print(path, "ok")
PY

python -m compileall -q \
  train.py inference_sp.py pipeline trainer utils wan_5b scripts tests
git diff --check
```

预期：`rg` 不输出旧个人服务器路径；配置解析、训练配置契约和矩阵展开测试全部通过；YAML 均可解析；编译检查与空白检查通过。

## 2. 服务器待验证

本轮只清理路径默认值，不改变训练、推理和稀疏算法行为。若在服务器运行训练或评测入口，需先设置真实部署路径：

```bash
cd /path/to/LongLive-oneday
conda activate /path/to/conda_envs/longlive

export GENERATION_ENV=/path/to/conda_envs/longlive
export AISBENCH_ENV=/path/to/conda_envs/aisbench_npu
export VBENCH_CACHE_DIR=/path/to/vbench_models
export CANN_ENV_SCRIPT=/path/to/cann-8.5/Ascend/cann-8.5.0/set_env.sh
export LONGLIVE_MODEL_ROOT=/path/to/Wan2.2-TI2V-5B
export LONGLIVE_GENERATOR_CKPT=/path/to/longlive2_merged_generator.pt
export MODEL_ROOT="${LONGLIVE_MODEL_ROOT}"
export GENERATOR_CKPT="${LONGLIVE_GENERATOR_CKPT}"
```

预期：使用真实路径覆盖后，`scripts/training/run_sparse_cag.sh`、`scripts/evaluation/run_benchmark.sh`、`scripts/evaluation/run_msprof.sh` 和 `scripts/evaluation/run_vbench.sh` 能解析到实际模型、环境和 checkpoint 路径。NPU 训练/推理仍为待服务器验证。

## 3. 结果回填

完成服务器验证后，将以下信息回填到本文件顶部或提交说明中：

```text
test_id=
device=
真实路径覆盖变量=
入口脚本=
resolved_config_path=
是否解析到实际 checkpoint=
异常与日志路径=
```
