# 昇腾 NPU 推理与评测指南

当前 NPU 分支只维护两条推理工作流：

- `scripts/run_msprof.sh`：固定提示词上的 msprof 性能采集与分析。
- `scripts/run_vbench.sh`：生成视频并运行 AISBench VBench Standard 质量评测。

SP、DP、视频帧数、数据集、随机种子和稀疏策略写在 YAML 中。Shell
脚本开头只保留可见卡、Python 环境、CANN 环境和端口等部署参数。

## 1. 配置入口

```text
configs/inference/
├── msprof.yaml
└── vbench.yaml
```

模型路径默认写在 YAML 中，也可以在启动时覆盖：

```bash
export LONGLIVE_MODEL_ROOT=/path/to/Wan2.2-TI2V-5B
export LONGLIVE_GENERATOR_CKPT=/path/to/longlive2_merged_generator.pt
```

两个配置都保留了 `sparsity` 段。当前为 `dense`，后续稀疏实现应通过该段
增加方法和参数，不需要复制整份配置或新增一组 Shell 脚本。

## 2. msprof 性能测试

### 2.1 时长 preset

`configs/inference/msprof.yaml` 只定义三档：

| preset | latent frames | 24 FPS 解码帧数 | 约时长 |
| --- | ---: | ---: | ---: |
| `16s` | 96 | 381 | 15.875 秒 |
| `32s` | 192 | 765 | 31.875 秒 |
| `64s` | 384 | 1533 | 63.875 秒 |

这里采用 Wan VAE 的关系 `pixel_frames = (latent_frames - 1) * 4 + 1`。
`16s/32s/64s` 是便于识别的任务名，并不表示输出恰好是整数秒。

默认布局为 `SP=4, DP=1`，并启用独立异步 VAE。因此需要 5 张可见卡：
4 张生成 worker 加 1 张 VAE 卡。SP/DP 和 VAE 模式只在 YAML 中修改。

### 2.2 启动

编辑 `scripts/run_msprof.sh` 顶部的环境变量，或在命令行覆盖：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4 \
GENERATION_ENV=/path/to/longlive-env \
bash scripts/run_msprof.sh 16s

bash scripts/run_msprof.sh 32s
bash scripts/run_msprof.sh 64s
```

脚本始终运行完整 msprof，不再提供 `--no-profile`。它会依次完成：

1. 根据 preset 生成本次运行的 resolved YAML。
2. 用 msprof 包裹 `torchrun inference_sp.py`。
3. 汇总 warmup 后的延迟和 FPS。
4. 运行算子、HCCL、通信矩阵、慢卡和 advisor 等分析。

msprof 会插桩并采集数据，因此日志中的延迟和 FPS 包含 profiler 开销。
这些数字适合同一套 msprof 参数下的版本对比，不应当作无侵入生产吞吐，也不应
与过去的 `--no-profile` 结果直接比较。

默认使用 `performance/prompts.txt` 的两条提示词，每个 rank 跳过第一条 warmup。
该数据集只服务性能复现，不参与 VBench 分数计算。

## 3. VBench Standard 评测

### 3.1 数据组织

```text
data/benchmarks/
├── performance/
│   └── prompts.txt
├── vbench_standard/
│   ├── full/
│   │   ├── prompts.txt
│   │   └── full_info.json
│   ├── 5pct/
│   │   ├── prompts.txt
│   │   └── full_info.json
│   └── 20pct/
│       ├── prompts.txt
│       └── full_info.json
├── vbench_augmented/
│   ├── full/prompts.txt
│   ├── 5pct/prompts.txt
│   └── 20pct/prompts.txt
└── vbench_long/
    └── evaluator_configs/
```

`standard` 是原始 VBench prompt；`augmented` 是与其逐行对齐的 Qwen2.5
seed-42 生成 prompt。增强集仅存生成文本，命名 prompt 和 `full_info.json`
复用对应的 Standard 子集，避免维护三份重复数据。

- Full：944 条唯一提示词。
- 5%：43 条提示词，来自 AISBench K-Means Mini。
- 20%：186 条提示词，来自另一轮独立 K-Means 抽样。

5% 和 20% 都来自 Full，但两者独立抽样；5% **不是** 20% 的子集。因此
`wan_mini` 更准确的名字是 `wan22_standard_5pct`，不能把它理解成 20% 的
进一步裁剪。

### 3.2 preset 与启动

`configs/inference/vbench.yaml` 对 `longlive2` 和 `wan22` 各提供六组 preset：

```text
<engine>_standard_{full,5pct,20pct}
<engine>_augmented_{full,5pct,20pct}
```

例如：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
GENERATION_ENV=/path/to/longlive-env \
AISBENCH_ENV=/path/to/aisbench-env \
VBENCH_CACHE_DIR=/path/to/vbench_models \
bash scripts/run_vbench.sh longlive2_standard_20pct

bash scripts/run_vbench.sh longlive2_augmented_5pct
bash scripts/run_vbench.sh wan22_standard_full
```

默认 `SP=2, DP=6`，使用 12 张 worker 卡，生成 125 个像素帧，并运行
seed 0 到 4。脚本会逐个 seed 生成、转换为 VBench 命名，最后执行一次
AISBench。修改布局、帧数或 seeds 时只编辑 YAML。

生成阶段会在终端显示全部 seeds 的视频进度、耗时、ETA 和平均生成时间；
AISBench 阶段会显示 16 个 VBench dimension 的完成进度。推理和评测的原始
输出分别保存在 `seed_<seed>.log` 和 `aisbench.log`，不会与动态进度条混排。

如果运行中断，可使用同一个 `RUN_ID` 继续；已经完整生成的 seed 会被跳过：

```bash
RUN_ID=<existing-run-id> bash scripts/run_vbench.sh longlive2_standard_20pct
```

## 4. VBench-Long 为什么单独存在

VBench-Long 不是第三类 prompt，也不是 Full、5%、20% 之外的新比例。它复用
VBench Standard Full 的 944 条 prompt 和 metadata，区别在评测协议：

- 根据维度用不同 clip length 切分长视频。
- 分别计算 within-clip 和 cross-clip 指标。
- 使用 slow/fast 参数以及 subject/background 映射进行聚合或校准。

因此它必须保留独立 evaluator config，但没有必要复制 Full 的 prompts 和
metadata。当前 AISBench/NPU 适配层只支持 Standard 协议，Long 的评测器尚未
完成昇腾适配，所以 `vbench.yaml` 不暴露 Long preset。将来接入时应新增评测
协议选择，而不是再新增一套 Long 数据集。

## 5. 输出与 run ID

```text
runs/
├── msprof/<run-id>/
│   ├── manifest.json
│   ├── resolved.yaml
│   ├── msprof.log
│   ├── summary.txt
│   ├── videos/
│   └── profiling/{raw,analysis}/
└── vbench/<run-id>/
    ├── manifest.json
    ├── resolved_seed_<seed>.yaml
    ├── seed_<seed>.log
    ├── aisbench.log
    ├── aisbench/<evaluation-session>/
    └── videos/{raw,prepared}/
```

默认 run ID 示例：

```text
msprof-longlive2-32s-async-sp4-dp1-dense-20260801_143000
vbench-longlive2-standard-20pct-125f-5seed-sp2-dp6-dense-20260801_143000
```

run ID 显式记录任务、引擎、数据类别/时长、帧数或 seed 数、SP、DP 和稀疏
方法；时间戳只负责区分多次运行。`manifest.json` 保存机器可读的最终参数，
`resolved*.yaml` 保存实际传给推理入口的配置。

## 6. 运行前检查

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python -c "import torch, torch_npu; print(torch.npu.is_available())"
msprof --help
msprof-analyze --help
```

项目依赖使用 `requirements_npu_bf16.txt`。`torch`、`torchvision` 和
`torch_npu` 必须与服务器 CANN 版本匹配，不要用普通 PyPI/CUDA wheel 覆盖。
