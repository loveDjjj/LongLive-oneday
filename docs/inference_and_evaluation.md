# LongLive2.0 昇腾推理与评测指南

本文是当前推理、性能和质量评测的统一说明。环境与算子测试见[环境安装与测试](setup_and_validation.md)，checkpoint 训练和导出见[训练指南](training.md)。

## 1. 脚本层级

`scripts/evaluation/` 按调用职责分为三层。日常实验优先使用主入口；只有调试单个 case 或采集专项 profile 时才直接调用单项入口。内部 helper 由启动脚本调用，不作为日常实验入口。

### 1.1 主入口

| 脚本 | 职责 |
| --- | --- |
| `run_performance_matrix.sh` | 编排方法、时长、SP 布局和 VAE 模式，调用 benchmark 或 msprof 单项入口并生成统一汇总。 |
| `run_vbench_matrix.sh` | 编排多个方法与 VBench preset，支持方法级 checkpoint、断点续跑和统一汇总。 |

### 1.2 单项入口

| 脚本 | 职责 |
| --- | --- |
| `run_benchmark.sh` | 运行一个无 profiler 性能 case，保存 resolved 配置、视频或 latent、原始日志和延迟汇总。 |
| `run_msprof.sh` | 运行一个生成过程 profile case，并执行 DiT、通信和慢 rank 分析。 |
| `run_vae_msprof.sh` | 从已有 latent 单独采集 VAE decode profile，不重复运行 DiT。 |
| `run_vbench.sh` | 运行一个 VBench preset，负责多 seed 生成、断点恢复、视频整理、AISBench 评测和结果汇总。 |

### 1.3 内部 helper

| 脚本 | 职责 |
| --- | --- |
| `resolve_config.py` | 将紧凑 YAML preset 和环境变量解析为运行时 resolved YAML 与 manifest 元数据。 |
| `summarize_benchmark.py` | 从单 case 日志提取延迟、吞吐、显存和异步 VAE 指标。 |
| `summarize_vbench.py` | 从 AISBench 产物提取 16 个官方维度及官方聚合分数。 |
| `summarize_suite.py` | 汇总性能、msprof 或 VBench 矩阵为 CSV/JSON，并按相同 SP/DP 匹配 dense 基线。 |

不要为固定方法、固定时长或固定数据子集增加一次性包装脚本。使用矩阵入口的 `METHODS`、`DURATIONS`、`MODES`、`SP_SIZES` 和 `VBENCH_PRESETS` 缩小范围，避免入口功能重复。

## 2. 支持矩阵

LongLive2.0 入口 `inference_sp.py` 支持：

| 方法 | 稀疏路径 | 推理后端 |
| --- | --- | --- |
| `dense` | 不启用 | 原生 dense attention |
| `hsa_cag` | 帧级 HSA + block sparse + CAG | MindIE-SD RainFusion |
| `sla_cag` | 全局 Smooth-K block Top-K + CAG + 线性补偿 | MindIE-SD RainFusion |
| `hsa_sla_cag` | HSA 候选帧 + SLA block Top-K + CAG + 线性补偿 | MindIE-SD RainFusion |

Wan2.2 原生入口 `inference_wan22_sp.py` 只支持 dense。设置稀疏方法时会在配置解析阶段失败。

训练后的权重与运行时稀疏是两个变量。正式质量对比至少包含：基础 checkpoint dense、训练后 checkpoint dense、训练后 checkpoint sparse。前两者衡量后训练影响，后两者衡量运行时稀疏影响。

## 3. 配置与路径

```text
configs/inference/msprof.yaml
configs/inference/vbench.yaml
```

常用路径覆盖：

```bash
export GENERATION_ENV=/mnt/a800_share/r50063443/conda_envs/longlive
export AISBENCH_ENV=/mnt/a800_share/r50063443/conda_envs/aisbench_npu
export VBENCH_CACHE_DIR=/mnt/a800_share/r50063443/vbench_models
export CANN_ENV_SCRIPT=/mnt/a800_share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
export LONGLIVE_MODEL_ROOT=/mnt/a800_share/r50063443/Wan2.2-TI2V-5B
export LONGLIVE_GENERATOR_CKPT=/mnt/a800_share/r50063443/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt
```

评测其他合并权重时，再将 `LONGLIVE_GENERATOR_CKPT` 覆盖为对应 checkpoint 路径。

`train_state.pt` 不能直接当作完整 Generator。先按照[训练指南](training.md)导出合并权重。推理加载器支持 `{"generator": state_dict}`、`{"model": state_dict}`、raw state dict，以及配置开启 EMA 时的 `generator_ema`。

## 4. 稀疏方法与稀疏率

LongLive2.0 采用 32 个 latent 帧的滚动 KV 窗口，每个 AR chunk 生成 8 帧。超过 32 帧后丢弃最旧 KV，这是模型训练和推理契约的一部分；不要为了追求更高稀疏率直接改变窗口长度，除非重新训练并完成质量评测。

### 4.1 HSA+CAG

HSA 先在 latent 帧层面筛选历史，默认保留 6 帧，其中包含 1 个 sink 帧、2 个相邻帧和动态重要帧；当前 8 帧保持 dense。之后只对保留帧对应的 blocks 计算注意力。CAG 根据 AR chunk 位置调整目标预算，默认目标/基准稀疏率为 `0.85/0.95`。

### 4.2 SLA+CAG

SLA 不先丢弃 latent 帧，而是对滚动 KV 中所有 128-token blocks 计算 Smooth-K 代表并全局 Top-K。sink 与 recent 帧 blocks 强制保留，当前 chunk 默认也参与稀疏。完整 KV 同时进入线性统计分支作为补偿。默认目标/基准稀疏率为 `0.90/0.93`，与混合方法一致，以保证两种路由的性能和质量对比使用相同预算。

### 4.3 HSA+SLA+CAG

混合方法先由 HSA 将 32 帧缩小为 8 个候选帧，再由 SLA 在候选帧的 blocks 中继续 Top-K，最终只计算选中 blocks 的稀疏 softmax；线性补偿仍覆盖完整 KV。默认目标/基准稀疏率为 `0.90/0.93`。

SP4、32 秒尾部 shape 为 `Q=7040=55x128`、`KV=28160=220x128`。混合方法当前通常选中约 20 至 22 个/220 个 KV blocks，即约 90% 有效稀疏率。CAG 会使不同 chunk 的实际值变化，必须以 `selected/total` 日志为准，不能只引用配置中的目标值。

第一块或历史不足时会回退 dense。相同 chunk 的多个去噪 step 可复用历史 K summaries 和线性统计，但 query、当前 K、Top-K 和最终 LUT 仍需逐层逐步更新。

### 4.4 物理稀疏边界

当前稀疏 softmax 不是“先算完整 QK，再用 mask 清零”的逻辑稀疏：

- MindIE-SD RainFusion 接收紧凑的 `select_idx` 和每行 `select_num_idx`，算子接口只声明选中 KV blocks。
- Ascend Triton 前向和反向都只在 `SELECTED_BLOCKS` 循环中按 LUT 加载 K/V，不构造完整 QK 矩阵。
- NPU smoke 会扰动所有未选 K/V blocks；稀疏输出必须保持不变，而 dense 输出必须变化。

但 SLA 与混合方法的完整 attention 模块并非只访问选中 KV：路由阶段需要读取 K 的 block/frame 代表，线性补偿分支有意使用完整 KV 统计量。准确表述应为“稀疏 softmax 主分支只计算选中 blocks，路由和线性补偿仍覆盖完整 KV”。HSA+CAG 没有线性补偿，但仍需计算帧/block 路由摘要。

## 5. 算子微基准

```bash
test_id="sparse-kernel-$(date +%Y%m%d-%H%M%S)"
mkdir -p "logs/tests/${test_id}"
for method in hsa_cag sla_cag hsa_sla_cag; do
  ASCEND_RT_VISIBLE_DEVICES=15 \
  python tests/npu/benchmark_sparse_attention.py \
    --method "${method}" --backend mindiesd --device npu:0 \
    --latent-frames 192 --warmup 5 --iterations 20 \
    | tee "logs/tests/${test_id}/${method}-mindiesd-32s.txt"
done
```

报告中应同时记录 dense、未缓存路由、缓存路由、稀疏 kernel、线性补偿、完整稀疏路径、选中 block 数与有效稀疏率。只有完整稀疏路径低于同次 dense，才有进入整网测试的价值。

微基准不能替代 DiT-only：它不包含 30 层调用、缓存命中差异、SP 通信、其他 attention 分支和 Python 调度开销。

`mindiesd_bsa` 需要 `libopapi.so` 提供 `aclnnBlockSparseAttentionV2`。若报该符号缺失，应继续使用 RainFusion；重复 `source` 同一 CANN 不会补充算子。

## 6. 性能模式

统一支持三种模式，并支持通过 `LONGLIVE_SP_SIZE` 或性能矩阵的 `SP_SIZES` 在 SP1 与 SP4 间切换：

| 模式 | NPU 数 | 行为 | 用途 |
| --- | ---: | --- | --- |
| `dit_only` | SP 数 | 不加载 VAE，只保存 latent | 判断稀疏是否加速 DiT |
| `sync_vae` | SP 数 | SP leader 同步解码完整视频 | 观察串行 VAE 代价 |
| `async_vae` | SP 数 + 1 | SP worker + 1 张专用 VAE 卡 | 测量真实异步关键路径 |

当前 `async_vae` 确实使用后台队列/线程，在独立 NPU 上执行分块 `cached_decode`。它不是把 DiT 时间和 VAE 时间简单相加；最终墙钟时间由重叠后的关键路径和 drain 决定。应同时查看 `ar_loop_seconds`、`vae_decode_seconds`、`vae_enqueue_seconds`、`vae_drain_seconds`、`vae_overlap_seconds`、chunk 数和队列峰值。

## 7. 单项无 profiler 基准

默认执行 1 次预热和 3 次有效测量。先做相同 checkpoint 的 dense/sparse DiT-only：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
LONGLIVE_GENERATOR_CKPT=/path/to/merged_generator.pt \
LONGLIVE_SPARSE_METHOD=dense BENCHMARK_MODE=dit_only \
RUN_ID=dense-dit-32s \
bash scripts/evaluation/run_benchmark.sh 32s

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
LONGLIVE_GENERATOR_CKPT=/path/to/merged_generator.pt \
LONGLIVE_SPARSE_METHOD=sla_cag BENCHMARK_MODE=dit_only \
RUN_ID=sla-dit-32s \
bash scripts/evaluation/run_benchmark.sh 32s
```

`dit_only` 的 FPS/RTF 为 0 是预期现象，只比较 `generation_seconds` 的 p50。禁止比较不同 checkpoint、不同设备集合或单次结果。

再测试异步端到端：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,15 \
LONGLIVE_GENERATOR_CKPT=/path/to/merged_generator.pt \
LONGLIVE_SPARSE_METHOD=sla_cag BENCHMARK_MODE=async_vae \
RUN_ID=sla-async-32s \
bash scripts/evaluation/run_benchmark.sh 32s
```

已有历史实验中，32 秒 SP4 的 SLA DiT-only p50 从 `48.656 s` 降至 `36.240 s`，降低约 25.5%；HSA 从 `48.614 s` 降至 `40.059 s`，降低约 17.6%。但当时包含异步 VAE 的 SLA p50 为 `127.555 s`，dense 为 `121.983 s`，端到端反而变慢。这些数据只说明“稀疏 DiT 有收益，但旧端到端关键路径未获益”，不能作为当前代码的发布结论，必须按统一矩阵重测。

## 8. 完整性能矩阵

默认矩阵包含 4 种方法、3 个时长、SP1/SP4 和 3 种模式，共 72 个 case：

```bash
DRY_RUN=1 SP_SIZES=1,4 PERF_DEVICES=0,1,2,3,15 \
bash scripts/evaluation/run_performance_matrix.sh
```

确认展开正确后运行：

```bash
SP_SIZES=1,4 PERF_DEVICES=0,1,2,3,15 TASK=benchmark \
SUITE_ID=sparse-release-01 \
bash scripts/evaluation/run_performance_matrix.sh
```

可缩小范围：

```bash
METHODS=dense,hsa_cag DURATIONS=32s MODES=dit_only,async_vae SP_SIZES=4 \
PERF_DEVICES=0,1,2,3,15 TASK=benchmark SUITE_ID=hsa-retest-01 \
bash scripts/evaluation/run_performance_matrix.sh
```

方法可分别指定 checkpoint：

```text
DENSE_GENERATOR_CKPT
HSA_CAG_GENERATOR_CKPT
SLA_CAG_GENERATOR_CKPT
HSA_SLA_CAG_GENERATOR_CKPT
```

未指定时统一使用 `LONGLIVE_GENERATOR_CKPT`。`RESUME_SUITE=1` 会跳过已有 `summary.json` 的完整用例；不完整目录不会被覆盖。无 profiler 汇总写入 `runs/suites/<suite-id>/performance/`，生成过程 msprof 汇总写入 `runs/suites/<suite-id>/msprof/dit/`。

## 9. VAE-only 测试

复用 DiT-only 产物，避免重复生成 latent：

```bash
python tests/npu/benchmark_vae_decode.py \
  --latent runs/performance/dense-dit-32s/videos/rank0-1-0_regular_sp4.pt \
  --device npu:15 --chunk-frames 8 --iterations 1 \
  | tee logs/tests/vae-decode/vae-chunk8.txt
```

该测试复现异步 VAE worker 的 cached decode、逐 chunk pinned DtoH、CPU 拼接和归一化，并拆分 `vae_device_seconds`、`dtoh_device_seconds` 与 `cpu_post_seconds`。

已有同一 192-frame latent 测试中，chunk 8 和 16 的完整解码均约 `114.9 s`，其中 NPU VAE 约 `114.1 s`，DtoH device 时间约 `0.194 s`。因此当时瓶颈在 VAE 计算而非 DtoH 或 chunk 大小。msprof 中 Conv3DV2 累计时间最高，其后为 Conv2D、TransData、Transpose 和 Pad。

后续 VAE 优化优先级：减少格式转换和冗余 pad/concat；验证更合适的 Conv3D layout 与算子实现；再评估时间切分或空间切分的多卡 VAE。卷积并行必须处理 temporal halo、padding、cache 边界和归一化统计，并用同一 latent 对比帧数、边界误差与画质，不能只比较速度。

采集 VAE-only profile：

```bash
ASCEND_RT_VISIBLE_DEVICES=15 RUN_ID=vae-16f-baseline \
bash scripts/evaluation/run_vae_msprof.sh \
  runs/performance/dense-dit-32s/videos/rank0-1-0_regular_sp4.pt
```

## 10. msprof

msprof 用于定位算子和通信，不用于发布延迟。profile 插桩本身有开销，必须先得到稳定的无 profiler p50。

采集 DiT-only：

```bash
METHODS=dense,sla_cag DURATIONS=32s MODES=dit_only \
PERF_DEVICES=0,1,2,3 TASK=msprof SUITE_ID=sla-dit-profile-01 \
bash scripts/evaluation/run_performance_matrix.sh
```

采集异步端到端：

```bash
METHODS=dense,sla_cag DURATIONS=32s MODES=async_vae \
PERF_DEVICES=0,1,2,3,15 TASK=msprof SUITE_ID=sla-async-profile-01 \
bash scripts/evaluation/run_performance_matrix.sh
```

默认启用 AICore PMU。若环境在 PMU 初始化时报 `DrvFftsProfileStart failed` 或 `561103`，可用新的 `RUN_ID` 和 `MSPROF_AI_CORE=false` 重跑，以区分模型问题和 PMU 环境问题；关闭 PMU 的结果不能用于 AICore 流水线利用率结论。

## 11. VBench 质量评测

当前维护 VBench Standard 的 Full、5% 和 20%，以及与其逐行对齐的增强提示词版本。5% 与 20% 是独立抽样，5% 不是 20% 的子集。原生 Wan2.2 仅运行 dense。

单项评测：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
LONGLIVE_GENERATOR_CKPT=/path/to/merged_generator.pt \
LONGLIVE_SPARSE_METHOD=hsa_sla_cag \
RUN_ID=hybrid-20pct \
bash scripts/evaluation/run_vbench.sh longlive2_standard_20pct
```

统一矩阵：

```bash
VBENCH_PRESETS=longlive2_standard_20pct \
DENSE_GENERATOR_CKPT=/path/to/dense.pt \
HSA_CAG_GENERATOR_CKPT=/path/to/hsa.pt \
SLA_CAG_GENERATOR_CKPT=/path/to/sla.pt \
HSA_SLA_CAG_GENERATOR_CKPT=/path/to/hybrid.pt \
SUITE_ID=vbench-release-01 \
bash scripts/evaluation/run_vbench_matrix.sh
```

正式对比必须固定提示词版本、checkpoint、seed、分辨率、帧数、FPS 和采样步数。最终只采用 AISBench 输出的 16 个官方维度及 `quality`、`semantic`、`total` 汇总，不自行更改维度权重。

## 12. 结果判定

一项稀疏方法进入默认推理路径前，应同时满足：

1. 推理算子数值 smoke test 通过，且真实 shape 的完整稀疏路径快于 dense。
2. 相同 checkpoint 的 32 秒 DiT-only p50 稳定优于 dense，并至少有 3 次有效测量。
3. VBench 质量下降在预先约定范围内，且 dense/sparse 使用完全一致的评测设置。
4. 同步和异步 VAE 模式分别报告，不用 DiT 收益替代端到端收益。
5. 64 秒结果不出现缓存增长、OOM、队列阻塞或稀疏率异常。
6. msprof 结论与无 profiler 的墙钟结果一致，不使用 profile 延迟计算发布加速比。
