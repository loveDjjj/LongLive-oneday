# 昇腾 NPU 推理与评测指南

本文覆盖当前维护的两条评测工作流：`scripts/evaluation/run_vbench.sh` 生成视频并执行 AISBench VBench Standard，`scripts/evaluation/run_msprof.sh` 采集 LongLive2 推理性能。环境准备见 [环境安装与测试](getting_started.md)，LoRA checkpoint 和合并方法见 [SLA+CAG 训练指南](sla_cag_training.md)。

## 1. 支持矩阵

| 工作流 | 模型类别 | Dense | SLA+CAG | 入口 |
| --- | --- | --- | --- | --- |
| VBench | LongLive2 causal Generator | 支持 | 支持 | `inference_sp.py` |
| VBench | Wan2.2-TI2V-5B 原生模型 | 支持 | 不支持 | `inference_wan22_sp.py` |
| msprof | LongLive2 causal Generator | 支持 | 支持 | `inference_sp.py` |

SLA+CAG 是运行时 attention 路由，不是另一份 checkpoint。同一合并权重分别跑 dense 和 sparse，才能隔离稀疏计算造成的质量与性能变化。Wan2.2 原生入口启用 SLA 会在配置解析阶段报错。

当前 VBench 只执行 Standard 评测协议。`data/benchmarks/vbench_long/` 保留 Long 所需映射和 clip 配置，但 AISBench/NPU Long 适配尚未完成，不能通过当前 preset 启动。

## 2. 配置、环境变量和权重

```text
configs/inference/vbench.yaml
configs/inference/msprof.yaml
```

YAML 管理模型结构、SP/DP、帧数、seed、数据和稀疏参数；Shell 变量管理部署路径、可见卡和端口。

通用覆盖：

```bash
export GENERATION_ENV=/path/to/longlive-env
export CANN_ENV_SCRIPT=/mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
export LONGLIVE_MODEL_ROOT=/path/to/Wan2.2-TI2V-5B
export LONGLIVE_GENERATOR_CKPT=/path/to/merged_generator.pt
```

`LONGLIVE_GENERATOR_CKPT` 只影响 LongLive2；Wan2.2 原生入口从 `LONGLIVE_MODEL_ROOT` 加载官方模型目录。

训练生成的 `train_state.pt` 或旧 `checkpoint_model_*/model.pt` 只包含 LoRA 时，不能直接作为 Generator。先运行：

```bash
python scripts/checkpoints/merge_lora.py \
  --config_path configs/train/sla_cag.yaml \
  --generator_ckpt /path/to/longlive2_merged_generator.pt \
  --lora_ckpt /path/to/train_state.pt \
  --output_path runs/merged/longlive2_sla_cag_2k.pt \
  --device npu:0
```

推理加载器支持完整 BF16 `{"generator": state_dict}`、`{"model": state_dict}`、raw state dict，或在 `use_ema=true` 时加载 `generator_ema`。VBench resolver 不现场加载 LoRA，因此正式评测使用预先合并的 checkpoint。

## 3. SLA+CAG 推理开关

Dense 不设置环境变量：

```bash
bash scripts/evaluation/run_vbench.sh longlive2_standard_5pct
```

SLA+CAG 设置：

```bash
LONGLIVE_SPARSE_METHOD=sla_cag \
bash scripts/evaluation/run_vbench.sh longlive2_standard_5pct
```

默认稀疏 kernel 是 RainFusion。仅在 BSA smoke 与 microbenchmark 通过后，用下面的
覆盖变量做同 checkpoint 对照；resolved YAML、metadata 和 run tag 会记录后端：

```bash
LONGLIVE_SPARSE_METHOD=sla_cag \
LONGLIVE_SLA_BACKEND=mindiesd_bsa \
bash scripts/evaluation/run_msprof.sh 32s
```

resolver 会向 `model_kwargs` 注入：

```yaml
sparse_config:
  enabled: true
  backend: mindiesd
  sparsity: 0.95
  sparsity_base: 0.97
  block_q: 128
  block_k: 128
  feature_map: softmax
  keep_sink_frames: 1
  keep_recent_frames: 1
  dense_current: false
  min_sparse_history_frames: 1
  query_block_batch: 1
  linear_cache: true
  linear_eps: 1.0e-5
```

正式推理的稀疏 softmax 分支默认使用 MindIE-SD `RainFusionAttention`；不可用时
直接失败。通过显式后端覆盖可验证官方 BSA，但在完整 SLA 实测通过前不会自动切换。
线性补偿分支由 PyTorch NPU 算子计算，其中输出投影已等价折叠进 `K^T V` 统计量。
训练使用支持反向的
`ascend_triton`。第一块或历史不足时回到 dense；之后对全部滚动 KV 做全局
128-token block Top-K，并强制保留 sink/recent blocks。默认不保持当前 chunk
稠密，以免 8/32 帧当前块把理论稀疏率限制在 75%。

SP4 尾部形状是 `Q=7040=55x128`、`KV=28160=220x128`。CAG 在不同 AR chunk
调整 Top-K 预算。相同 chunk 的多个去噪 step 会复用历史 K 的 block summaries
和线性注意力统计量；query、当前 K、Top-K 和最终 LUT 仍逐层逐步更新。
`selectIdx` 保持紧凑且升序，满足 RainFusionAttention 契约。

先用真实 SP4 尾部形状运行 microbenchmark。默认测试正式推理使用的
MindIE-SD 128 block；Ascend Triton 用于训练 kernel 回归：

```bash
python tests/npu/benchmark_sla_inference.py --device npu:0
python tests/npu/benchmark_sla_inference.py --device npu:0 \
  --backend mindiesd_bsa
python tests/npu/benchmark_sla_inference.py --device npu:0 \
  --backend ascend_triton
```

输出分别报告未缓存/缓存路由、稀疏 kernel、显式/折叠投影线性分支以及完整 SLA
延迟。`linear_projection_speedup` 只衡量线性分支的代数优化；判断稳态收益必须比较
`cached_full_ms` 与同次运行的 `dense median_ms`，再用 msprof 验证整网收益。

在 MindIE-SD 3.0.0 + CANN 8.5 上，RainFusion 实测支持矩形 SLA。该环境虽然注册了
`block_sparse_attention` 的 PyTorch wrapper，但 `libopapi.so` 不包含
`aclnnBlockSparseAttentionV2`，因此不能使用 `mindiesd_bsa`。单独 source 同一套 CANN
环境不会补出缺失符号；必须整体升级支持该算子的 CANN/MindIE-SD 栈。

正式启用前，`cached_full_ms` 必须小于同次运行的 `dense median_ms`。只比较理论稀疏率
或 kernel 正确性不能证明加速。

可在启动前只展开配置确认：

```bash
LONGLIVE_SPARSE_METHOD=sla_cag \
python scripts/evaluation/resolve_config.py vbench \
  --config configs/inference/vbench.yaml \
  --preset longlive2_standard_5pct \
  --seed 0 \
  --output /tmp/vbench_sla.yaml

sed -n '1,35p' /tmp/vbench_sla.yaml
```

## 4. VBench 数据集和 preset

```text
data/benchmarks/
├── performance/prompts.txt
├── vbench_standard/{full,5pct,20pct}/
│   ├── prompts.txt
│   └── full_info.json
├── vbench_augmented/{full,5pct,20pct}/prompts.txt
└── vbench_long/evaluator_configs/
```

- Standard Full：944 条唯一提示词。
- Standard 5%：43 条，AISBench K-Means Mini。
- Standard 20%：186 条，另一轮独立 K-Means 抽样。
- Augmented：Qwen2.5 seed-42 生成且与 Standard 逐行对齐；命名和 `full_info.json` 复用对应 Standard 子集。

5% 与 20% 是独立抽样，5% 不是 20% 的子集。

VBench 提供 12 个 preset：

```text
longlive2_standard_{full,5pct,20pct}
longlive2_augmented_{full,5pct,20pct}
wan22_standard_{full,5pct,20pct}
wan22_augmented_{full,5pct,20pct}
```

默认参数是 125 个像素帧、24 FPS、SP2 x DP8、seed 0 到 4，共需 16 张 worker NPU。LongLive2 会换算为 32 个 latent 帧；Wan2.2 原生入口直接生成 125 个像素帧。

## 5. VBench 常用命令

VBench 需要生成环境、AISBench 环境和 VBench 模型缓存：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export GENERATION_ENV=/path/to/longlive-env
export AISBENCH_ENV=/path/to/aisbench-env
export VBENCH_CACHE_DIR=/mnt/share/weights/vbench_models/
```

### 5.1 LongLive2 原始权重 Dense

```bash
LONGLIVE_GENERATOR_CKPT=/path/to/longlive2_merged_generator.pt \
RUN_ID=longlive2_base_dense_5pct \
bash scripts/evaluation/run_vbench.sh longlive2_standard_5pct
```

### 5.2 LoRA 合并权重 Dense

```bash
LONGLIVE_GENERATOR_CKPT=runs/merged/longlive2_sla_cag_2k.pt \
RUN_ID=sla_cag_2k_dense_5pct \
bash scripts/evaluation/run_vbench.sh longlive2_standard_5pct
```

### 5.3 同一合并权重 SLA+CAG

```bash
LONGLIVE_GENERATOR_CKPT=runs/merged/longlive2_sla_cag_2k.pt \
LONGLIVE_SPARSE_METHOD=sla_cag \
RUN_ID=sla_cag_2k_sparse_5pct \
bash scripts/evaluation/run_vbench.sh longlive2_standard_5pct
```

三组关系：原始 Dense 对比训练后 Dense 衡量 LoRA 后训练影响；训练后 Dense 对比训练后 SLA 衡量纯稀疏影响；原始 Dense 对比最终 SLA 衡量总体效果。三组必须使用相同 subset、帧数和 seeds。

### 5.4 Augmented prompt

```bash
LONGLIVE_GENERATOR_CKPT=runs/merged/longlive2_sla_cag_2k.pt \
RUN_ID=sla_cag_2k_dense_augmented_5pct \
bash scripts/evaluation/run_vbench.sh longlive2_augmented_5pct
```

### 5.5 SLA+CAG 20% 串行评测

标准 20% 和 Augmented 20% 可使用同一合并权重串行执行 SLA+CAG 评测。第一项失败时脚本立即停止；`RUN_ID_PREFIX` 可选，默认包含启动时间：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
RUN_ID_PREFIX=sla_cag_step100_20pct \
bash scripts/evaluation/run_vbench_sla_cag_20pct.sh \
  runs/merged/longlive2_sla_cag_step100.pt
```

### 5.6 Wan2.2 原生模型

```bash
LONGLIVE_MODEL_ROOT=/path/to/Wan2.2-TI2V-5B \
RUN_ID=wan22_dense_5pct \
bash scripts/evaluation/run_vbench.sh wan22_standard_5pct
```

不要给 `wan22_*` 命令设置 `LONGLIVE_SPARSE_METHOD=sla_cag`。

### 5.7 扩大评测规模

先用 5% 验证链路，再切换：

```bash
bash scripts/evaluation/run_vbench.sh longlive2_standard_20pct
bash scripts/evaluation/run_vbench.sh longlive2_standard_full
```

默认每个 prompt 运行 5 个 seed，因此 5%/20%/Full 分别生成 215/930/4720 个视频。

## 6. VBench 终端输出、恢复和产物

启动时输出：

```text
[run] task=vbench preset=... run_id=...
[run] devices=... layout=SP2xDP8 prompts=43
[generate] seed=0 (1/5) port=...
[generate seed 0 ...] [####------] ... [...<..., ...s/video]
[evaluate] videos=runs/vbench/.../videos/prepared
[evaluate dimensions] [####------] .../16 [...]
[done] run=runs/vbench/...
```

生成和 AISBench 子进程日志不会与动态进度条混排。失败时脚本会显示日志末尾 100 行；若 AISBench 日志出现 `[RUNNER-TASK-001]`，或最终不足 16 个 dimension，脚本返回失败。

```text
runs/vbench/<run-id>/
├── manifest.json                    # preset、模型类别、数据、seed、SP/DP、稀疏方法
├── resolved_seed_0.yaml             # 每个 seed 的实际推理配置
├── videos/
│   ├── raw/seed_0/*.mp4             # 模型原始输出
│   └── prepared/*.mp4               # VBench 规范命名
└── aisbench/<evaluation-session>/   # 16 维结果和 AISBench 工作目录

logs/vbench/<run-id>/
├── seed_0.log
├── ...
└── aisbench.log
```

`RUN_ID` 只能是目录名，不能包含 `/`。中断后使用相同 `RUN_ID` 会按 seed 目录中的 mp4 数量继续，完整 seed 会显示 `[resume] ... already complete`；评测阶段每次创建新的 session 目录。

## 7. 性能测试与 msprof

### 7.1 无 profiler 端到端基准

先使用无 profiler 入口判断真实延迟。默认运行 1 次预热和 3 次有效测量，保存
manifest、resolved YAML、视频、原始日志和统计摘要：

```bash
LONGLIVE_SPARSE_METHOD=dense \
RUN_ID=dense-32s-3run \
bash scripts/evaluation/run_benchmark.sh 32s

LONGLIVE_SPARSE_METHOD=sla_cag \
LONGLIVE_SLA_BACKEND=mindiesd \
RUN_ID=sla-rainfusion-32s-3run \
bash scripts/evaluation/run_benchmark.sh 32s
```

可用 `BENCHMARK_REPEATS` 和 `BENCHMARK_WARMUP` 调整次数。Dense/SLA 对照必须保持
checkpoint、提示词、seed、SP/DP、VAE 模式和设备集合一致，并比较 summary 中的 p50；
不能使用保存时间，也不能用单次运行决定默认后端。

若完整生成被异步 VAE 临界路径掩盖，使用 latent-only 模式隔离 DiT。该模式不加载或
执行 VAE，只需要 4 张 SP worker 卡；summary 中 FPS/RTF 为 0 是预期行为，只比较
`generation_seconds`：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
BENCHMARK_LATENTS_ONLY=1 LONGLIVE_SPARSE_METHOD=dense \
RUN_ID=dense-dit-32s-3run \
bash scripts/evaluation/run_benchmark.sh 32s

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
BENCHMARK_LATENTS_ONLY=1 LONGLIVE_SPARSE_METHOD=sla_cag \
LONGLIVE_SLA_BACKEND=mindiesd RUN_ID=sla-dit-32s-3run \
bash scripts/evaluation/run_benchmark.sh 32s
```

当前 SP4 32 秒实测中，Dense/SLA latent-only p50 分别为 `48.656 s` 和 `36.240 s`，
SLA 将 DiT 延迟降低 `25.52%`。但包含 dedicated VAE 的 p50 分别为 `121.983 s` 和
`127.555 s`，未获得端到端收益。这两组指标必须分开报告：前者证明 attention/DiT
优化，后者反映当前单视频交付能力；不能用其中一项替代另一项。

latent-only 运行保存的 `.pt` 可直接用于 VAE-only 基准，不必重新执行 DiT：

```bash
python tests/npu/benchmark_vae_decode.py \
  --latent runs/benchmark/dense-dit-32s-3run/videos/rank0-1-0_regular_sp4.pt \
  --device npu:0 --chunk-frames 8 --iterations 1
```

该工具先用两个 chunk 预热，再复现当前 dedicated VAE worker 内的 cached decode、逐
chunk pinned DtoH、CPU 拼接和归一化，分别报告 `decode_dtoh_seconds` 与
`cpu_post_seconds`；NPU Event 进一步拆出 `vae_device_seconds` 与
`dtoh_device_seconds`。工具同时给出 latent/pixel FPS 和峰值显存。它不包含 DiT 卡到
VAE 卡的队列传输；后续 VAE 优化必须复用同一个 latent、chunk 大小和设备环境进行对照。

使用同一个 latent 采集 VAE-only 算子 profile。默认只采集 16 个 latent 帧，覆盖首块和
稳态块，避免对完整 32 秒视频重复插桩：

```bash
ASCEND_RT_VISIBLE_DEVICES=15 RUN_ID=vae-16f-baseline \
bash scripts/evaluation/run_vae_msprof.sh \
  runs/benchmark/dense-dit-32s-3run/videos/rank0-1-0_regular_sp4.pt
```

重点检查 `profiling/analysis/compute_op_sum` 中 Conv3D、上采样、归一化和 TransData/格式
转换的累计耗时。该结果用于决定是否需要算子替换或多卡 VAE，并不能替代无 profiler 延迟。

### 7.2 msprof 算子分析

msprof 当前只测试 LongLive2。三个 preset：

| preset | latent 帧 | 解码像素帧 | 24 FPS 时长 |
| --- | ---: | ---: | ---: |
| `16s` | 96 | 381 | 15.875 秒 |
| `32s` | 192 | 765 | 31.875 秒 |
| `64s` | 384 | 1533 | 63.875 秒 |

关系为 `pixel_frames = (latent_frames - 1) x 4 + 1`；preset 名是便于识别的近似时长。

默认 SP4 x DP1，并使用独立异步 VAE：4 张生成 worker 加 1 张 VAE 卡，共必须恰好暴露 5 张 NPU。固定提示词文件有 2 条，每 rank 第一条是 warmup，不计入 summary。

Dense：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4 \
GENERATION_ENV=/path/to/longlive-env \
LONGLIVE_GENERATOR_CKPT=runs/merged/longlive2_sla_cag_2k.pt \
RUN_ID=sla_cag_2k_dense_msprof_32s \
bash scripts/evaluation/run_msprof.sh 32s
```

SLA+CAG：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4 \
GENERATION_ENV=/path/to/longlive-env \
LONGLIVE_GENERATOR_CKPT=runs/merged/longlive2_sla_cag_2k.pt \
LONGLIVE_SPARSE_METHOD=sla_cag \
RUN_ID=sla_cag_2k_sparse_msprof_32s \
bash scripts/evaluation/run_msprof.sh 32s
```

脚本始终运行完整 msprof：生成 resolved YAML、用 msprof 包裹 torchrun、恢复/解析 profile、汇总 warmup 后延迟和 FPS，并执行算子、HCCL、通信矩阵、慢卡、free analysis 和 advisor。

默认开启 AICore PMU 并采集 `PipeUtilization`，用于分析 AICore 流水线利用率。多进程入口会先绑定各自的 `local_rank`，再解析默认 NPU，避免所有 rank 在 profiler 启动期间共同初始化逻辑 `npu:0`。

```bash
MSPROF_AI_CORE=true bash scripts/evaluation/run_msprof.sh 32s
```

部分芯片、驱动、固件与 CANN 组合仍可能在 PMU 初始化阶段报 `DrvFftsProfileStart failed` 或 `561103`。此时可使用新的 `RUN_ID` 并设置 `MSPROF_AI_CORE=false` 重跑一次，以区分模型执行问题和 PMU 环境问题。关闭 PMU 的结果只有 task timeline、AscendCL、Runtime、AICPU、HCCL 和系统内存数据，可用于定位耗时与通信，但不能用于 AICore 流水线利用率结论；正式性能报告仍必须在版本匹配的环境中启用 PMU。失败运行产生的 `PROF_*` 仅包含不完整初始化数据，也不能用于性能结论。

终端关键输出：

```text
[run] task=msprof preset=32s run_id=...
[run] devices=... layout=SP4xDP1
[run] config=.../resolved.yaml profile=.../profiling/raw
[analyze] compute_op_sum
[analyze] hccl_sum
...
[done] run=runs/msprof/...
[note] latency and FPS in this run include msprof collection overhead
```

产物：

```text
runs/msprof/<run-id>/
├── manifest.json
├── resolved.yaml
├── summary.txt
├── videos/
└── profiling/
    ├── raw/PROF_*/
    └── analysis/{all,compute_op_sum,hccl_sum,...}/

logs/msprof/<run-id>/
├── msprof.log
└── recovery.log                 # 仅触发恢复解析时存在
```

`summary.txt` 是从生成日志按 `warmup_per_rank` 汇总的延迟/FPS；msprof 插桩有开销，只能在相同 profiler 参数、硬件、帧数和布局下横向比较，不能当作无侵入生产吞吐。

## 8. 结果验收和常见问题

- 查看 `manifest.json` 和 resolved YAML，确认 checkpoint、模型类别、seed、SP/DP 和 `sparsity_method`。
- 稀疏运行的 resolved YAML 必须含 `model_kwargs.sparse_config.enabled: true`、
  `backend: mindiesd` 与 `block_q/block_k: 128`。
- Dense/SLA 使用不同 `RUN_ID`，否则已有视频可能被错误复用。
- 比较质量时保持同一 checkpoint、subset 和 seeds；比较性能时还要保持 profiler 参数和布局。
- VBench 生成失败先看 `logs/vbench/<run-id>/seed_<seed>.log`，评测失败看 `aisbench.log`。
- msprof 无 `PROF_*` 目录时视为失败；`msprof-analyze` 单项失败会输出 warning，但主 profile 和 summary 仍保留。
- `runs/` 保存配置、视频、checkpoint 和结构化结果；`logs/` 只保存文本日志和 JSONL。
