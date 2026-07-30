# 昇腾 NPU BF16 启动指南

本文档说明如何在昇腾 NPU 上用 BF16 路径启动 LongLive 5B 推理。昇腾上不要使用 `configs/nvfp4/` 下的 NVFP4 配置。

## 1. 权重路径

当前按你服务器上的路径配置：

```text
/mnt/share/weight/Wan2.2-TI2V-5B/
/mnt/share/weight/LongLive/checkpoints/longlive_5b/
```

推理配置文件：

```text
configs/inference_npu_bf16.yaml
```

其中 Wan2.2 基座模型路径已经写好：

```yaml
model_kwargs:
  model_name: Wan2.2-TI2V-5B
  model_root: /mnt/share/weight/Wan2.2-TI2V-5B
```

LongLive generator checkpoint 当前暂按下面路径填写：

```yaml
checkpoints:
  generator_ckpt: /mnt/share/weight/LongLive/checkpoints/longlive_5b/generator_checkpoint.pt
  lora_ckpt: null
```

请先在服务器上确认真实 checkpoint 文件名：

```bash
find /mnt/share/weight/LongLive/checkpoints/longlive_5b -maxdepth 2 -type f
```

如果真实文件名不是 `generator_checkpoint.pt`，请修改 `configs/inference_npu_bf16.yaml` 里的 `checkpoints.generator_ckpt`。

如果 LongLive 权重目录里有单独的 LoRA 文件，则设置：

```yaml
checkpoints:
  generator_ckpt: /mnt/share/weight/LongLive/checkpoints/longlive_5b/<base-generator>.pt
  lora_ckpt: /mnt/share/weight/LongLive/checkpoints/longlive_5b/<lora>.pt
```

这种情况下保留配置里的 `adapter` 段。如果你的 `generator_ckpt` 已经是合并后的完整 generator 权重，则保持 `lora_ckpt: null`，并删除或注释 `adapter` 段。

## 2. 环境准备

进入仓库目录后，先加载昇腾环境：

```bash
cd /path/to/LongLive-oneday
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LLV2_DEVICE=npu
export PYTHONPATH=$PWD:$PYTHONPATH
```

建议先检查 NPU 和 torch_npu：

```bash
npu-smi info
python -c "import torch, torch_npu; print(torch.npu.is_available())"
```

如果服务器环境还没有安装项目依赖，优先使用 NPU 精简依赖文件，不要直接装原始 `requirements.txt`：

```bash
pip install -r requirements_npu_bf16.txt
```

注意：`torch`、`torchvision`、`torch_npu` 必须使用和 CANN 匹配的昇腾版本，不要通过普通 PyPI 或 CUDA wheel 覆盖服务器已有环境。

如果是首次调试，可以加长 HCCL 超时时间：

```bash
export HCCL_CONNECT_TIMEOUT=1800
```

## 3. 单卡推理

建议先用单卡跑通模型加载和一条 prompt：

```bash
LLV2_DEVICE=npu python inference.py \
  --config_path configs/inference_npu_bf16.yaml
```

默认输出目录：

```text
videos/longlive2_npu_bf16/
```

当前单卡配置默认让 T5 和 VAE 都在 NPU 上运行，不做 CPU offload。CPU offload 会省显存，但会明显变慢，只有在 SP 仍然爆显存时再考虑。

## 4. 多卡推理：优先使用 SP/Ulysses

注意：普通 `inference.py` 的多卡启动主要是数据并行，每张卡仍会加载完整模型，并不会自动把同一条视频的模型显存切到多张卡上。因此如果单卡爆显存，直接用多卡 `inference.py` 通常仍会爆。

要降低单条视频的单卡显存，应该使用 Ulysses 序列并行入口。

16 卡推荐命令如下。这里显式设置 `master_addr` 和 `master_port`，避免 `torchrun --standalone` 自动选择主机名后出现 TCPStore 连接超时。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

LLV2_DEVICE=npu torchrun \
  --nnodes=1 \
  --nproc_per_node=16 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  inference_sp.py \
  --config_path configs/inference_sp_npu_bf16.yaml
```

如果端口被占用，把 `29501` 换成其他空闲端口，例如 `29511` 或 `29601`。

`configs/inference_sp_npu_bf16.yaml` 中：

```yaml
sp_size: 8
dp_size: 2
model_num_heads: 24
model_kwargs:
  num_frame_per_block: 8
```

注意：当前 Wan2.2-TI2V-5B 配置的 `model_num_heads: 24`，`inference_sp.py` 要求 `sp_size` 能整除 `gcd(model_num_heads, num_frame_per_block)`。因此 16 卡不能直接设置成 `sp_size: 16`；当前 16 卡配置使用 `sp_size: 8, dp_size: 2`，也就是两组 8 卡 SP 组并行跑样本。

16 卡的 `dp_size: 2` 会把 prompt 按两个 DP 组分配，建议 `data.data_path` 至少提供 2 条 prompt。如果只有 1 条 prompt，优先使用下面的 8 卡配置，或者在 prompt 文件中补充第二条任务。

如果只想使用 8 卡，改成：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p logs

LLV2_DEVICE=npu torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  inference_sp.py \
  --config_path configs/inference_sp_npu_bf16.yaml
```

并把配置改回：

```yaml
sp_size: 8
dp_size: 1
```

`num_frame_per_block` 不能随便设成 1。SP 分组要求 `sp_size` 能整除 `gcd(model_num_heads, num_frame_per_block)`；当前 Wan2.2-TI2V-5B 配置的 `model_num_heads: 24`，所以 8 卡 SP 下使用 `num_frame_per_block: 8`。

## 5. 服务器同步最新 npu 分支

如果服务器上仓库有临时修改，但现在希望全部丢弃，直接使用 GitHub 上 `npu` 分支的最新版本，可以执行：

```bash
cd /mnt/share/r50063443/LongLive-oneday

export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897

git fetch origin
git switch npu
git reset --hard origin/npu
git clean -fd
```

注意：`git reset --hard` 和 `git clean -fd` 会丢弃服务器仓库里的未提交修改和未跟踪文件。确认没有需要保留的本地改动后再执行。

## 6. Prompt 输入

当前配置使用仓库里的示例 prompt：

```yaml
data:
  data_path: example/long_example.txt
```

如果要换成自己的 prompt，新建一个 txt 文件，每行一个 prompt，然后修改 `data.data_path`。

例如：

```text
data/my_prompts.txt
```

配置：

```yaml
data:
  data_path: data/my_prompts.txt
```

## 7. VBench / AISBench 评测

### 7.1 已下载的数据

仓库内已经准备好评测 prompt 和元数据，不包含生成视频或 VBench 评测模型权重：

```text
data/benchmarks/
├── vbench_mini/       # AISBench VBench-1.0-mini 5% 子集，43 条去重 prompt
├── vbench_standard/   # VBench 1.0 Standard，946 条记录/944 条去重 prompt + 16 维元数据
├── vbench_standard_augmented_wan21_qwen25_seed42/
│                       # Wan2.1 + Qwen2.5-3B + seed 42 的 944 条公开增强 prompt
├── vbench_standard_20pct/
│                       # AISBench K-Means 20% 固定子集，187 条记录/186 条去重 prompt
├── vbench_standard_20pct_augmented_wan21_qwen25_seed42/
│                       # 同一 186 条子集对应的公开增强 prompt
├── vbench_long/       # VBench-Long 元数据和 slow-fast 切分配置
└── performance/       # 固定的昇腾性能测试 prompt
```

来源说明见 `data/benchmarks/README.md`。AISBench mini 使用
`VBench_kmeans_info_0.05.json`，在仓库中重命名为 `VBench_full_info.json`，便于直接填入
AISBench 的 `full_json_dir`。

LongLive-2.0 论文的 60 秒结果使用 MovieGenBench prompts，但论文和当前公开仓库没有提供
可直接下载的对应 prompt 文件。因此本仓库提供的是可复现的 VBench-Long Standard prompt；
使用它得到的结果不能直接宣称复现论文 Table 5。

### 7.2 latent 帧与视频秒数

Wan2.2 VAE 的时间压缩率是 4，正确的整段解码关系为：

```text
像素帧数 = (latent 帧数 - 1) × 4 + 1
视频秒数 = 像素帧数 / 24
```

已提供配置：

| 配置 | latent 帧 | 24 FPS 时长 | 用途 |
| --- | ---: | ---: | --- |
| `vbench_mini_5s_npu_bf16.yaml` | 32 | 5.21 秒 | mini 冒烟 |
| `vbench_standard_20pct_5s_npu_bf16.yaml` | 32 | 5.21 秒 | VBench Standard 20% 回归集 |
| `vbench_standard_20pct_augmented_5s_npu_bf16.yaml` | 32 | 5.21 秒 | 20% 公开增强 prompt 对照组 |
| `vbench_long_60s_npu_bf16.yaml` | 360 | 59.88 秒 | VBench-Long |
| `perf_16s_npu_bf16.yaml` | 96 | 15.88 秒 | 性能档位 |
| `perf_32s_npu_bf16.yaml` | 192 | 31.88 秒 | 性能档位 |
| `perf_64s_npu_bf16.yaml` | 384 | 63.88 秒 | 性能档位 |

这些配置均设置 `streaming_vae: false`，保证 VAE 按完整时间序列解码。当前 Wan VAE 不支持
真正的 `cached_decode`；如果把普通 `decode` 按 8 latent 帧独立调用，会得到错误的帧数并在
block 边界重置时间缓存。例如 128 latent 帧会输出约 19.33 秒，而不是正确整段解码的
21.21 秒。该分块 fallback 不能用于正式质量评测。

### 7.3 先运行 AISBench mini 单样本链路（默认 16 卡）

质量评测配置默认使用 16 卡，即两个独立的 8 卡 SP 组并行处理 prompt：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

LLV2_DEVICE=npu torchrun \
  --nnodes=1 \
  --nproc_per_node=16 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  inference_sp.py \
  --config_path configs/benchmarks/vbench_mini_5s_npu_bf16.yaml
```

三个 VBench 质量配置均为：

```yaml
sp_size: 8
dp_size: 2
```

因此 16 卡用于提高 prompt 吞吐量，不会让单条视频使用 16 卡 SP。`inference_sp.py` 使用不
补齐、不丢弃的分组采样，43 条 mini prompt 会由两个 DP 组分别处理 22 条和 21 条。

当前推理入口按数据集序号保存文件。生成结束后，运行仓库内的整理脚本，把序号映射回
prompt，并按 VBench 的 `{prompt}-{sample_index}.mp4` 规则创建评测目录：

```bash
python third_party/aisbench_adapter/prepare_vbench_videos.py \
  --benchmark mini \
  --sample-index 0
```

脚本默认严格检查 43 条 prompt 是否全部生成。只测试部分 prompt 时可以增加
`--allow-missing`；目标文件已存在且确认需要覆盖时增加 `--overwrite`。

### 7.4 AISBench 代码和配置

本仓库没有复制整个 AISBench，而是提供轻量适配层：

```text
third_party/aisbench_adapter/
├── eval_longlive_vbench.py       # 可直接传给 ais_bench 的独立配置
├── prepare_vbench_videos.py      # LongLive 输出文件整理脚本
└── run_vbench_16npu.sh           # 16 NPU / 16 workers 启动脚本
```

服务器继续使用已经部署的 AISBench。适配脚本不会下载模型权重；默认从
`/mnt/weight/vbench_models/` 读取 AISBench 已有缓存。执行：

```bash
export VBENCH_CACHE_DIR=/mnt/weight/vbench_models/
bash third_party/aisbench_adapter/run_vbench_16npu.sh
```

启动脚本会切换到 LongLive 仓库根目录，因此 AISBench 默认结果统一写入
`LongLive-oneday/outputs/default/<timestamp>/`，不受执行命令时所在目录影响。

适配依据是 AISBench 官方仓库 `https://github.com/AISBench/benchmark` 的 VBench 1.0
配置。服务器上的版本应至少包含下面这些类，可以先检查，不会触发权重下载：

```bash
python - <<'PY'
from ais_bench.benchmark.datasets import VBenchDataset
from ais_bench.benchmark.summarizers import VBenchSummarizer
from ais_bench.benchmark.tasks import VBenchEvalTask
print("AISBench VBench adapter is available")
PY
```

脚本默认设置：

```python
ASCEND_RT_VISIBLE_DEVICES = "0,1,...,15"
AISBENCH_MAX_WORKERS = 12
```

如果希望手工修改 AISBench 官方配置，只需要修改
`ais_bench/configs/vbench_examples/eval_vbench_standard.py`：

```python
DATA_PATH = "/mnt/share/r50063443/LongLive-oneday/videos/benchmarks/vbench_mini_5s_vbench"
VBENCH_CACHE_DIR = "/mnt/weight/vbench_models/"

vbench_eval_cfg = dict(
    load_ckpt_from_local=True,
    full_json_dir="/mnt/share/r50063443/LongLive-oneday/data/benchmarks/vbench_mini/VBench_full_info.json",
    device="npu",
)
```

以下 AISBench 文件不需要修改：

- `ais_bench/third_party/vbench/`：VBench 1.0 指标实现和默认 prompt。
- `ais_bench/benchmark/summarizers/vbench.py`：Quality、Semantic、Total 汇总。
- `ais_bench/benchmark/tasks/` 和 `datasets/`：本仓库配置直接复用这些实现。

首次配置 AISBench 环境时，四个依赖 GRiT 的维度需要在 AISBench 根目录安装仓库内的
detectron2。这个命令只安装代码，不下载 VBench 权重：

```bash
pip install -e ais_bench/third_party/detectron2 --no-build-isolation
```

VBench 还依赖 decord。x86_64 通常可以直接安装；ARM 服务器没有对应 wheel 时需要按
AISBench 文档从源码编译：

```bash
pip install decord
```

如果导入 decord 时出现以下错误：

```text
libstdc++.so.6: version `GLIBCXX_3.4.32' not found
```

说明 decord wheel 需要 GCC 13 的 C++ 运行库，但进程加载了较旧的
`/usr/lib64/libstdc++.so.6`。这与 NPU、VBench 数据或 16 卡并行无关。先检查 Conda 环境：

```bash
echo "$CONDA_PREFIX"
strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep GLIBCXX_3.4.32
```

如果能找到该符号，只需要让 Conda 动态库优先。仓库启动脚本已经自动执行这一设置；手工
启动时使用：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import decord; print(decord.__version__)"
```

注意 `LD_LIBRARY_PATH` 的 Conda 路径与原路径之间必须有冒号。下面这种写法是错误的，会把
两个目录直接拼接，并可能进一步导致 `libhccl.so` 找不到：

```bash
# 错误：展开原变量前缺少冒号
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH}"
```

如果已经执行过错误命令，重新加载 CANN 环境并恢复路径：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -c "import torch; import torch_npu; print(torch_npu.npu.device_count())"
python -c "import decord; print(decord.__version__)"
```

仓库的 `run_vbench_16npu.sh` 会自动尝试加载上述 CANN 环境脚本，并在启动 AISBench 前分别
检查 `torch_npu` 和 `decord`。

如果 Conda 的 `libstdc++.so.6` 也没有该符号，使用清华 conda-forge 镜像安装 GCC 13 运行
库，不要替换系统 `/usr/lib64/libstdc++.so.6`：

```bash
conda install -n aisbench_npu \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  "libstdcxx-ng>=13,<14" "libgcc-ng>=13,<14"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import decord; print(decord.__version__)"
```

安装运行库后仍无法导入时，卸载不兼容的 wheel，按照 AISBench 文档在服务器上使用
`-DUSE_CUDA=0` 从源码编译 decord，使其链接当前环境的 C++ 运行库。

上述 mini 配置只有 1 个 seed，用于验证“LongLive 生成 -> 文件整理 -> AISBench 评分”完整
链路。AISBench Standard 的正式协议要求视频文件名为 `{prompt}-{index}.mp4`，每条普通
prompt 需要 5 个不同 seed，`temporal_flickering` 需要 25 个。正式评测时应修改
`logging.seed` 后分 5 次顺序生成，temporal flickering 子集分 25 次生成，并在每轮结束后把
对应轮次传给 `--sample-index`，整理成 `{prompt}-0.mp4` 至 `{prompt}-4.mp4` 或
`{prompt}-24.mp4`。不要直接把 `num_samples` 改成 5 或 25，否则会把样本放进同一 batch，
显著增加显存。缺少这些重复样本时，所得结果只能标记为 single-sample diagnostic，不能
报告为完整 VBench Standard 分数。

当前 VBench-mini 单 seed 结果与论文只能做方向性观察：

| 设置 | Total | Quality | Semantic | 说明 |
| --- | ---: | ---: | ---: | --- |
| 当前昇腾 mini | 81.50 | 83.10 | 75.10 | 43 条 prompt，每条 1 个 seed，约 1280×704 |
| LongLive-2.0 BF16 4 步 | 85.06 | 86.67 | 78.63 | 论文 Table 4，完整 VBench，prompt augmentation，1280×720 |
| 原始 LongLive 1.3B | 84.87 | 86.97 | 76.47 | 论文 Table 1，完整 VBench，832×480 |

当前相对 LongLive-2.0 BF16 行分别低 3.56、3.57、3.53 分，但不能把该差值解释成昇腾精度
损失，原因包括数据子集、seed 数、prompt augmentation 和输出高度均不一致。正式对比必须用
`data/benchmarks/vbench_standard/` 的完整944条 prompt，并为每条 prompt 准备5个 seed。当前
默认流水线改为20%子集，所得分数只能标记为 VBench-20%-KMeans，不能作为论文 Full VBench
分数。

### 7.5 VBench Standard 20% 五 seed 流水线

正式质量配置使用空间 latent 高宽 `44 × 80`，经过 Wan VAE 空间 16 倍上采样后为
`704 × 1280`。Wan DiT 的空间 patch size 是 `(2, 2)`，latent 高宽必须都是偶数；直接将高度
改成 45 会在 patch embedding 中被截断回 44，并导致 diffusion state 与模型输出尺寸不一致。
精确 `720 × 1280` 需要实现 latent padding 和解码后裁剪，当前流水线不宣称已对齐论文的
720 高度。不要把 `num_samples` 改成 5；流水线会依次使用 seed 0、1、2、3、4 启动五次
生成，保持单次 batch 显存不变，并整理为 `{原始 prompt}-0.mp4` 到
`{原始 prompt}-4.mp4`。

20% Standard 原始 prompt 一键执行：

```bash
cd /mnt/share/r50063443/LongLive-oneday
bash scripts/run_npu_vbench_standard_pipeline.sh
```

`scripts/run_npu_vbench_quality_pipeline.sh`是标准版和增强版入口共用的内部实现，不应在其中
修改卡数或直接启动。

流水线自动完成：

1. 加载 `/usr/local/Ascend/ascend-toolkit/set_env.sh`。
2. 每轮生成自动使用 `/mnt/share/r50063443/conda_envs/longlive` 环境，不受外层当前 Conda 环境影响。
3. 默认使用12张 NPU，按 `sp_size=2, dp_size=6` 生成 K-Means 选出的186条 prompt；每个DP组每轮处理31条。
4. 依次生成五个 seed，不把五个样本放入同一 batch。
5. 终端显示每个 seed 和五轮总进度，并给出累计耗时、动态 ETA 与平均秒/视频；详细生成日志写入 `logs/npu_quality/`。
6. 用原始 prompt 文件名整理930个视频。
7. 切换到 `${AISBENCH_ENV}` 并调用 AISBench 的 16 维 VBench 质量评测。

默认服务器路径如下，可用同名环境变量覆盖：

```bash
export GENERATION_ENV=/mnt/share/r50063443/conda_envs/longlive
export AISBENCH_ENV=/mnt/share/r50063443/conda_envs/aisbench_npu
export VBENCH_CACHE_DIR=/mnt/weight/vbench_models/
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11
```

每次运行使用带时间戳的独立目录：

```text
logs/npu_quality/<run_id>/                 # 五轮 torchrun 日志和 AISBench 日志
videos/benchmarks/quality_runs/<run_id>/   # 原始视频和 VBench 命名视频
outputs/default/<AISBench_timestamp>/      # AISBench 指标汇总
```

默认 `186 × 5` 会生成930个视频，约为 Full 4720个视频的19.7%。它覆盖全部16个指标，适合
昇腾适配和模型版本间的固定回归比较，但不是 Full VBench。先确认 mini 链路、权重路径及单个
`704 × 1280` 视频显存正常，再启动20%流水线。

每个 seed 启动前，流水线会检查 `MASTER_PORT + sample_index` 是否可用；端口被占用时自动
选择当前节点的空闲端口。中断后可通过原 run id 续跑，已生成完整的 seed 会被跳过：

```bash
RUN_ID=20260729_092832_vbench_standard_20pct_5seed_sp4_dp2 \
bash scripts/run_npu_vbench_standard_pipeline.sh
```

如果某个 seed 只有部分 MP4，该 seed 会从头重新生成；已经完整生成的 seed 不会重复计算。

要依次完成“增强 prompt 生成+AISBench”和“原始 prompt 生成+AISBench”，直接运行：

```bash
bash scripts/run_npu_vbench_all.sh
```

该脚本顶部集中定义了卡号、进程数、SP/DP大小、AISBench worker、端口和两个 Conda 环境路径。
运行时会用`SP_SIZE`和`DP_SIZE`改写临时YAML，仓库中的正式配置文件不会改变。默认配置为：

```text
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11
NPROC_PER_NODE=12
SP_SIZE=2
DP_SIZE=6
AISBENCH_MAX_WORKERS=12
```

改为8卡时无需修改YAML，只需在命令前覆盖变量：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
SP_SIZE=2 \
DP_SIZE=4 \
AISBENCH_MAX_WORKERS=8 \
bash scripts/run_npu_vbench_all.sh
```

`DP_SIZE`默认按`NPROC_PER_NODE / SP_SIZE`计算，`AISBENCH_MAX_WORKERS`默认等于
`NPROC_PER_NODE`，所以上述命令也可只显式设置可见卡、进程数和SP：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
SP_SIZE=2 \
bash scripts/run_npu_vbench_all.sh
```

旧名称`scripts/run_npu_vbench_12npu_all.sh`保留为兼容转发入口，但新任务应使用不绑定卡数的
`scripts/run_npu_vbench_all.sh`。

prompt数量不要求能被`DP_SIZE`整除。入口按DP rank做步进分片；例如186条数据在8卡
`SP2 × DP4`下会自动分成`47/47/46/46`，五个seed仍会完整生成930个视频。生成完成后的
AISBench继承同一组可见NPU，但不使用生成阶段的SP/DP分组。

`SP2` 满足当前模型的 head/block 约束，186条 prompt 又能被6个DP组整除，每组恰好31条。
与已验证的SP4相比，SP2会增加单卡模型中间状态和KV cache占用，因此全量启动前应先观察
第一条视频的峰值HBM。AISBench阶段不使用生成阶段的SP/DP，但会继承同一组12张可见NPU，
并默认并行启动12个评测worker。AISBench的`VBenchEvalTask`声明每个任务使用1张卡，`LocalRunner`
会从`ASCEND_RT_VISIBLE_DEVICES`维护的资源池中为任务分配独立NPU，因此12个worker会使用12张卡。
生成阶段导出的`MASTER_PORT`不能传入评测阶段：每个VBench worker都会创建独立的单进程HCCL
进程组，共用端口会触发`EADDRINUSE`。适配脚本会在启动AISBench前清除torchrun的分布式环境变量，
由各worker自动选择空闲端口。如果主机内存或共享存储承受不了12路模型加载和视频读取，可用
`AISBENCH_MAX_WORKERS=4`、`6`或`8`降低并发，不需要修改其他配置。

SP组内各rank在去噪结束后持有相同的完整latent，但每组只需要输出一份视频。当前入口因此只在
每个SP组的leader（`sp_rank=0`）上将VAE放入NPU并执行整段解码；其他rank直接返回latent并等待
同步。`SP2 × DP6` 最多同时执行6路VAE解码，而不是12路。这不会改变生成结果或视频数量，但可
避免非leader重复解码造成的HBM和主机内存峰值。如果进程仍无Python异常地以`exitcode=-9`退出，
应检查系统或容器OOM记录，并优先退回`SP4 × DP3`降低模型副本和并行解码数量。

### 7.6 公开增强 prompt 对照组

公开 VBench 仓库确实提供了一份 Wan2.1 的增强产物和生成说明：使用 Wan2.1
`QwenPromptExpander`、`Qwen/Qwen2.5-3B-Instruct`、seed 42。该文件的 946 行可无冲突映射到
944 条去重 Standard prompt；默认增强流水线从中选取与20% K-Means子集完全对应的186条，
无需在生成服务器上再次下载 Qwen 权重。

一键生成并评测增强版：

```bash
cd /mnt/share/r50063443/LongLive-oneday
bash scripts/run_npu_vbench_augmented_pipeline.sh
```

增强描述只用于 LongLive 的文本条件。整理视频和 AISBench 语义评测仍使用对应的原始 VBench
prompt，这是因为 `VBench_full_info.json` 的维度标签和语义目标属于原始 prompt。否则把长增强
描述直接作为文件名和评测标签，会改变基准定义。

这套数据可以回答“公开 Wan2.1/Qwen2.5 prompt augmentation 对 LongLive 指标有什么影响”，
但不能声称精确复现 LongLive-2.0 论文的 prompt augmentation。论文没有公开其最终增强文本、
模型版本和完整模板；因此与论文 Table 4 对比时，应明确标注为 public augmentation proxy。

流水线参数可以这样覆盖：

```bash
AISBENCH_ENV=/mnt/share/r50063443/conda_envs/aisbench_npu \
VBENCH_CACHE_DIR=/mnt/weight/vbench_models/ \
MASTER_PORT=29630 \
bash scripts/run_npu_vbench_augmented_pipeline.sh
```

### 7.7 VBench-Long（暂不纳入一键流水线）

VBench-Long 后续单独建立脚本。当前原生仓库位于
`/mnt/share/r50063443/Vench`，其 `vbench2_beta_long/eval_long.py` 默认将设备硬编码为 CUDA，
不能直接在仅有昇腾 NPU 的环境中复用 Standard/AISBench 启动方式。本次 Standard/增强版
流水线不会调用或修改该仓库。

60 秒整段 VAE 解码显存较高，且原生 VBench-Long 仍需完成 NPU 设备适配。后续脚本应先用
单条 prompt 验证生成和六维评测，再扩展到正式五 seed；不要复用本节 Standard 的 AISBench
入口，也不要切回会破坏时间连续性的逐 block 普通 VAE 解码。

### 7.8 昇腾性能测试

依次运行 `perf_16s_npu_bf16.yaml`、`perf_32s_npu_bf16.yaml` 和
`perf_64s_npu_bf16.yaml`。8 卡配置 `sp_size=8, dp_size=1` 用于单条视频延迟；16 卡吞吐测试
把配置改为 `dp_size=2`，并用 `--nproc_per_node=16` 启动。16 卡是两个 8 卡 SP 组，不会
降低单条视频延迟。

`inference_sp.py` 会为每条视频输出：

```text
[benchmark] rank=0 prompt_index=2 generation_seconds=... save_seconds=...
pixel_frames=... video_seconds=... generation_fps=... rtf=... peak_memory_gb=...
```

- `generation_seconds`：`pipeline.inference` 时间，包含文本编码、4 步生成和 VAE 解码，排除 MP4 写盘。
- `save_seconds`：CPU 搬运和 MP4 编码写盘时间，应与论文生成延迟分开。
- `generation_fps = pixel_frames / generation_seconds`。
- `rtf = generation_seconds / video_seconds`；小于 1 才是快于实时。
- `peak_memory_gb`：当前进程在该样本期间的峰值 HBM；报告所有 rank 中最大值。

推荐直接运行单次生成性能流水线：

```bash
bash scripts/run_npu_generation_benchmark.sh
```

脚本默认使用 `configs/benchmarks/perf_16s_npu_bf16.yaml` 和 16 卡 `SP8×DP2`。它只执行所
指定配置的一次生成测试，不会自动循环 16/32/64 秒。测试其他时长可以修改脚本顶部的
`CONFIG_PATH`，也可以在命令行临时指定：

```bash
CONFIG_PATH=configs/benchmarks/perf_32s_npu_bf16.yaml \
  bash scripts/run_npu_generation_benchmark.sh
```

流水线会把 torchrun 完整输出保存到日志，控制台只显示全局 MP4 完成进度条、失败日志尾部
和最终汇总。生成视频、原始日志与汇总分别保存在带时间戳的独立目录，避免覆盖已有结果。

需要手工测试时，8 卡单视频延迟示例如下：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LLV2_DEVICE=npu torchrun \
  --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29511 \
  inference_sp.py \
  --config_path configs/benchmarks/perf_16s_npu_bf16.yaml \
  2>&1 | tee logs/perf_16s_sp8_bf16.log

rg '^\[benchmark\]' logs/perf_16s_sp8_bf16.log

python scripts/summarize_npu_benchmark.py \
  logs/perf_16s_sp8_bf16.log \
  --warmup-per-rank 1
```

16 卡吞吐测试先创建临时 DP2 配置，不修改仓库基线：

```bash
cp configs/benchmarks/perf_16s_npu_bf16.yaml /tmp/perf_16s_npu_bf16_dp2.yaml
sed -i 's/^dp_size: 1$/dp_size: 2/' /tmp/perf_16s_npu_bf16_dp2.yaml
mkdir -p logs

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

LLV2_DEVICE=npu torchrun \
  --nnodes=1 --nproc_per_node=16 \
  --master_addr=127.0.0.1 --master_port=29512 \
  inference_sp.py \
  --config_path /tmp/perf_16s_npu_bf16_dp2.yaml \
  2>&1 | tee logs/perf_16s_sp8_dp2_bf16.log

python scripts/summarize_npu_benchmark.py \
  logs/perf_16s_sp8_dp2_bf16.log \
  --warmup-per-rank 1
```

性能 prompt 文件有 10 条。每个 SP 组的第 1 条用于算子和缓存预热，统计其余样本的均值、
P50 和 P95。DP2 吞吐应按两个组完成有效样本的共同墙钟时间计算，并报告视频/小时；不能把
两个组各自的 FPS 直接相加当作单视频 FPS。

论文可作为两个不同硬件参照：

| 论文设置 | 16 秒 E2E | 32 秒 E2E | 64 秒 E2E | 备注 |
| --- | ---: | ---: | ---: | --- |
| GB200 BF16 4 步 | 26.6 s | 53.2 s | 112.9 s | Table 3，36.4 GB，FPS 24.8 |
| H100 SP=2 BF16 | 19.3 s | 38.1 s | 62.5 s | Table 6 |
| H100 SP=4 BF16 | 26.2 s | 38.6 s | 65.4 s | Table 6，通信开销更高 |

当前昇腾设置是 910B、SP=8、BF16、无 KV 量化、无异步 VAE，不能直接要求达到 GB200 的
24.8 FPS。应同时报告 HCCL 通信占比；如果 SP=8 比较慢，需要分别测 SP=4/SP=8，判断收益
是否被 All-to-All 通信抵消。

最终建议形成三张表：

1. **质量**：VBench 完整 16 维、5 seed；60 秒再报告 VBench-Long 六维。
2. **延迟与吞吐**：16/32/64 秒的平均、P50、P95、FPS、RTF、视频/小时。
3. **资源与稳定性**：每 rank 峰值 HBM、NPU 利用率、HCCL 时间、失败率、实际输出帧数。

AISBench 自身的评测运行时间只代表评测器性能，不是 LongLive 生成性能，必须单独记录。

## 8. 是否需要训练

如果目标只是“在昇腾上跑起来并复现推理效果”，不需要训练，直接使用已经下载好的 LongLive 5B 权重即可。

训练复现是下一阶段工作。如果之后要跑 AR 训练，可以用：

```bash
LLV2_DEVICE=npu torchrun --standalone --nproc_per_node=8 train.py \
  --config_path configs/train_ar_npu_bf16.yaml \
  --logdir logs/train_ar_npu_bf16 \
  --wandb-save-dir wandb \
  --disable-wandb
```

训练前需要先修改 `configs/train_ar_npu_bf16.yaml` 的数据路径：

```yaml
data:
  data_path: /path/to/train_dataset
  eval_data_path: /path/to/eval_prompts
```

## 9. 注意事项

- 当前使用的是 `Wan2.2-TI2V-5B`，不是 Wan2.1。
- 昇腾上只走 BF16：`model_quant: false`、`kv_quant: false`、`torch_compile: false`。
- 不要使用 NVFP4 checkpoint 或 `configs/nvfp4/` 配置。
- 第一次在真实昇腾环境运行时，仍可能遇到 `torch_npu` 算子兼容问题。建议先单卡、短 prompt、默认配置跑通，再扩大到多卡。
