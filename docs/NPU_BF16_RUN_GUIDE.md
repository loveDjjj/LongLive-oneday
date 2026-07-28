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
| `vbench_standard_5s_npu_bf16.yaml` | 32 | 5.21 秒 | VBench Standard |
| `vbench_long_60s_npu_bf16.yaml` | 360 | 59.88 秒 | VBench-Long |
| `perf_16s_npu_bf16.yaml` | 96 | 15.88 秒 | 性能档位 |
| `perf_32s_npu_bf16.yaml` | 192 | 31.88 秒 | 性能档位 |
| `perf_64s_npu_bf16.yaml` | 384 | 63.88 秒 | 性能档位 |

这些配置均设置 `streaming_vae: false`，保证 VAE 按完整时间序列解码。当前 Wan VAE 不支持
真正的 `cached_decode`；如果把普通 `decode` 按 8 latent 帧独立调用，会得到错误的帧数并在
block 边界重置时间缓存。例如 128 latent 帧会输出约 19.33 秒，而不是正确整段解码的
21.21 秒。该分块 fallback 不能用于正式质量评测。

### 7.3 先运行 AISBench mini 单样本链路

建议先用 8 卡 SP、单 DP 组生成 mini 视频：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LLV2_DEVICE=npu torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  inference_sp.py \
  --config_path configs/benchmarks/vbench_mini_5s_npu_bf16.yaml
```

当前推理入口按数据集序号保存文件。生成结束后，先把序号映射回 prompt，并按 VBench 的
`{prompt}-{sample_index}.mp4` 规则创建评测目录：

```bash
python - <<'PY'
import re
import shutil
from pathlib import Path

root = Path("/mnt/share/r50063443/LongLive-oneday")
src_dir = root / "videos/benchmarks/vbench_mini_5s"
dst_dir = root / "videos/benchmarks/vbench_mini_5s_vbench"
prompts = (root / "data/benchmarks/vbench_mini/prompts.txt").read_text(
    encoding="utf-8"
).splitlines()
dst_dir.mkdir(parents=True, exist_ok=True)

for src in src_dir.glob("rank*-*-0_*.mp4"):
    match = re.match(r"rank\d+-(\d+)-0_", src.name)
    if match is None:
        continue
    prompt_idx = int(match.group(1))
    shutil.copy2(src, dst_dir / f"{prompts[prompt_idx]}-0.mp4")

print(f"prepared {len(list(dst_dir.glob('*.mp4')))} videos in {dst_dir}")
PY
```

然后在已经部署好的 AISBench 仓库中，将 VBench 示例配置改为：

```python
DATA_PATH = "/mnt/share/r50063443/LongLive-oneday/videos/benchmarks/vbench_mini_5s_vbench"

vbench_eval_cfg = dict(
    load_ckpt_from_local=True,
    full_json_dir="/mnt/share/r50063443/LongLive-oneday/data/benchmarks/vbench_mini/VBench_full_info.json",
)
```

在 AISBench 仓库根目录执行：

```bash
ais_bench ais_bench/configs/vbench_examples/eval_vbench_standard.py \
  --mode eval \
  --max-num-workers 8
```

上述 mini 配置只有 1 个 seed，用于验证“LongLive 生成 -> 文件整理 -> AISBench 评分”完整
链路。AISBench Standard 的正式协议要求视频文件名为 `{prompt}-{index}.mp4`，每条普通
prompt 需要 5 个不同 seed，`temporal_flickering` 需要 25 个。正式评测时应修改
`logging.seed` 后分 5 次顺序生成，temporal flickering 子集分 25 次生成，并在每轮结束后把
文件整理成 `{prompt}-0.mp4` 至 `{prompt}-4.mp4` 或 `{prompt}-24.mp4`。不要直接把
`num_samples` 改成 5 或 25，否则会把样本放进同一 batch，显著增加显存。缺少这些重复样本
时，所得结果只能标记为 single-sample diagnostic，不能报告为完整 VBench Standard 分数。

### 7.4 完整 VBench 和 VBench-Long

完整 5 秒 prompt 生成命令：

```bash
LLV2_DEVICE=npu torchrun \
  --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29501 \
  inference_sp.py \
  --config_path configs/benchmarks/vbench_standard_5s_npu_bf16.yaml
```

60 秒 VBench-Long 命令：

```bash
LLV2_DEVICE=npu torchrun \
  --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29501 \
  inference_sp.py \
  --config_path configs/benchmarks/vbench_long_60s_npu_bf16.yaml
```

60 秒整段 VAE 解码显存较高。先用单条 prompt 验证；如果 OOM，不要切回错误的逐 block
普通解码，应先实现保持时间缓存的 Wan VAE streaming decode。

原生 VBench-Long 测评示例：

```bash
python vbench2_beta_long/eval_long.py \
  --videos_path /mnt/share/r50063443/LongLive-oneday/videos/benchmarks/vbench_long_60s \
  --dimension subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
  --mode long_vbench_standard \
  --num_of_samples_per_prompt 1 \
  --dev_flag
```

该命令同样要求视频按 `{prompt}-0.mp4` 命名。可以复用 7.3 的整理脚本，只需把
`vbench_mini_5s`、`vbench_mini_5s_vbench` 和 prompt 文件路径分别替换成
`vbench_long_60s`、`vbench_long_60s_vbench` 和 `vbench_long/prompts.txt`。这里显式使用
`--num_of_samples_per_prompt 1` 是为了与当前配置一致；正式多样本协议需生成 5 个 seed。

### 7.5 昇腾性能测试

依次运行 `perf_16s_npu_bf16.yaml`、`perf_32s_npu_bf16.yaml` 和
`perf_64s_npu_bf16.yaml`。8 卡配置 `sp_size=8, dp_size=1` 用于单条视频延迟；16 卡吞吐测试
把配置改为 `dp_size=2`，并用 `--nproc_per_node=16` 启动。16 卡是两个 8 卡 SP 组，不会
降低单条视频延迟。

每档至少记录：端到端延迟、diffusion 时间、VAE 时间、最终像素帧数、端到端 FPS、RTF、
每卡峰值 HBM，以及 16 卡时的视频/小时。性能测试建议预热 2 次，再测 5 次；64 秒至少测
3 次。AISBench 的 VBench 运行时间是评测器性能，不是 LongLive 生成性能，应分别记录。

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
