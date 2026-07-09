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

## 7. 是否需要训练

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

## 8. 注意事项

- 当前使用的是 `Wan2.2-TI2V-5B`，不是 Wan2.1。
- 昇腾上只走 BF16：`model_quant: false`、`kv_quant: false`、`torch_compile: false`。
- 不要使用 NVFP4 checkpoint 或 `configs/nvfp4/` 配置。
- 第一次在真实昇腾环境运行时，仍可能遇到 `torch_npu` 算子兼容问题。建议先单卡、短 prompt、默认配置跑通，再扩大到多卡。
