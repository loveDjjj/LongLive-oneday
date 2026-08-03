<p align="center" style="border-radius: 10px">
  <img src="assets/longlive2/logo.png" width="100%" alt="LongLive2.0 logo"/>
</p>

# 🎬 LongLive 2.0: An NVFP4 Parallel Infrastructure for Long Video Generation

[![Paper](https://img.shields.io/badge/Paper-LongLive_2.0-brown)](https://arxiv.org/abs/2605.18739)
[![Paper](https://img.shields.io/badge/Paper-LongLive_1.0-orange)](https://github.com/NVlabs/LongLive/tree/v1.0)
[![Paper](https://img.shields.io/badge/Paper-LongLive_RAG-yellow)](https://github.com/qixinhu11/LongLive-RAG)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=7oQALy32fiU)
[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/NVlabs/LongLive)
[![Demo](https://img.shields.io/badge/Demo-Page-green)](https://nvlabs.github.io/LongLive/LongLive2/)
[![Docs](https://img.shields.io/badge/Full-Documentation-brightgreen)](https://nvlabs.github.io/LongLive/LongLive2/docs/)


<div align="center">

<!-- TODO: replace this text block with the final project-page video/demo embed. -->

[![Watch the video](assets/longlive2/first-video-frame.png)](https://www.youtube.com/watch?v=7oQALy32fiU)

</div>

## 💡 TLDR: Infra with NVFP4 and parallelism for both training and inference

<p align="center" style="border-radius: 10px">
  <img src="assets/longlive2/teaser.jpg" width="100%" alt="LongLive2.0 teaser"/>
</p>

## News
- 🔥 [2026.06.01] We released [LongLive-RAG](https://github.com/qixinhu11/LongLive-RAG), a general retrieval-augmented framework for long video gen.
- 🔥 [2026.05.30] LongLive2.0 now supports I2V AR teacher-forcing training and I2V DMD distillation for Wan2.2-TI2V-5B.
- ⚡ [2026.05.25] We optimized the NVFP4 inference path with fused Triton RoPE/adaLN kernels, reduced KV-cache synchronization overhead, in-place quantized KV-cache updates, faster FP4 KV dequantization, pinned VAE transfers, and safer LoRA-before-quantization setup, improving overall throughput by **18.6%**.
- 🔥 [2026.05.13] We release **LongLive 2.0**, infra with NVFP4, parallelism and multi-shot for AR training, DMD distillation, and inference (⚡45.7 FPS). The original LongLive 1.0 is now in the [v1.0](https://github.com/NVlabs/LongLive/tree/v1.0) branch.
- 🔥 [2026.04.12] LongLive supports kv cache compression with [TriAttention](https://github.com/WeianMao/triattention), with 50% KV reduction and no quality drop. Check it [here](https://github.com/WeianMao/triattention/tree/main/longlive)
- 🎉 [2026.1.27] LongLive is accepted by **ICLR-2026**.
- 🔥 [2026.1.11] LongLive supports adapting LongLive's original RoPE into KV-cache relative RoPE and generates infinite long videos!
- 🔥 [2025.11.3] We implement LongLive on linear attention model [SANA-Video](https://nvlabs.github.io/Sana/Video/)! Now SANA-Video can generate 60s interactive videos in real-time.
- 🔥 [2025.9.29] We release [Paper](https://arxiv.org/abs/2509.22622), this GitHub repo [LongLive](https://github.com/NVlabs/LongLive) with all training and inference code, the model weight [LongLive-1.3B](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B), and demo page [Website](https://nvlabs.github.io/LongLive).

## Introduction

**LongLive 1.0**: Real-time Interactive Long Video Generation. [You can find it here](https://github.com/NVlabs/LongLive/tree/v1.0) in our V1.0 branch.

**LongLive 2.0**: an NVFP4 Parallel Infrastructure for Long Video Generation
- For training, it supports
  - [x] Balanced sequence parallel for T2V/I2V AR training (teacher-forcing).
  - [x] T2V/I2V AR training on multi-shot (or single-shot) videos.
  - [x] NVFP4 (or BF16) for both AR training and few-step distillation.
- For inference, it supports
  - [x] NVFP4 inference (W4A4) and NVFP4 KV Cache.
  - [x] Multi-shot attention sink.
  - [x] Sequence parallel inference.
  - [x] Async decoding.


<p align="left" style="border-radius: 10px">
  <img src="assets/longlive2/fig_framework_overview.png" width="80%" alt="LongLive2.0 framework overview"/>
</p>


**LongLive 1.0**: Real-time Interactive Long Video Generation. It accepts sequential user prompts and generates corresponding videos in real time, enabling user-guided long video generation. The key insights are attention sink, KV-recache, and streaming long tuning. 


<p align="left" style="border-radius: 10px">
  <img src="assets/longlive2/LongLive1_teaser.png" width="80%" alt="LongLive1.0 framework overview"/>
</p>

## Getting Started
- [Full Documentation](https://nvlabs.github.io/LongLive/LongLive2/docs/)
- [Installation](https://nvlabs.github.io/LongLive/LongLive2/docs/#installation)
- [NVFP4 Setup](https://nvlabs.github.io/LongLive/LongLive2/docs/#nvfp4-installation)
- [Training Modes](https://nvlabs.github.io/LongLive/LongLive2/docs/#training)
- [Inference](https://nvlabs.github.io/LongLive/LongLive2/docs/#inference)
- [Data Organization](https://nvlabs.github.io/LongLive/LongLive2/docs/#training-data)


The default git clone fetches objects from all branches, including our demopage branch, which contains large assets. For normal use, only the main branch is needed. Please clone only main with:

```git clone --single-branch --branch main --depth 1 https://github.com/NVlabs/LongLive.git```

### Quick Start

The `npu` branch maintains two Ascend BF16 inference workflows:

```bash
bash scripts/run_msprof.sh 32s
bash scripts/run_vbench.sh longlive2_standard_20pct
```

Runtime presets live in `configs/inference/`. See
[`docs/NPU_BF16_RUN_GUIDE.md`](docs/NPU_BF16_RUN_GUIDE.md) for device layout,
environment variables, datasets, output naming, and profiler semantics. NVFP4
configs are intentionally not shipped on this Ascend-only branch.

## Training Modes

LongLive2.0 supports both T2V and I2V training. Each modality follows the same two-stage recipe: AR teacher-forcing training first, then DMD distillation from the AR checkpoint.

The Ascend HSA+CAG sparse-attention adaptation, prompt preparation, DMD recipe,
and dense/sparse evaluation procedure are documented in
[`docs/hsa_cag_training.md`](docs/hsa_cag_training.md).

### T2V Training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config_path configs/train_ar.yaml \
  --logdir logs/train_ar \
  --wandb-save-dir wandb \
  --disable-wandb

torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config_path configs/train_dmd.yaml \
  --logdir logs/train_dmd \
  --wandb-save-dir wandb \
  --disable-wandb
```

### I2V Training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config_path configs/train_i2v_ar.yaml \
  --logdir logs/train_i2v_ar \
  --wandb-save-dir wandb \
  --disable-wandb

torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config_path configs/train_i2v_dmd.yaml \
  --logdir logs/train_i2v_dmd \
  --wandb-save-dir wandb \
  --disable-wandb
```

For I2V configs, set `algorithm.i2v: true` and `algorithm.independent_first_frame: true`. `data.image_or_video_shape[1]` is the full latent sequence length, for example `96`, not `96 + 1`: the clean image latent replaces the first latent during denoising and that first latent is masked out of the training loss. For I2V DMD, set `checkpoints.generator_ckpt` to the I2V AR checkpoint used to initialize the student.

## Models

| Model | FPS ↑ | Params | VBench ↑ | Multi-shot |
| --- | ---: | ---: | ---: | :---: |
| [LongLive-1.3B](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B) | 20.7 | 1.3B | 84.87 |  |
| [LongLive-2.0-5B](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B) | 24.8 | 5B | 85.06 | ✅ |
| [LongLive-2.0-5B-NVFP4-4Step](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4) | 29.7 | 5B | 84.51 | ✅ |
| [LongLive-2.0-5B-NVFP4-2Step](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2) | 45.7 | 5B | 83.14 | ✅ |

## License
This repository is released under the Apache 2.0 license. See [LICENSE](LICENSE) for details.

## Citation
Please consider citing our work if you find them useful:

```bibtex
@article{longlive_2.0,
  title={LongLive2.0: An NVFP4 Parallel Infrastructure for Long Video Generation},
  author={Chen, Yukang and Wang, Luozhou and Huang, Wei and Yang, Shuai and Zhang, Bohan and Xiao, Yicheng and Chu, Ruihang and Mao, Weian and Hu, Qixin and Liu, Shaoteng and Zhao, Yuyang and Mao, Huizi and Chen, Ying-Cong and Xie, Enze and Qi, Xiaojuan and Han, Song},
  journal={arXiv preprint arXiv: 2605.18739},
  year={2026}
}
```

```bibtex
@inproceedings{longlive,
    title={Longlive: Real-time interactive long video generation}, 
    author={Yang, Shuai and Huang, Wei and Chu, Ruihang and Xiao, Yicheng and Zhao, Yuyang and Wang, Xianbang and Li, Muyang and Xie, Enze and Chen, Yingcong and Lu, Yao and others},
    booktitle={ICLR},
    year={2026},
}
```

```bibtex
@article{longlive_rag,
  title         = {LongLive-RAG: A General Retrieval-Augmented Framework for Long Video Generation},
  author        = {Hu, Qixin and Yang, Shuai and Huang, Wei and Han, Song and Chen, Yukang},
  journal       = {arXiv preprint arXiv:2606.02553},
  year          = {2026}
}
```

## Acknowledgement
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing): the AR training codebase and formulation we build upon.
- [Wan2.2](https://github.com/Wan-Video/Wan2.2): the base video diffusion model components used in this release.
