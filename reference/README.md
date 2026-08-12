# 参考论文

本目录保存当前 LongLive2.0 昇腾稀疏注意力适配所依据的三篇论文。文件名统一使用小写英文、论文简称和 arXiv 编号，便于命令行检索与引用。

- `longlive-1-real-time-interactive-video-arxiv-2509.22622v2.pdf`
  - 论文：LongLive: Real-time Interactive Long Video Generation
  - 用途：核对 LongLive 1.x 的 Wan2.1-T2V-1.3B 基础模型、因果分块、局部注意力与 KV 策略。
- `longlive-2-nvfp4-parallel-infrastructure-arxiv-2605.18739v2.pdf`
  - 论文：LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation
  - 用途：核对 LongLive2.0-5B 的 Wan2.2 基础模型、滚动 KV、并行结构和性能设置。
- `light-forcing-sparse-attention-arxiv-2602.04789v3.pdf`
  - 论文：Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention
  - 用途：核对 HSA、CAG、稀疏率调度、保留帧策略与 GPU 稀疏注意力实验。

仓库中的昇腾实现是结合上述方法与 LongLive2.0 结构完成的工程适配，不应表述为论文官方 Ascend 实现。当前实现、训练后端和验收标准见[训练指南](../docs/training.md)与[推理与评测指南](../docs/inference_and_evaluation.md)。
