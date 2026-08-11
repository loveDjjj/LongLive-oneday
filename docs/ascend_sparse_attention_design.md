# 昇腾 SLA+CAG 技术选型

本文记录 LongLive2.0 滚动 KV、Ulysses SP4、BF16 场景下从 HSA+CAG 切换到
SLA+CAG 的依据、实现边界和验收规则。

## 1. 为什么停止 HSA 路线

真实 32 秒端到端结果没有复现 attention microbenchmark 的收益：Dense 为
`121.282 s / 6.308 FPS`，早期 HSA 为 `122.134 s`，最新 HSA 为
`128.025 s / 5.975 FPS`。最新结果比 Dense 慢约 5.6%。这说明单个稀疏 kernel
的约 2 倍加速被路由、布局转换、逐层调用和未稀疏部分抵消，HSA 不再作为发布路径。

## 2. 上游事实

- [Light Forcing](https://arxiv.org/abs/2602.04789) 提供 CAG：根据生成 chunk
  调整稀疏预算。其 HSA CUDA 实现不能直接成为昇腾训练 kernel。
- [MindSpeed-MM-SLA](https://gitcode.com/hyz22/MindSpeed-MM-SLA) 快照
  `136f886` 不只是前向。`mindspeed_mm/models/common/sla_kernel.py` 中包含
  `_attn_bwd_preprocess`、`_attn_bwd_dq`、`_attn_bwd_dkdv` 和
  `_attention.backward`，Q/K/V 反向是完整的 Ascend Triton 实现；线性分支与
  `proj_l` 使用 PyTorch autograd。
- 该上游 kernel 以单个 `L=q.shape[2]` 处理 Q 和 KV，原生假定方形注意力。
  LongLive 尾部是 `Q=7040`、`KV=28160`，不能原样调用。
- 当前服务器是 MindIE-SD 3.0.0、CANN 8.5.0，并提示 Triton-Ascend 低于 3.2.1，
  因此统一 `SparseLinearAttention` 类不可用。现有
  `RainFusionAttention` 可承担推理时的矩形稀疏 softmax 前向，但不能承担训练反向。

所以，“官方仓库没有 SLA 反向”不是问题；真正缺口是滚动矩形 Q/KV 适配，以及当前
服务器版本无法直接启用新版统一 SLA 类。

## 3. 当前实现

当前分支实现 SLA v1 + CAG，而不是 SLA2：

1. 训练使用从 MindSpeed-MM-SLA 思路适配的 Ascend Triton kernel，分别接收
   Q 长度和 KV 长度，支持矩形前向及 Q/K/V 反向。
2. 每个 Q/K 128-token block 求代表向量，对全部滚动 KV 做全局 Top-K。
3. CAG 按 AR chunk 分配 Top-K 预算；sink/recent blocks 强制保留。
4. 默认 `dense_current=false`。尾部强制块约 14/220，实际稀疏率约 93.6%；若保持
   当前 8 帧稠密，理论稀疏率最多 75%，不够覆盖线性补偿分支开销。
5. 稀疏 softmax 输出叠加线性注意力补偿。`sla_linear` 从零初始化并进入 LoRA
   后训练，因此初始输出等于稀疏分支，训练后再学习补偿。
6. 推理缓存历史 block representatives 和线性注意力统计量；新 chunk 或 recache
   时失效。训练路径不复用无梯度缓存。
7. 推理的稀疏 softmax 分支复用 MindIE-SD RainFusionAttention，线性分支仍由
   PyTorch NPU 算子执行。

旧 HSA LoRA 不能直接转换为 SLA LoRA。训练 checkpoint 写入
`sparse_method: sla_cag`，恢复时拒绝旧 HSA 和无标签状态。旧的稠密 LongLive
基础权重可以加载，缺失的 SLA 投影会严格按零补齐。

## 4. 为什么暂不采用 SLA2

SLA2 还需要额外的 `proj_q/proj_k`、逐 query-block alpha、router 蒸馏和新的训练
阶段。直接给这些参数随机或零初始化后推理不具备算法意义，也不能和既有 VBench
结果比较。当前先验证结构更小、公开前后向更清晰的 SLA v1；若仍不能端到端加速，
应先定位线性分支和路由成本，不继续扩大模型改动。

## 5. 验收门槛

必须依次满足：

1. 全 LUT BF16 输出与 dense 对齐，稀疏 LUT 与 portable 参考对齐。
2. Ascend Triton 矩形 Q/K/V 前向、反向无 NaN/Inf，梯度误差在 smoke test 容差内。
3. `Q=7040`、`KV=28160` 下完整 `cached_full_ms < dense median_ms`；不能只看
   sparse kernel 时间。
4. 同配置 32 秒 msprof 的 generation time 和 FPS 优于 Dense，且至少重复 3 次，
   报告 median，排除编译、预热和保存视频时间。
5. 使用同一 checkpoint、提示词、seed、帧数的 Dense/SLA VBench 20% 对照通过质量容差。

第三项失败就不启动长训练；第四项失败则保持 Dense 为发布默认值。SLA 是否可行最终
取决于这两个实测门槛，而不是上游名称或理论稀疏率。
