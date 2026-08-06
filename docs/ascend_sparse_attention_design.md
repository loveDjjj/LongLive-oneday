# 昇腾稀疏注意力选型

本文记录 LongLive2.0 HSA+CAG 在昇腾上的实现边界、上游调研结论和性能验收规则。
结论针对当前的滚动 KV cache、Ulysses SP4、BF16 和已完成 HSA+CAG 后训练的
Generator，不代表所有双向 DiT。

## 1. 上游实现状态

调研快照如下，后续升级依赖时应重新核对：

| 项目 | 快照 | 已确认能力 | 对当前模型的限制 |
|---|---|---|---|
| [LightForcing](https://github.com/chengtao-lv/LightForcing) | `d1e6333` | CAG、两级 HSA、Triton/FA4 稀疏执行 | CUDA 路径；FA4 使用 128 block |
| [MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) | `814babb` | 官方训练框架与昇腾并行能力 | master 未包含 SLA/SLA2 核心 |
| [MindSpeed-MM-SLA](https://gitcode.com/hyz22/MindSpeed-MM-SLA) | `136f886` | SLA v1、线性补偿分支、昇腾 Triton 稀疏 kernel | 个人研究分支；整段方形注意力；不包含 SLA2 |
| [HuuBar/sla2](https://github.com/HuuBar/sla2) | `081c1ac` | SLA2 router、alpha、两阶段蒸馏、RainFusion 重排 | 非 MindSpeed-MM 官方分支；需要新增可训练权重 |
| [MindIE-SD](https://gitcode.com/Ascend/MindIE-SD) | `3.0.0` / `6ecd51a` | RainFusion2.0、`aclnnRainFusionAttention` | 推理前向；最低 CANN 8.5.0，推荐官方 8.5.1/torch-npu 2.9 组合 |
| [MindIE-SD dev](https://gitcode.com/Ascend/MindIE-SD/tree/dev) | `a6814ab` | 新增 SparseLinearAttention、AscendC BSA、RF v3 | 开发分支依赖变化快，不能作为发布环境默认值 |
| [ops-transformer](https://gitcode.com/cann/ops-transformer) | `b06bd74` | RainFusionAttention 算子源码与产品约束 | 只支持 A2/910B、A3，不支持 950/A5 |
| [LightX2V](https://github.com/ModelTC/LightX2V) | `aaa7b26` | 910B 上通过 MindIE-SD 接入 RainFusion | 使用 RainFusion 自身路由，不等同于 HSA |

## 2. SLA、SLA2 与 HSA 的区别

SLA 不是单纯的块稀疏 softmax。它同时计算稀疏 softmax 分支和线性注意力分支，
并通过可训练投影补偿稀疏误差。SLA2 分支进一步引入每层 `proj_q/proj_k`、
每个 query block 的 `alpha`、两阶段 router 蒸馏和可选三维窗口重排。

因此，直接把现有 HSA+CAG Generator 的注意力调用替换成 SLA/SLA2 会遇到三个问题：

1. checkpoint 中不存在 SLA2 router、alpha 和线性分支权重；
2. SLA/SLA2 的训练目标与当前 DMD HSA+CAG 后训练目标不同；
3. 公开 SLA kernel 主要面向整段双向注意力，未直接覆盖滚动、矩形 Q/KV cache。

SLA2 可以作为后续新训练方案研究，但不能作为当前合并权重的无损推理后端。

## 3. 当前实现决策

当前路径保留训练时使用的 HSA+CAG 语义：

1. CAG 按生成 chunk 分配历史稀疏率；
2. HSA 先在 880 token 的 latent frame 上选历史帧；
3. 再对候选历史 token block 打分，并始终保留当前 8 帧 chunk；
4. 训练使用支持反向的 Ascend Triton 40-token kernel；
5. 推理把帧候选映射到 128-token block，并交给
   `aclnnRainFusionAttention` 执行。

这里使用的是 RainFusionAttention 的算子 ABI，不是用 RainFusion2.0 路由替换 HSA。
这样不需要新增模型权重，也不会把已有 HSA+CAG 后训练变成另一种稀疏算法。

官方算子源码确认：

- A2/910B 和 A3 支持，950/A5 不支持；
- Q/K/V 支持 FP16 和 BF16，BF16 必须使用 FP32 softmax；
- Q 与 KV 可以是不同长度，也允许非 block 对齐；
- `selectIdx` 必须按行升序，有效索引位于前缀；
- `selectIdx.shape[-1]` 是每行最大选择数，不要求填充到全部 KV block。

因此适配层直接传紧凑、升序的 HSA LUT，避免每层扩展到完整 KV 宽度。

## 4. 为什么暂不使用统一 `mindiesd.sparse_attention`

统一接口的 `rf_v2` 会自行执行块代表 token 路由、三维窗口重排和首帧保护。
这些设计适合普通双向视频 DiT，但会替换当前 HSA 的帧级选择、CAG schedule、
滚动 sink/local cache 语义。即使视频能生成，也不能证明仍在评测训练得到的
HSA+CAG 模型。

当前只复用其底层融合算子。若未来训练 SLA2/RainFusion router，应使用独立配置、
checkpoint 类型和 VBench 结果，不能与现有 HSA+CAG 结果混记。

MindIE-SD 还提供 `sparse_block_estimate`/`ada_bsa` 预处理算子，可以融合普通
block pooling 和阈值选块。但该接口生成自己的 CDF/一阶段 mask，不能接收 HSA
已经选出的历史 frame allowlist，因此也不能在不改变语义的前提下替换当前两级路由。

## 5. 性能与质量验收

正式启用必须按以下顺序通过：

1. BF16 全 LUT 输出与 dense attention 对齐；
2. BF16 稀疏 LUT 输出与 portable block-sparse 参考对齐；
3. 真实 SP4 尾部 `Q=7040`、`KV=28160` 下 `cached_full_ms < dense_ms`；
4. 32 秒 msprof 中 Attention、路由和通信分别有可解释的耗时下降；
5. 同一合并权重、提示词和 seed 的 VBench 20% 质量不低于既定容差。

若第三项失败，优先分析 128-block 路由和 Ulysses 通信，不继续调整 Triton 40-block
tile。若第四或第五项失败，保留 dense 推理为发布默认值，不以理论稀疏率代替实测。

## 6. 后续 SLA2 实验边界

SLA2 方向需要单独完成：router 数据采样、逐层 Stage-1 蒸馏、alpha/线性分支
Stage-2 训练、滚动矩形 cache 适配、合并权重格式和 VBench 回归。只有这些权重
真实训练并通过质量验收后，才考虑新增 `sla2` 推理方法；当前不创建看似可选但
实际使用随机或零初始化 router 的配置入口。
