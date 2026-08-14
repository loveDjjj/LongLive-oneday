# LongLive2.0 稀疏注意力质量分析：基础权重与 200-step 后训练

## 1. 报告范围

本文分析 LongLive2.0-5B 在 VBench Standard 5% 上的两阶段质量结果。第一阶段使用基础权重比较四种注意力路径：

```text
dense
hsa_cag
sla_cag
hsa_sla_cag
```

所有 case 使用同一基础 checkpoint：

```text
/mnt/share/weight/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt
```

该路径来自本次历史 run 的原始记录，仅用于保证实验可追溯；当前服务器默认权重路径已迁移到 `/mnt/a800_share/r50063443/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt`，本文不改写历史字段。

第二阶段使用完成 200 step `lora_plus_linear` 训练并导出的 SLA+CAG 与 HSA+SLA+CAG 两份权重，分别运行 `dense`、`sla_cag` 与 `hsa_sla_cag`，形成训练权重 × 推理路由的完整 `2×3` 消融。该设计用于区分 Generator 主干 LoRA 后训练、运行时稀疏路由及二者交互造成的变化。

| 报告元数据 | 值 |
| --- | --- |
| 数据日期 | 基础权重：2026-08-13；训练后权重：2026-08-13 至 2026-08-14 |
| 仓库分支 | `feat/unified-sparse-attention` |
| 首次整理提交 | `67145af` |
| 数据类型 | AISBench VBench 聚合结果 |
| 报告状态 | 基础权重与 200-step 后训练 5% 阶段性分析 |

## 2. 实验条件与可比性

| 项目 | 设置 |
| --- | --- |
| 数据集 | `longlive2_standard_5pct` |
| 提示词 | 43 条 VBench Standard 5% K-Means Mini 原始提示词 |
| seed | 0、1、2、3、4 |
| 每组视频数 | 215 |
| 视频长度 | 125 pixel 帧，约 5.2 秒 |
| FPS | 24 |
| 采样步数 | 4 |
| 并行布局 | SP2 × DP8 |
| checkpoint | 四组完全相同 |
| 稀疏后端 | MindIE-SD RainFusion |
| 评测器 | AISBench VBench |

训练后六组沿用相同的 VBench preset、提示词、seed、视频长度和采样步数。按实验设计，每种训练权重内部的三种路由使用同一个导出 checkpoint：

| 训练后 run | 注意力 | 评测时间 |
| --- | --- | --- |
| `longlive2-hybrid-200step-dense-5pct` | dense | `20260814_001749` |
| `longlive2-hybrid-200step-sla_cag-5pct` | SLA+CAG | `20260814_120639` |
| `longlive2-hybrid-200step-hsa_sla_cag-5pct` | HSA+SLA+CAG | `20260813_172004` |
| `longlive2-sla-200step-dense-5pct` | dense | `20260814_131156` |
| `longlive2-sla-200step-sla_cag-5pct` | SLA+CAG | `20260814_123352` |
| `longlive2-sla-200step-hsa_sla_cag-5pct` | HSA+SLA+CAG | `20260814_125732` |

严格配对结论仍要求对应 run 的 manifest、resolved YAML、checkpoint SHA 和 SP/DP 布局一致。本文依据既定测试矩阵和用户回传结果进行分析；本地仓库未包含服务器 run 产物，因此原始运行目录是最终审计依据。如果 SLA 新增组使用 `SP4 × DP2` 而既有 Hybrid 组使用 `SP2 × DP8`，则跨训练权重比较只能视为阶段性观察，不能作为严格配对结论。

基础 checkpoint 不包含后续引入的 `sla_linear` 参数。加载 SLA/Hybrid 模型结构时，代码只为缺失的 `sla_linear` 合成严格的零初始化，因此：

```text
基础 SLA/Hybrid = 稀疏 softmax 路由 + 零线性补偿
```

SLA/Hybrid 使用 `0.90/0.93` CAG 目标/基准稀疏率；HSA 使用 `0.85/0.95`，并保留 6 个历史 latent 帧。故本报告比较的是当前各方法默认配置，不是相同实际稀疏率下的纯路由消融。

## 3. 指标与计算规范

所有官方维度和聚合分数均采用 AISBench 输出，不重新定义权重。本文计算：

```text
分数差 = sparse 分数 - dense 分数
相对变化率 = (sparse 分数 - dense 分数) / dense 分数 × 100%
```

正文主要报告“分数差”，单位为分或百分点。例如 `82.83 → 77.79` 表述为下降 `5.04` 分，而不是下降 `5.04%`。红色粗体标记主要退化或最重要结论。

派生差值使用用户提供的原始小数精度计算，正文统一保留 2 位小数；相对变化率也保留 2 位。由于展示表已取整，用展示值再次相减可能出现不超过 0.01 分的末位差异。

## 4. 官方聚合分数

### 4.1 原始分数与相对 dense 差值

| 方法 | Quality | Δ Quality | Semantic | Δ Semantic | Total | Δ Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 85.54 | 基线 | 71.99 | 基线 | **82.83** | 基线 |
| hsa_cag | 85.43 | -0.11 | **72.26** | +0.27 | 82.80 | **-0.03** |
| sla_cag | 80.63 | <span style="color:#c00000"><strong>-4.91</strong></span> | 66.43 | <span style="color:#c00000"><strong>-5.56</strong></span> | 77.79 | <span style="color:#c00000"><strong>-5.04</strong></span> |
| hsa_sla_cag | 80.49 | <span style="color:#c00000"><strong>-5.05</strong></span> | 66.79 | <span style="color:#c00000"><strong>-5.20</strong></span> | 77.75 | <span style="color:#c00000"><strong>-5.08</strong></span> |

### 4.2 相对变化率

| 方法 | Quality 相对变化 | Semantic 相对变化 | Total 相对变化 |
| --- | ---: | ---: | ---: |
| hsa_cag | -0.13% | +0.38% | -0.04% |
| sla_cag | -5.74% | -7.72% | -6.08% |
| hsa_sla_cag | -5.90% | -7.22% | -6.13% |

HSA 在该 5% 子集上的 Total 观测差异只有 -0.03 分，不能据此宣称统计等价，但可以表述为“未观察到有实际量级的聚合质量下降”。SLA 与 Hybrid 的 Total 分别下降 5.04 和 5.08 分，幅度远高于 HSA 的观测波动，需要视为明确的质量风险并通过后训练验证。

## 5. 16 个官方维度：原始分数

| VBench 维度 | dense | hsa_cag | sla_cag | hsa_sla_cag |
| --- | ---: | ---: | ---: | ---: |
| Aesthetic Quality | 60.70 | **61.32** | 57.35 | 56.93 |
| Appearance Style | 18.03 | 18.04 | 18.91 | **18.91** |
| Background Consistency | **94.61** | 94.38 | 86.89 | 86.85 |
| Color | 86.14 | **87.83** | 87.02 | 86.95 |
| Dynamic Degree | **93.33** | **93.33** | **93.33** | **93.33** |
| Human Action | 96.00 | 96.00 | **100.00** | **100.00** |
| Imaging Quality | **66.08** | 65.51 | 59.88 | 59.21 |
| Motion Smoothness | **98.56** | 98.52 | 98.46 | 98.52 |
| Multiple Objects | **53.44** | 51.25 | 38.13 | 37.81 |
| Object Class | **76.67** | **76.67** | 52.50 | 55.83 |
| Overall Consistency | 23.32 | 23.33 | **23.70** | 23.68 |
| Scene | 39.06 | **41.88** | 38.13 | 39.06 |
| Spatial Relationship | **91.54** | 90.88 | 71.78 | 71.13 |
| Subject Consistency | **95.74** | 95.66 | 88.27 | 88.30 |
| Temporal Flickering | **99.60** | 99.53 | 98.55 | 98.55 |
| Temporal Style | 25.29 | 25.34 | **25.78** | 25.73 |

## 6. 16 个官方维度：相对 dense 差值

正值表示高于 dense，负值表示低于 dense。

| VBench 维度 | HSA Δ | SLA Δ | Hybrid Δ |
| --- | ---: | ---: | ---: |
| Aesthetic Quality | +0.62 | -3.36 | -3.77 |
| Appearance Style | +0.01 | +0.88 | +0.88 |
| Background Consistency | -0.23 | <span style="color:#c00000"><strong>-7.72</strong></span> | <span style="color:#c00000"><strong>-7.76</strong></span> |
| Color | +1.69 | +0.88 | +0.81 |
| Dynamic Degree | 0.00 | 0.00 | 0.00 |
| Human Action | 0.00 | +4.00 | +4.00 |
| Imaging Quality | -0.58 | <span style="color:#c00000"><strong>-6.20</strong></span> | <span style="color:#c00000"><strong>-6.87</strong></span> |
| Motion Smoothness | -0.04 | -0.10 | -0.04 |
| Multiple Objects | -2.19 | <span style="color:#c00000"><strong>-15.31</strong></span> | <span style="color:#c00000"><strong>-15.63</strong></span> |
| Object Class | 0.00 | <span style="color:#c00000"><strong>-24.17</strong></span> | <span style="color:#c00000"><strong>-20.83</strong></span> |
| Overall Consistency | +0.01 | +0.38 | +0.36 |
| Scene | +2.81 | -0.94 | 0.00 |
| Spatial Relationship | -0.65 | <span style="color:#c00000"><strong>-19.75</strong></span> | <span style="color:#c00000"><strong>-20.41</strong></span> |
| Subject Consistency | -0.08 | <span style="color:#c00000"><strong>-7.47</strong></span> | <span style="color:#c00000"><strong>-7.44</strong></span> |
| Temporal Flickering | -0.06 | -1.05 | -1.05 |
| Temporal Style | +0.04 | +0.49 | +0.44 |

## 7. 退化模式分析

### 7.1 HSA+CAG

HSA 的 16 个维度中，最大下降为 Multiple Objects 的 -2.19 分，最大提高为 Scene 的 +2.81 分。其聚合结果为：

```text
Quality -0.11
Semantic +0.27
Total -0.03
```

Object Class、Human Action 和 Dynamic Degree 与 dense 相同；Subject/Background Consistency、Motion Smoothness 和 Temporal Flickering 的差值均小于 0.25 分。当前证据支持 HSA 与基础权重具有较好的 zero-shot 兼容性。

但 5% 子集各维度样本量有限，Scene、Color 等单维提高不能直接解释为 HSA 改善了模型能力，可能包含抽样、评测器离散性和生成随机性的影响。

### 7.2 SLA+CAG

SLA 的主要退化集中在对象语义、空间关系和一致性：

| 退化排名 | 维度 | Δ dense |
| ---: | --- | ---: |
| 1 | Object Class | <span style="color:#c00000"><strong>-24.17</strong></span> |
| 2 | Spatial Relationship | <span style="color:#c00000"><strong>-19.75</strong></span> |
| 3 | Multiple Objects | <span style="color:#c00000"><strong>-15.31</strong></span> |
| 4 | Background Consistency | -7.72 |
| 5 | Subject Consistency | -7.47 |
| 6 | Imaging Quality | -6.20 |

与此同时，Motion Smoothness 仅下降 0.10 分，Dynamic Degree 不变，Temporal Style 提高 0.49 分，Human Action 提高 4 分。该模式表明当前零补偿 SLA 的主要问题不是视频运动完全崩溃，而是高稀疏率下对象身份、对象组合与空间结构信息不足。

Human Action 从 96 提升至 100 处于指标上界附近，且 5% 子集样本较少，存在明显天花板效应；不能用这一项抵消对象与空间维度的大幅下降。

### 7.3 HSA+SLA+CAG

Hybrid 的退化模式与 SLA 高度一致：Multiple Objects、Spatial Relationship、Background/Subject Consistency 和 Imaging Quality 均大幅下降。与 SLA 的直接差值为：

| 指标 | Hybrid - SLA |
| --- | ---: |
| Quality | -0.14 |
| Semantic | +0.36 |
| Total | -0.04 |
| Object Class | +3.33 |
| Scene | +0.94 |
| Imaging Quality | -0.67 |
| Spatial Relationship | -0.65 |
| Multiple Objects | -0.31 |

Hybrid 在 Object Class 上优于 SLA 3.33 分，但 Spatial Relationship、Imaging Quality 和 Multiple Objects 更低，最终 Total 仅相差 -0.04 分。<span style="color:#c00000"><strong>当前基础权重实验不能证明 HSA 候选帧预筛选为 SLA 带来了稳定的总体质量收益</strong></span>。

## 8. 性能—质量联合判断

结合相同基础权重的 DiT-only 性能结果：

| 方法 | 32s SP4 p50 加速比 | VBench Total | Δ dense Total | 当前判断 |
| --- | ---: | ---: | ---: | --- |
| dense | 1.000x | 82.83 | 基线 | 质量基线 |
| hsa_cag | 1.217x | **82.80** | **-0.03** | 基础权重上最稳健的速度/质量折中 |
| sla_cag | **1.290x** | 77.79 | <span style="color:#c00000"><strong>-5.04</strong></span> | 速度最高，但 zero-shot 质量不可接受 |
| hsa_sla_cag | 1.268x | 77.75 | <span style="color:#c00000"><strong>-5.08</strong></span> | 比 SLA 更慢，基础权重质量未改善 |

基础权重阶段，HSA 是唯一同时表现出明确 DiT 加速和近似保持聚合质量的方法。第 10 节进一步表明，200-step 后训练能够恢复 SLA 与 Hybrid 的部分质量并产生路由专用化，但尚未消除运行时稀疏代价。

## 9. 机制解释与证据边界

可由当前数据支持的表述：

> 在 LongLive2.0 基础权重、零初始化 `sla_linear`、当前默认稀疏预算和 VBench Standard 5% 条件下，SLA+CAG 与 HSA+SLA+CAG 的 Total 分别比 dense 低 5.04 和 5.08 分，退化集中于对象类别、多对象、空间关系和一致性维度。

基础权重数据单独不能支持以下更强结论：

- “SLA 算法必然损失约 5 分”：200-step 训练结果已证明损失可以部分恢复，但不能外推到其他步数、稀疏率或数据规模。
- “HSA 与 dense 质量等价”：没有逐视频分数和置信区间，不能做等价性检验。
- “Hybrid 不如 SLA”：两者 Total 只差 0.04 分，远小于该子集可可靠解释的尺度。
- “Human Action 得到提升”：该指标接近满分，可能是小样本和天花板效应。

## 10. 200-step 训练权重 × 推理路由消融

### 10.1 完整 `2×3` 官方聚合分数

| 训练权重 | 推理路由 | Quality | Semantic | Total | 相对同权重 dense |
| --- | --- | ---: | ---: | ---: | ---: |
| SLA-trained | dense | **85.61** | **71.29** | **82.75** | 基线 |
| SLA-trained | SLA | 81.73 | 70.33 | 79.45 | <span style="color:#c00000"><strong>-3.30</strong></span> |
| SLA-trained | Hybrid | 81.14 | 68.93 | 78.70 | <span style="color:#c00000"><strong>-4.05</strong></span> |
| Hybrid-trained | dense | **85.81** | **70.88** | **82.82** | 基线 |
| Hybrid-trained | SLA | 81.06 | 68.66 | 78.58 | <span style="color:#c00000"><strong>-4.24</strong></span> |
| Hybrid-trained | Hybrid | 82.22 | 69.65 | 79.71 | <span style="color:#c00000"><strong>-3.12</strong></span> |

两种训练权重的 dense Total 分别为 82.75 和 82.82，相比基础权重 dense 82.83 仅变化 -0.08 和 -0.01 分。主干 LoRA 后训练没有造成可见量级的总体质量下降，约 3 至 4 分的剩余差距主要来自运行时稀疏路径。

在各自匹配路由下，Hybrid-trained + Hybrid 比 SLA-trained + SLA 高 0.26 分，但 5% 子集没有逐视频方差和置信区间，不能据此宣称 Hybrid 显著优于 SLA。基础权重 HSA+CAG 的 Total 为 82.80，仍明显高于两个 200-step 匹配稀疏方案。

### 10.2 全部 16 个维度

| VBench 维度 | SLA权重 dense | SLA权重 SLA | SLA权重 Hybrid | Hybrid权重 dense | Hybrid权重 SLA | Hybrid权重 Hybrid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aesthetic Quality | 60.72 | 58.63 | 58.25 | 60.30 | 57.78 | 58.38 |
| Appearance Style | 18.43 | 19.10 | 19.21 | 19.34 | 19.44 | 18.83 |
| Background Consistency | 94.19 | 88.53 | 89.75 | 95.29 | 88.36 | 88.41 |
| Color | 83.31 | 86.32 | 80.10 | 82.50 | 78.57 | 84.46 |
| Dynamic Degree | 93.33 | 86.67 | 73.33 | 93.33 | 80.00 | 93.33 |
| Human Action | 96.00 | 96.00 | 96.00 | 92.00 | 100.00 | 100.00 |
| Imaging Quality | 67.20 | 63.94 | 63.15 | 65.65 | 62.65 | 63.59 |
| Motion Smoothness | 98.51 | 98.54 | 98.61 | 98.63 | 98.66 | 98.55 |
| Multiple Objects | 48.44 | 48.44 | 49.38 | 61.25 | 58.44 | 45.00 |
| Object Class | 77.08 | 70.00 | 78.33 | 80.42 | 70.42 | 68.33 |
| Overall Consistency | 23.26 | 23.65 | 23.81 | 22.96 | 23.36 | 23.37 |
| Scene | 41.25 | 41.25 | 31.56 | 22.81 | 23.12 | 41.88 |
| Spatial Relationship | 88.30 | 79.26 | 74.66 | 92.75 | 79.10 | 77.34 |
| Subject Consistency | 95.85 | 90.00 | 91.70 | 96.64 | 90.83 | 90.46 |
| Temporal Flickering | 99.59 | 98.79 | 98.83 | 99.74 | 98.79 | 98.81 |
| Temporal Style | 25.46 | 25.84 | 25.80 | 25.68 | 25.89 | 25.75 |

### 10.3 匹配路由的残余损失

| 维度 | SLA-trained SLA - dense | Hybrid-trained Hybrid - dense |
| --- | ---: | ---: |
| Subject Consistency | -5.85 | -6.18 |
| Background Consistency | -5.66 | -6.88 |
| Aesthetic Quality | -2.10 | -1.92 |
| Imaging Quality | -3.26 | -2.06 |
| Object Class | <span style="color:#c00000"><strong>-7.08</strong></span> | <span style="color:#c00000"><strong>-12.08</strong></span> |
| Multiple Objects | 0.00 | <span style="color:#c00000"><strong>-16.25</strong></span> |
| Spatial Relationship | <span style="color:#c00000"><strong>-9.03</strong></span> | <span style="color:#c00000"><strong>-15.41</strong></span> |
| Dynamic Degree | -6.67 | 0.00 |
| Quality | <span style="color:#c00000"><strong>-3.88</strong></span> | <span style="color:#c00000"><strong>-3.59</strong></span> |
| Semantic | -0.96 | -1.23 |
| Total | <span style="color:#c00000"><strong>-3.30</strong></span> | <span style="color:#c00000"><strong>-3.12</strong></span> |

SLA 对 Multiple Objects 的保持明显好于 Hybrid，但仍在 Spatial Relationship、Object Class、Dynamic Degree 和主体/背景一致性上退化。Hybrid 的聚合分数略高，但多对象、空间关系和对象类别损失更严重；Human Action、Scene 和 Dynamic Degree 等维度抵消了部分聚合损失。因此不能仅凭 Total 认为 Hybrid 在复杂对象与空间结构上优于 SLA。

### 10.4 后训练恢复量

| 匹配方法 | 基础 sparse Total | 训练后 sparse Total | 直接提升 | dense 归一化差距恢复 |
| --- | ---: | ---: | ---: | ---: |
| SLA | 77.79 | 79.45 | +1.66 | 约 34.6% |
| Hybrid | 77.75 | 79.71 | +1.96 | 约 38.6% |

两种 `lora_plus_linear` 训练都产生了明确恢复，但只补回约三分之一的基础权重稀疏缺口。200 step 对当前约 90% 有效稀疏率仍不充分。

### 10.5 路由专用化与交互效应

| 固定训练权重 | SLA 路由 Total | Hybrid 路由 Total | Hybrid - SLA |
| --- | ---: | ---: | ---: |
| SLA-trained | **79.45** | 78.70 | **-0.75** |
| Hybrid-trained | 78.58 | **79.71** | **+1.12** |

在 SLA-trained 权重上，匹配的 SLA 路由高 0.75 分；在 Hybrid-trained 权重上，匹配的 Hybrid 路由高 1.12 分。两个路由差异之差约为 1.88 分，说明 LoRA 与 `sla_linear` 对训练时使用的路由产生了明显专用化。当前证据不支持训练一种方法后在推理时无损切换另一种路由；部署应优先采用训练与推理匹配的组合。

### 10.6 当前结论与下一步

当前最稳健的结论是：

> 200-step SLA 与 Hybrid 后训练均能恢复约三分之一的稀疏质量缺口，并形成明显的训练路由专用化；匹配部署的 Total 分别为 79.45 和 79.71，但相对各自 dense 仍低 3.30 和 3.12 分。基础权重 HSA+CAG 的 82.80 仍是当前质量最稳健的稀疏方案。

下一步按优先级执行：

1. 审计六组 manifest、resolved YAML、checkpoint SHA 和 SP/DP，确认 `2×3` 消融严格可比。
2. 对匹配组合执行 `0.85/0.88/0.90` 稀疏预算消融，重点检查 Object Class、Multiple Objects、Spatial Relationship 和一致性。
3. 在同 checkpoint 下重测 DiT-only，建立质量与加速的 Pareto 曲线。
4. 若降低稀疏率仍无法恢复关键维度，再延长至 500/1000 step。
5. 候选相对同权重 dense 的 Total 差距降到约 1 分后，再进入统一 20% 或 Full VBench。5% 与 20% 是独立抽样，不能合并为同一统计样本。

## 11. 科研有效性与限制

- VBench Standard 5% 只有 43 条提示词，虽然覆盖 16 个维度，但不等同于完整 VBench。
- 每组由 5 个 seed 生成 215 个视频，但当前输入只有每维聚合值，没有逐视频 evaluator 分数，因此无法计算方差、置信区间或配对显著性检验。
- 不同维度的有效提示词数量不同，维度分数不能按同等样本量理解。
- 本报告中的小差值，特别是 HSA Total 的 -0.03 和 SLA/Hybrid 间的 -0.04，不能过度解释。
- 训练后 `2×3` 消融虽使用既定测试设计，但报告尚未直接读取服务器 manifest、checkpoint SHA 和 SP/DP；本地仓库不包含服务器 run 产物。如果并行布局不同，跨权重差值不能视为严格配对结果。
- 基础 SLA/Hybrid 约 5 分的 Total 下降已由训练后对照补充；200-step 训练将匹配路由差距缩小到 3.30/3.12 分，但仍须用更大子集复验。
- 本报告不包含人工主观评测，不能排除自动指标与感知质量不一致。

## 附录 A：关键原始 run 与聚合数据

| run_id | 方法 | backend | Quality | Semantic | Total |
| --- | --- | --- | ---: | ---: | ---: |
| `longlive2-base-5pct-longlive2_standard_5pct-dense` | dense | dense | 85.54 | 71.99 | **82.83** |
| `longlive2-base-5pct-longlive2_standard_5pct-hsa_cag` | hsa_cag | mindiesd | 85.43 | 72.26 | **82.80** |
| `longlive2-base-5pct-longlive2_standard_5pct-sla_cag` | sla_cag | mindiesd | 80.63 | 66.43 | 77.79 |
| `longlive2-base-5pct-longlive2_standard_5pct-hsa_sla_cag` | hsa_sla_cag | mindiesd | 80.49 | 66.79 | 77.75 |
| `longlive2-hybrid-200step-dense-5pct` | dense | dense | 85.81 | 70.88 | **82.82** |
| `longlive2-hybrid-200step-sla_cag-5pct` | sla_cag | mindiesd | 81.06 | 68.66 | 78.58 |
| `longlive2-hybrid-200step-hsa_sla_cag-5pct` | hsa_sla_cag | mindiesd | 82.22 | 69.65 | 79.71 |
| `longlive2-sla-200step-dense-5pct` | dense | dense | 85.61 | 71.29 | **82.75** |
| `longlive2-sla-200step-sla_cag-5pct` | sla_cag | mindiesd | 81.73 | 70.33 | 79.45 |
| `longlive2-sla-200step-hsa_sla_cag-5pct` | hsa_sla_cag | mindiesd | 81.14 | 68.93 | 78.70 |

第 5 节保留基础权重全部 16 个维度，第 6 节给出基础权重派生差值，第 10 节保留训练后全部 16 个维度及关键派生计算。完整 AISBench JSON、官方 summary 和逐 run manifest 应继续以 `runs/vbench/<run-id>/` 产物作为权威实验记录。

## 附录 B：训练后官方聚合原始精度

| run_id | Quality | Semantic | Total |
| --- | ---: | ---: | ---: |
| `longlive2-hybrid-200step-dense-5pct` | 85.81054024372492 | 70.88211757495851 | 82.82485570997163 |
| `longlive2-hybrid-200step-sla_cag-5pct` | 81.06322976636243 | 68.65958171319687 | 78.58250015572932 |
| `longlive2-hybrid-200step-hsa_sla_cag-5pct` | 82.21941098663825 | 69.65160623564537 | 79.70585003643967 |
| `longlive2-sla-200step-dense-5pct` | 85.61397241728953 | 71.28535827975925 | 82.74824958978348 |
| `longlive2-sla-200step-sla_cag-5pct` | 81.73104081567811 | 70.32772280716797 | 79.45037721397608 |
| `longlive2-sla-200step-hsa_sla_cag-5pct` | 81.13951264453677 | 68.92637219896118 | 78.69688455542165 |
