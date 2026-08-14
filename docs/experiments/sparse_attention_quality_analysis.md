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

第二阶段使用完成 200 step `lora_plus_linear` 训练并导出的同一份 HSA+SLA+CAG 权重，分别运行 `dense` 与 `hsa_sla_cag`。该消融用于区分 Generator 主干 LoRA 后训练造成的变化与运行时稀疏注意力造成的额外变化。

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

训练后新增两组沿用相同的 VBench preset、提示词、seed、视频长度、采样步数和 `SP2 × DP8` 布局。按实验设计，两组使用同一个导出 checkpoint：

| 训练后 run | 注意力 | 评测时间 |
| --- | --- | --- |
| `longlive2-hybrid-200step-dense-5pct` | dense | `20260814_001749` |
| `longlive2-hybrid-200step-hsa_sla_cag-5pct` | HSA+SLA+CAG | `20260813_172004` |

严格配对结论仍要求两个 run 的 manifest、resolved YAML 和 checkpoint SHA 一致。本文依据既定测试矩阵和用户回传结果进行分析；原始运行目录是最终审计依据。

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

基础权重阶段，HSA 是唯一同时表现出明确 DiT 加速和近似保持聚合质量的方法。第 10 节进一步表明，200-step 后训练能够恢复 Hybrid 的部分质量，但尚未消除运行时稀疏代价。

## 9. 机制解释与证据边界

可由当前数据支持的表述：

> 在 LongLive2.0 基础权重、零初始化 `sla_linear`、当前默认稀疏预算和 VBench Standard 5% 条件下，SLA+CAG 与 HSA+SLA+CAG 的 Total 分别比 dense 低 5.04 和 5.08 分，退化集中于对象类别、多对象、空间关系和一致性维度。

基础权重数据单独不能支持以下更强结论：

- “SLA 算法必然损失约 5 分”：200-step 训练结果已证明损失可以部分恢复，但不能外推到其他步数、稀疏率或数据规模。
- “HSA 与 dense 质量等价”：没有逐视频分数和置信区间，不能做等价性检验。
- “Hybrid 不如 SLA”：两者 Total 只差 0.04 分，远小于该子集可可靠解释的尺度。
- “Human Action 得到提升”：该指标接近满分，可能是小样本和天花板效应。

## 10. 200-step 后训练消融

### 10.1 官方聚合分数

| 200-step 权重运行方式 | Quality | Semantic | Total |
| --- | ---: | ---: | ---: |
| dense | **85.81** | **70.88** | **82.82** |
| HSA+SLA+CAG | 82.22 | 69.65 | 79.71 |
| Hybrid - dense | <span style="color:#c00000"><strong>-3.59</strong></span> | <span style="color:#c00000"><strong>-1.23</strong></span> | <span style="color:#c00000"><strong>-3.12</strong></span> |

训练后 dense 的 Total 为 82.82，基础权重 dense 为 82.83，观测差异仅 -0.01 分。这说明 200-step 主干 LoRA 后训练没有造成可见量级的总体质量下降，但单项指标发生了重新分布。相同训练权重切换到 Hybrid 后，Total 下降 3.12 分，因此当前剩余退化主要来自运行时稀疏路径，而不是主干 LoRA 本身。

由官方结果可核验：

```text
Total = 0.8 × Quality + 0.2 × Semantic
```

训练后 Hybrid 的 Total 差距中，Quality 贡献约 2.87 分，约占总差距的 92%；Semantic 贡献约 0.25 分。因此后续恢复重点应放在视觉质量、主体/背景一致性和复杂对象结构，而不是只追求 Semantic 聚合分数。

### 10.2 全部 16 个维度及运行时稀疏差值

正值表示训练后 Hybrid 高于同 checkpoint dense，负值表示低于 dense。

| VBench 维度 | 训练后 dense | 训练后 Hybrid | Hybrid - dense |
| --- | ---: | ---: | ---: |
| Aesthetic Quality | **60.30** | 58.38 | -1.92 |
| Appearance Style | **19.34** | 18.83 | -0.51 |
| Background Consistency | **95.29** | 88.41 | <span style="color:#c00000"><strong>-6.88</strong></span> |
| Color | 82.50 | **84.46** | +1.96 |
| Dynamic Degree | **93.33** | **93.33** | 0.00 |
| Human Action | 92.00 | **100.00** | +8.00 |
| Imaging Quality | **65.65** | 63.59 | -2.06 |
| Motion Smoothness | **98.63** | 98.55 | -0.08 |
| Multiple Objects | **61.25** | 45.00 | <span style="color:#c00000"><strong>-16.25</strong></span> |
| Object Class | **80.42** | 68.33 | <span style="color:#c00000"><strong>-12.08</strong></span> |
| Overall Consistency | 22.96 | **23.37** | +0.41 |
| Scene | 22.81 | **41.88** | +19.06 |
| Spatial Relationship | **92.75** | 77.34 | <span style="color:#c00000"><strong>-15.41</strong></span> |
| Subject Consistency | **96.64** | 90.46 | <span style="color:#c00000"><strong>-6.18</strong></span> |
| Temporal Flickering | **99.74** | 98.81 | -0.93 |
| Temporal Style | 25.68 | **25.75** | +0.07 |

主要退化仍集中在 Multiple Objects、Spatial Relationship、Object Class、Background Consistency 和 Subject Consistency，与基础权重 Hybrid 的退化模式一致。200-step 训练降低了部分误差，但没有改变主要薄弱维度。

Scene 的 +19.06 和 Human Action 的 +8.00 不应直接解释为 Hybrid 改善模型能力。训练后 dense 的 Scene 相比基础 dense 从 39.06 降至 22.81，而训练后 Hybrid 为 41.88；Human Action 又接近满分上界。这些现象更可能同时受到小样本、指标离散性、天花板效应和注意力路径差异影响，需要更大子集和逐视频结果验证。

### 10.3 三组关键差分

| 差分 | Quality | Semantic | Total | 解释 |
| --- | ---: | ---: | ---: | --- |
| 训练后 dense - 基础 dense | +0.27 | -1.11 | **-0.01** | 主干 LoRA 的总体影响近似中性 |
| 训练后 Hybrid - 训练后 dense | <span style="color:#c00000"><strong>-3.59</strong></span> | -1.23 | <span style="color:#c00000"><strong>-3.12</strong></span> | 同权重下的运行时稀疏代价 |
| 训练后 Hybrid - 基础 Hybrid | +1.73 | +2.86 | **+1.96** | 200-step 后训练带来的恢复 |

基础权重的 Hybrid-dense Total 差距为 -5.08 分；训练后差距缩小到 -3.12 分，即约恢复 1.96 分，补回原差距的约 38.6%。分项恢复并不均衡：

| 聚合项 | 基础权重差距 | 训练后差距 | 约恢复分数 | 约恢复比例 |
| --- | ---: | ---: | ---: | ---: |
| Quality | -5.05 | -3.59 | +1.46 | 28.9% |
| Semantic | -5.20 | -1.23 | +3.97 | 76.3% |
| Total | -5.08 | -3.12 | +1.96 | 38.6% |

这证明 `lora_plus_linear` 训练线路具有补偿能力，但 200 step 尚不足以让约 90% 有效稀疏率的 Hybrid 接近 dense。尤其 Multiple Objects 的同权重差距仍为 -16.25 分，不能仅凭 Total 恢复就认为训练已经充分。

### 10.4 当前结论与下一步

当前最稳健的结论是：

> 在 VBench Standard 5% 条件下，200-step HSA+SLA+CAG 后训练将 Hybrid Total 从 77.75 提升到 79.71；同一训练权重运行 dense 为 82.82，仍存在 3.12 分运行时稀疏代价。剩余损失主要集中在多对象、空间关系、对象类别和主体/背景一致性。

下一步按优先级执行：

1. 核对两个训练后 run 的 checkpoint SHA、manifest 和 resolved YAML，排除 checkpoint 或配置不一致。
2. 检查推理加载日志，确认 30 层共 60 个非零 `sla_linear` 张量被加载，未落入缺失参数零初始化路径。
3. 使用同一训练后权重补测 `hsa_cag` 与 `sla_cag`，区分 HSA 帧预筛选和 SLA block 路由各自的质量代价。
4. 对 `0.85/0.88/0.90` 等稀疏预算做消融，判断约 90% 稀疏率是否超过复杂对象和空间关系的可恢复范围。
5. 在确认加载链无误后扩展训练步数，并在完全相同设置下进行 20% 或 Full VBench 复验。5% 与 20% 是独立抽样，不能合并为同一统计样本。

## 11. 科研有效性与限制

- VBench Standard 5% 只有 43 条提示词，虽然覆盖 16 个维度，但不等同于完整 VBench。
- 每组由 5 个 seed 生成 215 个视频，但当前输入只有每维聚合值，没有逐视频 evaluator 分数，因此无法计算方差、置信区间或配对显著性检验。
- 不同维度的有效提示词数量不同，维度分数不能按同等样本量理解。
- 本报告中的小差值，特别是 HSA Total 的 -0.03 和 SLA/Hybrid 间的 -0.04，不能过度解释。
- 训练后 dense 与 Hybrid 虽使用既定的同 checkpoint 测试设计，但报告尚未直接读取服务器 manifest 和 checkpoint SHA；推送前的本地仓库不包含服务器 run 产物。
- 基础 SLA/Hybrid 约 5 分的 Total 下降已由训练后对照补充；200-step 训练将 Hybrid 差距缩小到 3.12 分，但仍须用更大子集复验。
- 本报告不包含人工主观评测，不能排除自动指标与感知质量不一致。

## 附录 A：关键原始 run 与聚合数据

| run_id | 方法 | backend | Quality | Semantic | Total |
| --- | --- | --- | ---: | ---: | ---: |
| `longlive2-base-5pct-longlive2_standard_5pct-dense` | dense | dense | 85.54 | 71.99 | **82.83** |
| `longlive2-base-5pct-longlive2_standard_5pct-hsa_cag` | hsa_cag | mindiesd | 85.43 | 72.26 | **82.80** |
| `longlive2-base-5pct-longlive2_standard_5pct-sla_cag` | sla_cag | mindiesd | 80.63 | 66.43 | 77.79 |
| `longlive2-base-5pct-longlive2_standard_5pct-hsa_sla_cag` | hsa_sla_cag | mindiesd | 80.49 | 66.79 | 77.75 |
| `longlive2-hybrid-200step-dense-5pct` | dense | dense | 85.81 | 70.88 | **82.82** |
| `longlive2-hybrid-200step-hsa_sla_cag-5pct` | hsa_sla_cag | mindiesd | 82.22 | 69.65 | 79.71 |

第 5 节保留基础权重全部 16 个维度，第 6 节给出基础权重派生差值，第 10 节保留训练后全部 16 个维度及关键派生计算。完整 AISBench JSON、官方 summary 和逐 run manifest 应继续以 `runs/vbench/<run-id>/` 产物作为权威实验记录。

## 附录 B：训练后官方聚合原始精度

| run_id | Quality | Semantic | Total |
| --- | ---: | ---: | ---: |
| `longlive2-hybrid-200step-dense-5pct` | 85.81054024372492 | 70.88211757495851 | 82.82485570997163 |
| `longlive2-hybrid-200step-hsa_sla_cag-5pct` | 82.21941098663825 | 69.65160623564537 | 79.70585003643967 |
