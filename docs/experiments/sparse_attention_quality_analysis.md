# LongLive2.0 基础权重稀疏注意力质量分析

## 1. 报告范围

本文分析 LongLive2.0-5B 基础权重在四种注意力路径下的 VBench Standard 5% 质量结果：

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

因此本文衡量的是“在未进行对应稀疏后训练适配时，直接替换运行时注意力路径”的质量变化，不代表训练后 SLA 或 Hybrid 的最终质量。

| 报告元数据 | 值 |
| --- | --- |
| 数据日期 | 2026-08-13 收到汇总结果 |
| 仓库分支 | `feat/unified-sparse-attention` |
| 整理时提交 | `67145af` |
| 数据类型 | AISBench VBench 聚合结果 |
| 报告状态 | 基础权重 5% 阶段性分析 |

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

在训练后结果产生之前，HSA 是唯一同时表现出明确 DiT 加速和近似保持聚合质量的方法。SLA/Hybrid 的性能潜力更高，但必须依靠稀疏后训练恢复对象和空间语义。

## 9. 机制解释与证据边界

可由当前数据支持的表述：

> 在 LongLive2.0 基础权重、零初始化 `sla_linear`、当前默认稀疏预算和 VBench Standard 5% 条件下，SLA+CAG 与 HSA+SLA+CAG 的 Total 分别比 dense 低 5.04 和 5.08 分，退化集中于对象类别、多对象、空间关系和一致性维度。

当前数据不能支持以下更强结论：

- “SLA 算法必然损失约 5 分”：训练后的非零补偿层和 LoRA 尚未纳入本组数据。
- “HSA 与 dense 质量等价”：没有逐视频分数和置信区间，不能做等价性检验。
- “Hybrid 不如 SLA”：两者 Total 只差 0.04 分，远小于该子集可可靠解释的尺度。
- “Human Action 得到提升”：该指标接近满分，可能是小样本和天花板效应。

## 10. 后续实验判定框架

完成 200 step HSA+SLA+CAG 训练后，至少需要增加：

```text
训练后完整权重 + dense attention
训练后完整权重 + hsa_sla_cag attention
```

建议按下列差分解释：

| 对比 | 研究问题 |
| --- | --- |
| 训练后 dense - 基础 dense | 主干 LoRA 后训练本身是否导致分布漂移 |
| 训练后 Hybrid - 训练后 dense | 同一训练权重下的运行时稀疏代价 |
| 训练后 Hybrid - 基础 Hybrid | 200 step 适配对稀疏误差的补偿量 |

首要恢复指标是 Object Class、Multiple Objects、Spatial Relationship、Background Consistency 和 Subject Consistency。仅恢复 Total 而这些关键维度仍大幅下降，不能认为补偿已充分。

5% 回归通过后，应在完全相同提示词版本、seed、分辨率和采样设置下进行 20% 或 Full 复验。5% 与 20% 是独立抽样，不能把两者当作嵌套样本直接合并。

## 11. 科研有效性与限制

- VBench Standard 5% 只有 43 条提示词，虽然覆盖 16 个维度，但不等同于完整 VBench。
- 每组由 5 个 seed 生成 215 个视频，但当前输入只有每维聚合值，没有逐视频 evaluator 分数，因此无法计算方差、置信区间或配对显著性检验。
- 不同维度的有效提示词数量不同，维度分数不能按同等样本量理解。
- 本报告中的小差值，特别是 HSA Total 的 -0.03 和 SLA/Hybrid 间的 -0.04，不能过度解释。
- SLA/Hybrid 约 5 分的 Total 下降和多个维度超过 7–24 分的下降具有较大效应量，但仍须用训练后对照和更大子集复验。
- 本报告不包含人工主观评测，不能排除自动指标与感知质量不一致。

## 附录 A：关键原始 run 与聚合数据

| run_id | 方法 | backend | Quality | Semantic | Total |
| --- | --- | --- | ---: | ---: | ---: |
| `longlive2-base-5pct-longlive2_standard_5pct-dense` | dense | dense | 85.54 | 71.99 | **82.83** |
| `longlive2-base-5pct-longlive2_standard_5pct-hsa_cag` | hsa_cag | mindiesd | 85.43 | 72.26 | **82.80** |
| `longlive2-base-5pct-longlive2_standard_5pct-sla_cag` | sla_cag | mindiesd | 80.63 | 66.43 | 77.79 |
| `longlive2-base-5pct-longlive2_standard_5pct-hsa_sla_cag` | hsa_sla_cag | mindiesd | 80.49 | 66.79 | 77.75 |

第 5 节保留全部 16 个维度的关键原始分数，第 6 节给出全部派生差值。完整 AISBench JSON、官方 summary 和逐 run manifest 应继续以 `runs/vbench/<run-id>/` 产物作为权威实验记录。
