# 稀疏路由与 5s 生成稀疏度分析

本文按当前代码和 `configs/inference/msprof.yaml` 的 `5s` preset，说明 `dense`、`hsa_cag`、`sla_cag`、`hsa_sla_cag` 的候选帧、候选 block 和最终 LUT 稀疏度。本文只统计 sparse softmax 主分支；`sla_cag` 和 `hsa_sla_cag` 还有完整 KV 线性补偿分支，不能把 LUT 稀疏度直接等同于整个 attention 模块的访存稀疏度。

## 1. 5s 基础形状

`5s` preset 使用：

```yaml
model:
  num_frame_per_block: 8
  local_attn_size: 32
generation:
  sink_size: 8
  sampling_steps: 4
  multi_shot_sink: true
presets:
  5s:
    latent_frames: 32
sparsity:
  profiles:
    hsa_cag / sla_cag / hsa_sla_cag:
      sparsity: 0.85
      sparsity_base: 0.95
      block_q: 128
      block_k: 128
```

5s 生成 32 个 latent frames。LongLive2.0 每个 AR chunk 生成 8 帧，因此一共有 4 个 chunk：

| chunk_id | current frames | completed history frames | resident KV frames |
| --- | --- | --- | --- |
| 0 | 0-7 | 无 | 0-7 |
| 1 | 8-15 | 0-7 | 0-15 |
| 2 | 16-23 | 0-15 | 0-23 |
| 3 | 24-31 | 0-23 | 0-31 |

每个 latent frame 对应 `22 x 40 = 880` 个 token；`block_k=128` 时，一个 8 帧 chunk 是 `7040` tokens，对应 `55` 个 KV blocks。

| chunk_id | resident frames | full_kv_blocks |
| --- | ---: | ---: |
| 0 | 8 | 55 |
| 1 | 16 | 110 |
| 2 | 24 | 165 |
| 3 | 32 | 220 |

5s 的总长度正好等于 `local_attn_size=32`，所以 `hsa_history_mode=rolling` 与 `hsa_history_mode=full` 在 5s 中看到的 resident KV 范围相同。`full` 的差异要到 32 帧之后才出现：rolling 只让 HSA 看到 LongLive resident 32 帧窗口，full 则让 HSA frame router 看到从 frame 0 到当前 chunk 前一帧的全部 completed history。

## 2. CAG 稀疏率调度

CAG 不按候选池大小设 K，而是按完整 resident KV block 数设 K。入口是 `calculate_chunk_sparsities`。rolling mode 下 resident 长度受 `local_attn_size=32` 限制；`hsa_history_mode=full` 下，HSA 的 resident 长度就是完整已完成历史加当前 chunk，因此 32 帧之后的 CAG schedule 会按 full resident 长度生成。

```text
chunk_frames = [16, 24, 32]
kv_lengths = min(chunk_frames, local_attn_size)  # rolling
kv_lengths = chunk_frames                       # full-history HSA
alpha_i = 1 / sqrt(chunk_frames_i)
target = sum((1 - sparsity) * kv_length_i)
base = sum((1 - sparsity_base) * kv_length_i)
beta = (target - base) / sum(alpha_i * kv_length_i)
chunk_sparsity_i = clamp(sparsity_base - alpha_i * beta, 0, 0.999)
```

首个 chunk 没有历史，固定 dense。5s 没超过 32 帧，所以 rolling/full 的 CAG schedule 数值相同：

| chunk_id | CAG sparsity | full_kv_blocks | requested_k | LUT 稀疏度 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.000000 | 55 | 55 | 0.00% |
| 1 | 0.826338 | 110 | 19 | 82.73% |
| 2 | 0.849031 | 165 | 24 | 85.45% |
| 3 | 0.862558 | 220 | 30 | 86.36% |

公式是：

```text
requested_k = floor((1 - CAG sparsity) * full_kv_blocks + 1e-9)
final_k = min(requested_k, candidate_capacity)
LUT 稀疏度 = 1 - final_k / full_kv_blocks
```

`sparsity_base=0.95` 是尾部基准，倾向于让后期 chunk 更稀疏；`sparsity=0.85` 是整段生成的平均目标。若想显著提高 chunk 1/2 的 `final_k`，应优先降低 `sparsity`，因为它直接提高全局目标预算；降低 `sparsity_base` 会改变调度形状，让后期基准也变密，但不是专门解决前期 K 过小的最直接旋钮。

## 3. sink、shot sink 与 near

LongLive KV manager 有三类和 HSA frame routing 相关的帧：

| 名称 | 层级 | 当前语义 |
| --- | --- | --- |
| global sink | KV cache | 视频开头 `sink_size=8` 帧不被 rolling eviction |
| shot sink | KV cache | scene cut 后 pin 的 chunk，后续 chunk 中不被 eviction |
| near history | HSA frame router | 从最新 completed history 向前取 `keep_near_history_frames=4` 帧 |

`protect_longlive_sink_frames=true` 时，HSA 的 frame-stage 会把当前 KV layout 中仍属于 history 的 global sink 和 shot sink 加入 protected history frames。`keep_near_history_frames=4` 在 rolling 和 full mode 都保留：它从最新 history 向前取 4 帧，跳过 protected frames。

普通 5s 性能 prompt 通常没有 scene cut，因此只有 global sink：

| chunk_id | global sink history | shot sink history |
| --- | --- | --- |
| 0 | 无，0-7 仍是 current | 无 |
| 1 | 0-7 | 无 |
| 2 | 0-7 | 无 |
| 3 | 0-7 | 无 |

如果某个 chunk 触发 scene cut，pipeline 会在该 chunk 完成 clean recache 后 pin 当前 chunk，影响之后的 chunk；它不会回头改变刚完成 chunk 的路由。

## 4. dense

`dense` 不进入 sparse 分支，所有 resident KV blocks 都参与 attention：

| chunk_id | candidate frames | candidate_blocks | final_k | LUT 稀疏度 |
| --- | --- | ---: | ---: | ---: |
| 0 | 0-7 | 55 | 55 | 0.00% |
| 1 | 0-15 | 110 | 110 | 0.00% |
| 2 | 0-23 | 165 | 165 | 0.00% |
| 3 | 0-31 | 220 | 220 | 0.00% |

## 5. HSA+CAG

### 5.1 路由算法

HSA 保持 Light Forcing 原始两级层次路由：

```text
q_tilde[r] = mean_pool(Q block r)
k_hat[f] = mean_pool(all K blocks in historical frame f)
frame_score[r, f] = dot(q_tilde[r], k_hat[f])

candidate_frames[r] =
  protected LongLive sink history frames
  UNION near history frames
  UNION dynamic Top-K historical frames for query block r
  UNION all current chunk frames
```

然后第二级只在 `candidate_frames[r]` 覆盖的 KV blocks 内做 block score，再由 CAG 的 `requested_k` 决定最终 LUT K。current 8 帧始终进入 candidate frame pool，但 `dense_current_blocks=false`，所以 current blocks 不保证进入最终 LUT。

`hsa_history_mode` 只改变 HSA 能看到的历史范围：

| mode | HSA frame search 范围 | KV eviction 行为 |
| --- | --- | --- |
| `rolling` | LongLive resident rolling 32-frame cache 中的 completed history | 保持原 LongLive2.0 rolling |
| `full` | frame 0 到当前 chunk 前一帧的全部 completed history | 为 HSA 额外维护 on-device full-history K/V，不驱逐已完成历史；CAG schedule 使用 full resident 长度 |

`full` 不修改 Q/K token compression、不修改 per-query-block frame Top-K、不修改 block routing、不修改 CAG。它只让 HSA 的 historical frame retrieval 不再被 rolling KV cache 截断。

### 5.2 5s 无 scene cut 候选帧

5s 没超过 32 帧，因此 rolling/full 的表相同：

| chunk_id | protected history | near history | dynamic history | current | selected_history_count |
| --- | --- | --- | --- | --- | ---: |
| 0 | 无 | 无 | 无 | 0-7 | 0 |
| 1 | 0-7 | 无 | 无 | 8-15 | 8 |
| 2 | 0-7 | 12-15 | 8-11 | 16-23 | 16 |
| 3 | 0-7 | 20-23 | 从 8-19 中按 `frame_score[r,f]` 选 4 帧 | 24-31 | 16 |

chunk 1 和 chunk 2 的 history 被 `protected + near + dynamic` 覆盖，所以没有实际 frame pruning。chunk 3 开始才真正裁 history：24 帧 history 中只选 16 帧，其中 12 帧由 sink/near 确定，4 帧由每个 query block 和 head 的 QK frame score 决定。

### 5.3 5s block 数

HSA 的候选容量按候选帧数量换算：

```text
candidate_capacity =
  ceil((selected_history_count + current_frames) * 880 / 128)
```

5s 下 `requested_k` 都小于候选容量，所以最终 K 完全由 CAG 决定：

| chunk_id | selected frames | candidate_capacity | full_kv_blocks | requested_k | final_k | LUT 稀疏度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 current | 55 | 55 | 55 | 55 | 0.00% |
| 1 | 8 history + 8 current | 110 | 110 | 19 | 19 | 82.73% |
| 2 | 16 history + 8 current | 165 | 165 | 24 | 24 | 85.45% |
| 3 | 16 history + 8 current | 165 | 220 | 30 | 30 | 86.36% |

日志中的 `candidate_blocks` 是 `eligible` mask 的真实 block 数。由于 880 tokens/frame 与 128 tokens/block 不整除，非连续帧可能共享边界 block，因此实际 `candidate_blocks` 可能比表中容量略高。`final_k` 只在候选池小于 `requested_k` 时才被候选池截断。

### 5.4 超过 32 帧后的 full-history 差异

以当前 chunk 从 frame 96 开始为例：

| mode | HSA history search 范围 | protected sink | near history | dynamic history |
| --- | --- | --- | --- | --- |
| `rolling` | 约 64-95，加上 LongLive 组装进 resident KV 的 sink | global/shot sink 中仍被 KV layout 标记的帧 | 92-95 | 只能从 resident history 剩余帧选 |
| `full` | 0-95 全部 completed history | global sink 0-7 和已有 shot sink | 92-95 | 可从 0-91 中按分数选，允许选 frame 17 |

因此 full mode 的 debug 合法输出应允许：

```text
[hsa-frame-route] mode=full ... current_frames=96-103 ...
query_block=12 ... selected_history_frames=0,1,2,3,4,5,6,7,17,92,93,94,95 ...
contains_pre_window_frame=true
```

这正是本次改造要验证的性质：HSA 原论文要求的 Q-based historical retrieval 可以访问完整已完成历史，而不是只能访问 LongLive rolling 32-frame cache。

## 6. SLA+CAG

SLA 不做 frame-stage pruning，候选池始终是完整 resident KV。纯 SLA 当前仍保留 block-stage hard anchors：

```text
hard_keep_sink_frames = 1
hard_keep_recent_frames = 1
```

这两个 anchor 会强制 resident KV 第 1 帧和最新 1 帧覆盖的 blocks 进入最终 LUT，并占用 CAG K。每帧 880 tokens，通常覆盖约 7 个 128-token blocks，所以两个 anchor 合计约 14 blocks。

5s 表如下：

| chunk_id | candidate frames | candidate_blocks | hard anchor blocks | requested_k | final_k | LUT 稀疏度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0-7 | 55 | 不进入 sparse 分支 | 55 | 55 | 0.00% |
| 1 | 0-15 | 110 | 约 14 | 19 | 19 | 82.73% |
| 2 | 0-23 | 165 | 约 14 | 24 | 24 | 85.45% |
| 3 | 0-31 | 220 | 约 14 | 30 | 30 | 86.36% |

如果某个配置让 `requested_k < fixed_anchor_blocks`，纯 SLA 会把 `final_k` 提升到 fixed anchor block 数；当前 5s 默认配置下 `19/24/30` 都大于约 14，所以 hard anchor 不改变 K，只改变哪些 blocks 必进。

## 7. HSA+SLA+CAG

Hybrid 的 frame-stage 是 HSA，block-stage 是 SLA Smooth-K/QK 打分，但只在 HSA 候选帧覆盖的 blocks 内 Top-K。Hybrid 当前没有 `hard_keep_sink_frames` 和 `hard_keep_recent_frames`，也就是没有 block-stage 强制 anchor。

Hybrid 对 LongLive sink 的 frame-stage protected 上限是：

```text
max_global_sink_frames = 2
max_shot_sink_frames = 2
```

5s 无 scene cut 时，默认没有 shot sink，因此 protected 只有 global sink 前 2 帧：

| chunk_id | protected history | near history | dynamic history | current | selected_history_count |
| --- | --- | --- | --- | --- | ---: |
| 0 | 无 | 无 | 无 | 0-7 | 0 |
| 1 | 0-1 | 4-7 | 2-3 | 8-15 | 8 |
| 2 | 0-1 | 12-15 | 从 2-11 中按分数选 4 帧 | 16-23 | 10 |
| 3 | 0-1 | 20-23 | 从 2-19 中按分数选 4 帧 | 24-31 | 10 |

对应 block 数：

| chunk_id | selected frames | candidate_capacity | full_kv_blocks | requested_k | final_k | LUT 稀疏度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 current | 55 | 55 | 55 | 55 | 0.00% |
| 1 | 8 history + 8 current | 110 | 110 | 19 | 19 | 82.73% |
| 2 | 10 history + 8 current | 124 | 165 | 24 | 24 | 85.45% |
| 3 | 10 history + 8 current | 124 | 220 | 30 | 30 | 86.36% |

Hybrid 的 candidate pool 比 HSA 更小，但 5s 默认下 `requested_k` 仍小于候选容量，所以最终 K 仍是 `55 -> 19 -> 24 -> 30`。

## 8. 有 shot sink 时

假设 chunk 1 触发 scene cut，chunk 1 完成后 frame 8-15 被 pin 为 shot sink。它会影响 chunk 2/3：

| 方法 | chunk 2 protected history | chunk 3 protected history |
| --- | --- | --- |
| HSA+CAG | global 0-7 + shot 8-15 | global 0-7 + shot 8-15 |
| HSA+SLA+CAG | global 0-1 + shot 8-9 | global 0-1 + shot 8-9 |
| SLA+CAG | 不使用 frame-stage protected | 不使用 frame-stage protected |

HSA 在 chunk 3 中会变成：

```text
selected_history_count = protected 16 + near 4 + dynamic 4 = 24
candidate_capacity = ceil((24 + 8) * 880 / 128) = 220
final_k = 30
```

Hybrid 在 chunk 3 中会变成：

```text
selected_history_count = protected 4 + near 4 + dynamic 4 = 12
candidate_capacity = ceil((12 + 8) * 880 / 128) = 138
final_k = 30
```

shot sink 会扩大 HSA/Hybrid 的候选池，但不会直接增加最终 K；最终 K 仍由 CAG 决定，除非候选容量小于 `requested_k`。

## 9. 结论口径

报告稀疏度时建议分三层：

| 口径 | 公式 | 含义 |
| --- | --- | --- |
| CAG sparsity | `sparsity_list[chunk_id]` | CAG 给该 chunk 的目标稀疏率 |
| 候选池裁剪率 | `1 - candidate_blocks / full_kv_blocks` | HSA/Hybrid frame-stage 裁掉多少 resident KV blocks；SLA 为 0 |
| 最终 LUT 稀疏率 | `1 - final_k / full_kv_blocks` | sparse softmax 主分支实际读取多少 KV blocks |

5s 无 scene cut 时，三种稀疏方法的最终 LUT K 相同，都是 `55 -> 19 -> 24 -> 30`。差异在候选来源：

| 方法 | frame-stage 候选 | block-stage 强制保留 | 线性补偿 |
| --- | --- | --- | --- |
| HSA+CAG rolling | LongLive resident history 中的 global/shot sink + near4 + dynamic4 + current8 | 无 | 无 |
| HSA+CAG full | 全部 completed history 中的 global/shot sink + near4 + dynamic4 + current8 | 无 | 无 |
| SLA+CAG | 不裁 frame，完整 resident KV 全部候选 | 第 1 帧和最新 1 帧 hard anchor | 完整 KV |
| HSA+SLA+CAG | global 前 2 + shot 前 2 + near4 + dynamic4 + current8 | 无 | 完整 KV |

如果目标是提高 chunk 1/2 的实际 block 数，优先降低 `sparsity`。`sparsity_base` 主要控制 CAG 后段调度基准，降低它会整体改变调度曲线；它能提高前期 K，但同时也会让后期更密，不是只针对 chunk 1/2 的局部修正。
