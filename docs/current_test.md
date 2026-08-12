# 当前测试清单

本文件只记录当前提交需要在昇腾服务器执行的临时测试。下一轮测试任务必须覆盖重写本文件；长期稳定命令应迁移到[环境安装与测试](setup_and_validation.md)。

## 测试目标

使用同一个 LongLive2.0 基础 checkpoint，对 `dense`、`hsa_cag`、`sla_cag` 和 `hsa_sla_cag` 做统一性能对比：

- 时长：5 秒、32 秒和 64 秒；
- 并行：SP1 单卡和 SP4 四卡；
- 模式：DiT-only、同步 VAE 完整视频、独立设备异步 VAE 完整视频；
- 异步拓扑：SP1+VAE 共 2 卡，SP4+VAE 共 5 卡。

基础 checkpoint 不含后续新增的 `sla_linear` 参数。加载器只为缺失的 `sla_linear` 补零，并对其他参数保持严格加载。因此本轮 SLA 与混合方法用于测量未训练路由和物理稀疏算子的速度，不代表训练后线性补偿的质量或速度。

## 服务器准备

```bash
git pull origin feat/unified-sparse-attention

conda activate /mnt/share/r50063443/conda_envs/longlive
source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
cd /mnt/share/r50063443/LongLive-oneday

export LONGLIVE_GENERATOR_CKPT=/mnt/share/weight/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt
export SUITE_ID="base-weight-speed-$(date +%Y%m%d-%H%M%S)"
export METHODS=dense,hsa_cag,sla_cag,hsa_sla_cag
export DURATIONS=5s,32s,64s
export TASK=benchmark
```

同一终端内完成所有阶段，保证 `SUITE_ID` 不变。若重新登录，先恢复原来的 `SUITE_ID`；不要生成新 ID，否则结果不会进入同一汇总。

## 1. 展开检查

```bash
DRY_RUN=1 SP_SIZES=1,4 MODES=dit_only,sync_vae,async_vae \
PERF_DEVICES=4,5,6,7,9 \
bash scripts/evaluation/run_performance_matrix.sh
```

预期共展开 72 个 case。SP1 的非异步 case 使用 1 张卡，SP4 使用 4 张卡；完整运行采用下方分阶段命令，以固定异步 VAE 为 NPU 9。

## 2. DiT-only

先运行 SP1。若 5 秒即 OOM，保留日志并停止 SP1 的更长时长，不把失败目录删除后伪装成未测试。

```bash
SP_SIZES=1 MODES=dit_only PERF_DEVICES=4 \
bash scripts/evaluation/run_performance_matrix.sh

SP_SIZES=4 MODES=dit_only PERF_DEVICES=4,5,6,7 \
bash scripts/evaluation/run_performance_matrix.sh
```

## 3. 同步 VAE 完整视频

同步模式由每个 SP 组的 leader 在 DiT 设备上解码。SP1 同时驻留完整 DiT 与 VAE，最容易 OOM，应先观察 5 秒 case。

```bash
SP_SIZES=1 MODES=sync_vae PERF_DEVICES=4 \
bash scripts/evaluation/run_performance_matrix.sh

SP_SIZES=4 MODES=sync_vae PERF_DEVICES=4,5,6,7 \
bash scripts/evaluation/run_performance_matrix.sh
```

## 4. 异步 VAE 完整视频

NPU 9 固定为专用 VAE 卡。矩阵按设备列表顺序取前 `SP_SIZE` 张卡作为 DiT worker，最后一张逻辑设备自动解析为 `vae_device=npu:<SP_SIZE>`。

```bash
SP_SIZES=1 MODES=async_vae PERF_DEVICES=4,9 \
bash scripts/evaluation/run_performance_matrix.sh

SP_SIZES=4 MODES=async_vae PERF_DEVICES=4,5,6,7,9 \
bash scripts/evaluation/run_performance_matrix.sh
```

## 5. 结果位置与检查项

单 case 结果：

```text
runs/performance/${SUITE_ID}-<method>-<duration>-<mode>-sp<sp_size>/
logs/performance/${SUITE_ID}-<method>-<duration>-<mode>-sp<sp_size>/
```

统一汇总：

```text
runs/suites/${SUITE_ID}/performance/results.csv
runs/suites/${SUITE_ID}/performance/results.json
```

每个正式 case 默认包含 1 次预热和 3 次有效测量。重点检查：

- `generation_seconds_p50` 和相同 SP、相同时长、相同 VAE 模式下的 `dense_speedup`；
- DiT-only 的峰值显存和 SP1 是否 OOM；
- 异步模式的 `ar_loop_seconds`、`vae_decode_seconds`、`vae_drain_seconds`、`vae_overlap_seconds` 和队列峰值；
- 64 秒是否出现 KV cache 增长、VAE 队列阻塞、OOM 或稀疏率异常。

`RESUME_SUITE=1` 为默认值：已有 `summary.json` 的 case 会跳过。不完整 run 目录不会被覆盖，需保留日志并先分析失败原因。
