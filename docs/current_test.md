# 当前测试清单

本文件只记录当前提交需要在昇腾服务器执行的临时测试。下一轮测试任务必须覆盖重写本文件；长期稳定命令应迁移到[环境安装与测试](setup_and_validation.md)。

## 测试目标

1. 验证 SLA+CAG 与 HSA+SLA+CAG 在相同 `0.90/0.93` CAG 预算下的实际选块数和 MindIE-SD 完整路径耗时。
2. 验证 MindIE-SD 稀疏 softmax 对未选 KV blocks 的扰动不敏感，证明算子只消费 LUT 选中 blocks。
3. 验证新的 `runs/`、`logs/` 落盘路径。

本轮只验证运行时预算、物理稀疏和路径迁移。若 SLA checkpoint 原先按 `0.95/0.97` 训练，本轮结果不能替代使用 `0.90/0.93` 重新训练或继续微调后的正式 VBench。

## 服务器准备

```bash
git pull origin feat/unified-sparse-attention

conda activate /mnt/share/r50063443/conda_envs/longlive
source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh
cd /mnt/share/r50063443/LongLive-oneday

test_id="sla-budget-physical-sparse-$(date +%Y%m%d-%H%M%S)"
mkdir -p "logs/tests/${test_id}"
```

## 1. 物理稀疏审计

```bash
ASCEND_RT_VISIBLE_DEVICES=15 \
python tests/npu/mindiesd_sla_kernel_smoke.py \
  --device npu:0 --dtype bf16 --audit-unselected-kv \
  | tee "logs/tests/${test_id}/mindiesd-physical-sparse.txt"
```

期望出现 `unselected_kv_audit=passed`。测试会大幅扰动未选 K/V；稀疏输出应保持不变，而 dense 输出必须变化。

## 2. 相同预算微基准

```bash
for method in sla_cag hsa_sla_cag; do
  ASCEND_RT_VISIBLE_DEVICES=15 \
  python tests/npu/benchmark_sparse_attention.py \
    --method "${method}" --backend mindiesd --device npu:0 \
    --latent-frames 192 --warmup 5 --iterations 20 \
    | tee "logs/tests/${test_id}/${method}-same-budget-32s.txt"
done
```

两种方法都应接近 90% 有效稀疏率。记录 `selected/total`、`route_ms`、`kernel_ms`、`full_ms` 和 `speedup`。

## 3. 选块数与物理计算量扫描

```bash
ASCEND_RT_VISIBLE_DEVICES=15 \
python tests/npu/benchmark_sla_inference.py \
  --device npu:0 --backend mindiesd \
  --warmup 5 --iterations 20 \
  --kernel-selected-sweep 14,22,110,220 \
  | tee "logs/tests/${test_id}/rainfusion-selected-block-sweep.txt"
```

总 Q/K/V shape 始终不变。`kernel_sweep` 的耗时应随选中 block 数明显增长；该结果与未选 KV 扰动审计、紧凑 LUT 接口共同作为物理稀疏证据。

## 4. 新性能目录烟测

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
LONGLIVE_SPARSE_METHOD=dense BENCHMARK_MODE=dit_only \
BENCHMARK_WARMUP=0 BENCHMARK_REPEATS=1 \
RUN_ID=path-smoke-dense-5s \
bash scripts/evaluation/run_benchmark.sh 5s
```

期望生成：

```text
runs/performance/path-smoke-dense-5s/
logs/performance/path-smoke-dense-5s/
```

不要把本轮临时日志写入 `logs/benchmarks/`。
