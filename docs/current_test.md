# 当前测试清单

本文件只记录当前提交需要在昇腾服务器执行的临时测试。下一轮测试任务必须覆盖重写本文件；长期稳定命令应迁移到[环境安装与测试](setup_and_validation.md)。

## 测试目标

验证所有 shell 分布式启动器不再使用固定端口：单机评测和训练使用 `torchrun --standalone`，多机训练由 rank 0 动态申请端口并通过共享 run 目录发布。

## 1. 更新与静态回归

```bash
cd /mnt/share/r50063443/LongLive-oneday
git pull origin feat/unified-sparse-attention

conda activate /mnt/share/r50063443/conda_envs/longlive
PYTHONPATH=. pytest -q tests/test_shell_launchers.py tests/test_evaluation_matrix.py

while IFS= read -r script; do
  bash -n "${script}"
done < <(git ls-files '*.sh')
```

预期测试全部通过，所有 shell 脚本语法检查成功。

## 2. 单机动态 rendezvous 回归

脚本不接收端口参数，直接使用 standalone rendezvous：

```bash
source /mnt/share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh

export LONGLIVE_GENERATOR_CKPT=/mnt/share/weight/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt
export SUITE_ID="dynamic-port-$(date +%Y%m%d-%H%M%S)"

METHODS=hsa_cag DURATIONS=5s SP_SIZES=1 MODES=dit_only PERF_DEVICES=4 \
bash scripts/evaluation/run_performance_matrix.sh
```

预期日志包含 `rendezvous=standalone`，完成 1 次预热和 3 次有效测量并生成：

```text
runs/performance/${SUITE_ID}-hsa_cag-5s-dit_only-sp1/summary.json
runs/suites/${SUITE_ID}/performance/results.csv
```

多节点动态端口需要两台机器共享 `runs/training/<TRAIN_RUN_NAME>/`，并先启动 rank 0；本轮若不安排多节点训练，可只完成上述单机验收。
