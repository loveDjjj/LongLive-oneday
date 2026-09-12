# Codex 项目规范

本文件是 Codex 进入仓库后必须首先阅读的项目级约束。开始任何任务前，还必须阅读[当前测试清单](docs/current_test.md)，再按任务范围阅读对应中文文档。

## 1. 项目边界

- 当前主线是 LongLive2.0-5B 在昇腾 NPU 上的 BF16 训练、推理和评测。
- CUDA/H100 复用同一套入口，以 `LLV2_DEVICE=cuda` 显式选择；NPU 默认行为不变。CUDA 默认使用 `portable` 参考后端，`cuda_flex` 必须显式启用并完成真实 GPU 前向、Q/K/V 反向及多卡准入，不能把主机测试或 NPU 结果当作 H100 验证。
- 维护的方法为 `dense`、`hsa_cag`、`sla_cag` 和 `hsa_sla_cag`。
- 训练稀疏后端为 `ascend_triton`；推理稀疏后端默认为 MindIE-SD RainFusion。未经真实 NPU 验证，不得把实验后端改为默认值。
- LongLive2.0 的滚动 KV 窗口为 32 个 latent 帧。改变窗口长度属于模型行为变更，必须重新训练和评测，不能作为普通推理参数优化。
- 三种稀疏方法默认使用相同的 CAG 目标/基准稀疏率 `0.85/0.95`，用于公平比较路由差异。
- SLA+CAG 与 HSA+SLA+CAG 默认训练范围均为 `lora_plus_linear`：主干使用 LoRA，原始 `sla_linear` 直接训练且不得包装 LoRA；`linear_only` 只作为混合方法隔离补偿层能力的对照。

## 2. 开始任务前

1. 阅读本文件和 `docs/current_test.md`。
2. 查看 `git status`，保留用户已有修改。
3. 训练任务阅读 `docs/training.md`；推理、性能或质量任务阅读 `docs/inference_and_evaluation.md`；环境与长期验收阅读 `docs/setup_and_validation.md`。
4. 修改前核对真实调用链、配置解析器和测试，不根据文档单独推断代码行为。

## 3. 测试文档维护

- `docs/setup_and_validation.md` 只保存长期稳定、可重复使用的环境检查、smoke test 和发布验收命令。
- `docs/current_test.md` 只保存当前一轮需要服务器执行的临时命令、预期结果和结果回填位置。
- 每次产生新一轮临时测试命令时，必须覆盖重写 `docs/current_test.md`，不得不断追加历史命令。
- 临时命令验证稳定并成为长期准入条件后，将其整理到 `docs/setup_and_validation.md`，并从 `docs/current_test.md` 删除。
- 向用户交付命令前，必须同步更新上述对应文档；文档命令与聊天命令必须一致。
- 主机测试不能替代 NPU 测试。NPU 未执行时应明确写“待服务器验证”，不能表述为已通过。

## 4. 配置规范

- 所有 YAML 配置必须使用中文注释说明用途、单位、默认行为和关键约束。
- 稀疏配置至少解释：`sparsity`、`sparsity_base`、128-token block、frame-stage protected 策略、block-stage hard anchor、`dense_current_blocks`、首 chunk 行为、缓存和后端。
- 训练配置至少解释：并行布局、训练范围、有效 batch、更新频率、checkpoint、数据形状和训练内验证。
- 推理配置至少解释：模型路径、帧数换算、SP/DP、VAE 模式、预热、profiler 参数和稀疏 profile。
- 修改配置默认值时，必须同步 dataclass 默认值、配置解析测试、NPU benchmark 默认值和中文文档。
- resolved 配置是运行证据；启动脚本必须将最终配置写入对应 run 目录。

## 5. 代码与实验规范

- 优先复用现有模块和脚本，不复制新的平行入口。
- 稀疏收益分三层报告：算子完整路径、DiT-only 整网、包含 VAE 的端到端。三者不能互相替代。
- 无 profiler benchmark 用于发布延迟；msprof 只用于归因，不能用其插桩延迟计算发布加速比。
- Dense 与 sparse 对比必须固定 checkpoint、提示词、seed、分辨率、帧数、SP/DP、采样参数、VAE 模式和设备集合。
- “物理稀疏”必须证明算子只消费选中 KV blocks。不能仅凭 mask、理论 FLOPs 或输出稀疏率下结论。
- SLA 与混合方法包含全 KV 线性补偿分支；报告时应将“稀疏 softmax 只读选中 blocks”和“线性分支读取完整 KV 统计量”分开描述。
- 不得用一次测量确定性能结论。正式对比至少 1 次预热、3 次有效测量，并报告 p50。
- Shell 启动器不得硬编码或要求用户指定 rendezvous 端口。单机 `torchrun` 使用 `--standalone`；多机由 rank 0 动态申请端口并通过共享 run 目录发布。

## 6. 产物目录

`runs/` 保存结构化产物、模型输出、汇总和 profiler 数据：

```text
runs/training/<run-id>/
runs/performance/<run-id>/
runs/msprof/dit/<run-id>/
runs/msprof/vae/<run-id>/
runs/vbench/<run-id>/
runs/suites/<suite-id>/performance/
runs/suites/<suite-id>/msprof/dit/
runs/suites/<suite-id>/vbench/
```

`logs/` 保存人类可读文本日志，并尽量镜像 `runs/` 的任务层级：

```text
logs/training/<run-id>/
logs/performance/<run-id>/
logs/msprof/dit/<run-id>/
logs/msprof/vae/<run-id>/
logs/vbench/<run-id>/
logs/tests/<test-id>/
```

- `logs/tests/` 保存算子 smoke、microbenchmark、临时正确性与人工 `tee` 日志。
- 不再创建 `logs/benchmark/`、`logs/benchmarks/`、`runs/benchmark/` 或 `runs/vae_msprof/`。
- 不将无 profiler benchmark 合并到 msprof。二者测量目的和开销不同，必须分目录保留。
- 运行目录不可静默覆盖。复用已有结果必须显式 resume；不完整目录应先人工检查并保留证据。

## 7. 文档与格式

- README、`docs/`、数据说明、配置注释和面向用户的项目文档统一使用中文。
- 命令、环境变量、方法 ID、文件名、API 名称保持原始英文。
- 文件名使用小写英文和下划线；已有对外脚本名除非必要不重命名。
- 文档只维护一个权威入口：环境测试、训练、推理评测分别对应三份主文档，避免重复说明漂移。
- 代码修改后至少执行相关单元测试、`git diff --check` 和适用的配置/脚本解析检查。

## 8. Git 交付

- 提交只包含当前任务相关文件，不回退用户修改。
- 提交前检查工作区、差异、旧路径引用和本地 Markdown 链接。
- 推送后核对远端分支指针，并在交付中给出提交号、服务器拉取命令和 `docs/current_test.md` 中的测试命令。
