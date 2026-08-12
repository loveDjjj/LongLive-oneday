# 推理评测数据

本目录只保存提示词与评测元数据，不提交生成视频和评测模型权重。

- `performance/`：供性能与 msprof 测试使用的固定提示词。
- `vbench_standard/full/`：VBench Standard，946 条元数据记录、944 条唯一提示词。
- `vbench_standard/5pct/`：AISBench K-Means Mini，43 条唯一提示词。
- `vbench_standard/20pct/`：独立 K-Means 子集，186 条唯一提示词。
- `vbench_augmented/`：由 Qwen2.5、seed 42 生成的增强提示词，分别与三个 Standard 集合对齐；输出命名和评测元数据仍使用对应 Standard 集合。
- `vbench_long/`：仅保存长视频协议参数，提示词和元数据复用 `vbench_standard/full/`。

5% 和 20% 从 Full Standard 独立抽样，5% 不是 20% 的子集。来源见各数据集目录的 README，运行方法见[推理与评测指南](../../docs/inference_and_evaluation.md)。
