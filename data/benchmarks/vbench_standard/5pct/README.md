# VBench Standard 5% K-Means 子集

该数据是包含 43 条提示词的 AISBench VBench-1.0 Mini K-Means 样本，目录命名与推理配置中的 `full/5pct/20pct` 约定一致。

- `prompts.txt`：43 条唯一提示词，覆盖全部 16 个 VBench 维度。
- `full_info.json`：对应的 evaluator 元数据。

该子集用于快速回归和模型对比，不应作为完整 VBench 结果报告。5% 与 20% 独立抽样，本目录不是 `../20pct/` 的子集。

来源：https://github.com/AISBench/datasets/tree/main/mini_datasets/vbench_1.0_mini
