# VBench Standard 20% K-Means 子集

该数据集是 VBench 1.0 Standard 的固定代表性子集，用于较快的 LongLive 质量回归测试。

- `prompts.txt`：186 条用于生成的唯一提示词。
- `full_info.json`：187 条官方元数据记录，覆盖 16 个 VBench 维度；其中一条提示词对应两条元数据记录。

该子集基于 AISBench/datasets 提交 `26f93b6`，使用官方 VBench 元数据特征和 K-Means 流程生成：

```bash
python mini_datasets/select_metadata_by_kmeans.py \
  --input mini_datasets/vbench_1.0_mini/vbench_metadata \
  --work-dir <output> \
  --compression-ratio 0.2 \
  -a
```

抽样比例分别作用于 11 个源提示词集合，再合并为 186 条唯一提示词，占完整 944 条提示词的 19.7%。该子集适合回归和方案对比，但其分数不能作为完整 VBench Standard 结果报告。

来源：https://github.com/AISBench/datasets/tree/main/mini_datasets/vbench_1.0_mini
