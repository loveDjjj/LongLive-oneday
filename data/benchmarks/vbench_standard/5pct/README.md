# VBench Standard 5% K-Means subset

This is the 43-prompt AISBench VBench-1.0 Mini K-Means sample, normalized under
the same `full/5pct/20pct` naming used by the inference config.

- `prompts.txt`: 43 unique prompts covering all 16 VBench dimensions.
- `full_info.json`: matching evaluator metadata.

It is intended for quick regression and model comparison rather than full
VBench reporting. The 5% and 20% sets were sampled independently; this set is
not a subset of `../20pct/`.

Source: https://github.com/AISBench/datasets/tree/main/mini_datasets/vbench_1.0_mini
