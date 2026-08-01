# Inference benchmark data

This directory contains prompt metadata only. Generated videos and evaluator
checkpoints are not committed.

- `performance/`: ten fixed prompts used only by `scripts/run_msprof.sh`.
- `vbench_standard/full/`: VBench Standard, 946 metadata records and 944 unique prompts.
- `vbench_standard/5pct/`: AISBench K-Means Mini, 43 unique prompts.
- `vbench_standard/20pct/`: independent K-Means subset, 186 unique prompts.
- `vbench_augmented/`: Qwen2.5 seed-42 generation prompts for the same three sets.
  Evaluation naming and metadata always point back to the matching Standard set.
- `vbench_long/`: long-video evaluator parameters only; prompts and metadata are
  shared with `vbench_standard/full/`.

The 5% and 20% sets were sampled independently from Full Standard. The 5% set
is not a subset of the 20% set.

Sources are documented in the per-dataset README files and in
`docs/NPU_BF16_RUN_GUIDE.md`.
