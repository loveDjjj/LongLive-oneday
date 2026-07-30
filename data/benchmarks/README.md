# LongLive benchmark datasets

This directory contains prompt metadata only. It does not include generated
videos or VBench evaluator checkpoints.

- `vbench_mini/`: AISBench VBench-1.0-mini 5% k-means subset. The downloaded
  `VBench_kmeans_info_0.05.json` is stored as `VBench_full_info.json` so it can
  be selected directly by the AISBench VBench configuration.
- `vbench_standard/`: Official VBench 1.0 Standard prompts, metadata, and
  `VBench_full_info.json` from the Vchitect/VBench repository. The metadata
  contains 946 records and 944 unique English prompts.
- `vbench_standard_augmented_wan21_qwen25_seed42/`: the same 944 prompts after
  the public Wan2.1 `QwenPromptExpander` augmentation with
  `Qwen/Qwen2.5-3B-Instruct` and seed 42. Original prompts are retained for
  VBench file names and labels.
- `vbench_standard_20pct/`: a fixed AISBench K-Means subset with 187 metadata
  records and 186 unique prompts, covering all 16 VBench metrics.
- `vbench_standard_20pct_augmented_wan21_qwen25_seed42/`: the same 186-prompt
  subset mapped to the public Wan2.1/Qwen2.5 seed-42 augmentation.
- `vbench_mini_augmented_wan21_qwen25_seed42/`: the 43-prompt VBench Mini
  subset mapped to the same public augmentation, with original prompts retained
  for VBench-compatible output names.
- `vbench_long/`: Official VBench-Long metadata and clip-splitting configs.
- `performance/`: A small fixed prompt set for Ascend latency and throughput
  measurements. It is not a quality benchmark dataset.

Sources:

- https://modelers.cn/datasets/AISBench/VBench-1.0-mini
- https://github.com/Vchitect/VBench/tree/master/prompts
- https://github.com/Vchitect/VBench/tree/master/vbench2_beta_long

The LongLive-2.0 paper evaluates 60-second videos generated from MovieGenBench
prompts. No official downloadable MovieGenBench prompt artifact was found in
the paper repository or the benchmark sources above. Do not label results from
`vbench_long/prompts.txt` as a direct reproduction of the paper's Table 5.
