# VBench Mini with Wan2.1/Qwen2.5 augmentation

This dataset maps all 43 original prompts from `../vbench_mini` to the public
Wan2.1/Qwen2.5-3B seed-42 prompt augmentation in
`../vbench_standard_augmented_wan21_qwen25_seed42`.

- `prompts.txt`: augmented prompts used for video generation.
- `original_prompts.txt`: original VBench prompts used for output filenames.
- `VBench_full_info.json`: the unchanged 43-record VBench Mini metadata.

The mapping is exact and one-to-one. There are no missing or ambiguous prompts.
This subset covers all 16 VBench dimensions, but its scores are intended for
fast regression and model comparison rather than full VBench reporting.
