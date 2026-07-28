# VBench Standard 20% augmented prompts

This dataset uses the same 186 original prompts and 187 metadata records as
`../vbench_standard_20pct`, with generation text mapped to the public
Wan2.1/Qwen2.5-3B seed-42 augmentation artifact.

- `prompts.txt`: augmented prompts used by LongLive generation.
- `original_prompts.txt`: original prompts used for VBench file names.
- `VBench_full_info.json`: matching 20% VBench metadata.

The mapping is selected from
`../vbench_standard_augmented_wan21_qwen25_seed42`; therefore the generated and
original prompt files are index-aligned. This is a public augmentation proxy,
not a proven reproduction of the unpublished LongLive-2.0 prompt augmentation.
