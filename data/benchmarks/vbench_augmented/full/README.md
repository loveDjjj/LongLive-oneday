# VBench Standard 完整增强提示词

本目录包含 944 条增强提示词，顺序与 `../../vbench_standard/full/prompts.txt` 一致。

- 生成使用 `prompts.txt`。
- 输出命名使用 `../../vbench_standard/full/prompts.txt`。
- 评测使用 `../../vbench_standard/full/full_info.json`。

增强提示词来自 VBench 公开产物 `prompts/augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt`，由 Wan2.1 的 `QwenPromptExpander`、`Qwen/Qwen2.5-3B-Instruct` 和 seed 42 生成。源文件有 946 行，其中两组重复原始提示词的增强文本相同，因此可无歧义映射到本目录的 944 条唯一提示词。

来源：https://github.com/Vchitect/VBench/tree/master/prompts/augmented_prompts/Wan2.1-T2V-1.3B

该数据是可公开复现的提示词增强基线，不能证明与 LongLive-2.0 论文未公开的提示词增强逐字一致。
