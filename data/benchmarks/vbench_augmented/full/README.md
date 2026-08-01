# VBench Standard augmented prompts

This directory contains 944 generated prompts in the same order as
`../../vbench_standard/full/prompts.txt`.

- Generation uses `prompts.txt`.
- Output naming uses `../../vbench_standard/full/prompts.txt`.
- Evaluation uses `../../vbench_standard/full/full_info.json`.

The generated prompts come from the public VBench artifact
`prompts/augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt`.
The artifact was produced with Wan2.1's `QwenPromptExpander`,
`Qwen/Qwen2.5-3B-Instruct`, and seed 42. The source contains 946 rows; two
duplicate source prompts have identical generated text, so it maps without
ambiguity to the 944 unique prompts used here.

Source:
https://github.com/Vchitect/VBench/tree/master/prompts/augmented_prompts/Wan2.1-T2V-1.3B

This is a reproducible public prompt-augmentation baseline. It is not proven to
be byte-for-byte identical to the unpublished prompt augmentation used for the
LongLive-2.0 paper results.
