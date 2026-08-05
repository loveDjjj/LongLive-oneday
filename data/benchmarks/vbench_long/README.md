# VBench-Long protocol assets

VBench-Long reuses the 944 prompts and metadata from
`../vbench_standard/full/`. It is not another Standard sampling ratio.

Only its protocol-specific evaluator parameters live here. They control
per-dimension clip lengths, slow/fast within-clip and cross-clip aggregation,
and subject/background score calibration. The official VBench-Long evaluator
still requires Ascend adaptation, so these files are not exposed by the current
`scripts/evaluation/run_vbench.sh` presets.
