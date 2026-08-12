# Unified sparse-attention baseline

This branch keeps four concerns separate: LongLive rolling KV management,
algorithm routing, block-sparse execution, and evaluation. The current baseline
supports `dense`, native `hsa_cag`, native `sla_cag`, and the experimental
`hsa_sla_cag` hybrid.

## Method selection

Both inference YAML files contain independent HSA and SLA profiles. Select one
without editing YAML:

```bash
export LONGLIVE_SPARSE_METHOD=dense       # LongLive2 baseline
export LONGLIVE_SPARSE_METHOD=hsa_cag     # frame-then-block routing
export LONGLIVE_SPARSE_METHOD=sla_cag     # global block routing + linear branch
export LONGLIVE_SPARSE_METHOD=hsa_sla_cag # frame-filtered SLA + linear branch
export LONGLIVE_SPARSE_BACKEND=mindiesd   # optional backend override
```

HSA preserves its original dense-current policy. SLA preserves its global
Smooth-K block routing and linear compensation. Both emit the same compact
block LUT and use the shared RainFusion/Ascend Triton execution code.

The hybrid uses CAG for the final block budget, selects an eight-frame candidate
pool using HSA, applies SLA Smooth-K Top-K inside that pool, and adds the full-KV
linear compensation. Its initial profile uses target/base sparsity `0.90/0.93`,
one sink frame, one recent frame, and does not force the current chunk dense.

## Performance matrix

The supported durations are:

- `5s`: 32 latent frames, 125 pixel frames, 5.208 seconds at 24 FPS.
- `32s`: 192 latent frames, 765 pixel frames.
- `64s`: 384 latent frames, 1533 pixel frames.

The supported runtime modes are:

- `dit_only`: four SP workers, VAE disabled, latent saved.
- `sync_vae`: four SP workers, leader performs synchronous VAE decode.
- `async_vae`: four SP workers plus one dedicated VAE NPU.

Run one case:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
LONGLIVE_SPARSE_METHOD=hsa_cag BENCHMARK_MODE=dit_only \
bash scripts/evaluation/run_benchmark.sh 32s

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
LONGLIVE_SPARSE_METHOD=sla_cag MSPROF_MODE=dit_only \
bash scripts/evaluation/run_msprof.sh 32s

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4 \
LONGLIVE_SPARSE_METHOD=dense BENCHMARK_MODE=async_vae \
bash scripts/evaluation/run_benchmark.sh 64s
```

Run the complete unprofiled matrix:

```bash
PERF_DEVICES=0,1,2,3,4 \
TASK=benchmark \
SUITE_ID=sparse_release_01 \
bash scripts/evaluation/run_performance_matrix.sh
```

Every successful case writes both `summary.txt` and `summary.json`. After the
matrix completes, the comparable rows are collected under
`runs/suites/<suite-id>/benchmark/results.{csv,json}`. `dense_speedup` uses the
dense row with the same duration preset and VAE mode. Always check
`checkpoint_matched_to_dense`: a false value means the timing also changed the
model checkpoint and is not a controlled kernel comparison.

For VAE modes, the summary also reports `ar_loop_seconds_mean`,
`vae_decode_seconds_mean`, `vae_enqueue_seconds_mean`,
`vae_drain_seconds_mean`, and `vae_overlap_seconds_mean`. The dedicated-NPU
pipeline defines overlap as `max(vae_decode - final_drain, 0)`. This is pipeline
wall-clock telemetry rather than summed device-kernel time; use msprof to
attribute operator execution. A large enqueue value indicates cross-device
latent transfer is stalling the producer, while a large drain value means VAE
decode remains behind the DiT producer at the final chunk.

Use a dry run to audit all 36 combinations without reserving devices:

```bash
DRY_RUN=1 PERF_DEVICES=0,1,2,3,4 \
bash scripts/evaluation/run_performance_matrix.sh
```

Matrix launchers default to `RESUME_SUITE=1`: cases with a completed
`summary.json` (or `vbench_results.json`) are skipped. An existing incomplete
performance case is never overwritten; inspect it and use a new `SUITE_ID` or
remove it explicitly after preserving its logs.

Set `LONGLIVE_GENERATOR_CKPT` to use one checkpoint for a controlled routing
comparison. For best-per-method comparisons, set any of
`DENSE_GENERATOR_CKPT`, `HSA_CAG_GENERATOR_CKPT`, `SLA_CAG_GENERATOR_CKPT`, and
`HSA_SLA_CAG_GENERATOR_CKPT`; method-specific values take precedence over the
shared checkpoint in both performance and VBench matrices.

```bash
DENSE_GENERATOR_CKPT=/weights/longlive2.pt \
HSA_CAG_GENERATOR_CKPT=/weights/hsa.pt \
SLA_CAG_GENERATOR_CKPT=/weights/sla.pt \
HSA_SLA_CAG_GENERATOR_CKPT=/weights/hybrid.pt \
PERF_DEVICES=0,1,2,3,4 \
bash scripts/evaluation/run_performance_matrix.sh
```

When only four NPUs are available, select modes that do not require a dedicated
VAE device:

```bash
PERF_DEVICES=0,1,2,3 MODES=dit_only \
bash scripts/evaluation/run_performance_matrix.sh
```

Use `TASK=msprof` for profiling. A profile includes collection overhead and is
diagnostic evidence, not the release latency number. Use `dit_only` profiles
for attention/operator comparison, then sync/async profiles for VAE and overlap
analysis. Profile suites are written separately under
`runs/suites/<suite-id>/msprof`; their rows carry `profiled=true`.

If a matrix stops before all cases finish, completed run directories remain
usable. Collect the partial evidence explicitly with:

```bash
python scripts/evaluation/summarize_suite.py benchmark \
  --suite-id sparse_release_01
```

Before an end-to-end run, validate both sparse implementations at the real SP4
tail shape:

```bash
python tests/npu/benchmark_sparse_attention.py \
  --method hsa_cag --device npu:0 --warmup 5 --iterations 20
python tests/npu/benchmark_sparse_attention.py \
  --method sla_cag --device npu:0 --warmup 5 --iterations 20
python tests/npu/benchmark_sparse_attention.py \
  --method hsa_sla_cag --device npu:0 --warmup 5 --iterations 20 \
  --check-linear-backward
```

The hybrid backward check keeps Q/K/V frozen and verifies finite, nonzero
gradients on `sla_linear`, matching the generator's `linear_only` training
scope before committing a multi-card DMD smoke run.

## VBench

`run_vbench.sh` uses the same method selector:

```bash
LONGLIVE_GENERATOR_CKPT=/path/to/checkpoint.pt \
LONGLIVE_SPARSE_METHOD=hsa_cag \
bash scripts/evaluation/run_vbench.sh longlive2_standard_20pct
```

For a valid attribution, compare dense and sparse execution using the same
merged checkpoint, prompt set, seed, frame count, resolution, and sampling
settings. Comparing an HSA-trained checkpoint with an SLA-trained checkpoint
mixes routing effects with training effects.

Run the same checkpoint through all four methods:

```bash
VBENCH_PRESETS=longlive2_standard_20pct \
SUITE_ID=vbench_release_01 \
bash scripts/evaluation/run_vbench_matrix.sh /path/to/merged_generator.pt
```

Each run records its 16 dimension scores in `vbench_results.json` and copies the
AISBench summary files beside it. The matrix writes
`runs/suites/<suite-id>/vbench/results.{csv,json}`. Quality, semantic, and total
scores are imported only when they are present in the official AISBench summary;
the repository does not reconstruct those aggregates with a local formula.

The same method-specific checkpoint variables can be used with
`run_vbench_matrix.sh` when the goal is to compare each method's best trained
checkpoint instead of isolating routing behavior.

## Training

The native training entry points are:

```bash
bash scripts/training/run_hsa_cag.sh
bash scripts/training/run_sla_cag.sh
bash scripts/training/run_hsa_sla_cag.sh
```

All three delegate to `run_sparse_cag.sh`. The resolved config and checkpoint record
the selected sparse method, and resume rejects a checkpoint from another
method. The current native recipes retain their historical LoRA scope. The
hybrid recipe uses `generator_train_scope=linear_only`: the original generator
is frozen and only the 30 per-layer `sla_linear` projections are updated. The
fake-score critic still uses LoRA for DMD. Checkpoints save the generator tensors
under `generator_linear` and can be exported with `merge_lora.py` using
`configs/train/hsa_sla_cag.yaml`.

```bash
python scripts/checkpoints/merge_lora.py \
  --config_path configs/train/hsa_sla_cag.yaml \
  --generator_ckpt /path/to/longlive2_merged_generator.pt \
  --lora_ckpt /path/to/checkpoints/step_0001000/train_state.pt \
  --output_path /path/to/hsa_sla_cag_linear_1000.pt \
  --device npu:0 --dtype bf16
```
