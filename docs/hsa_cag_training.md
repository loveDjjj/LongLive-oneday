# LongLive2 HSA+CAG on Ascend NPU

This branch adds a Light-Forcing-style sparse post-training and inference path
without replacing LongLive's existing causal KV cache, Ulysses SP, checkpoint,
or asynchronous VAE implementations.

## What Is Implemented

- **CAG:** each generated chunk receives a concrete sparsity value. The first
  chunk is dense and later chunks follow the `base - beta/sqrt(frames)` schedule
  while matching the configured average historical sparsity budget.
- **HSA frame routing:** every query block keeps sink frames, recent frames, and
  the most relevant middle historical frames.
- **HSA block routing:** token blocks are ranked only inside the retained
  historical frames. Current-chunk blocks remain dense by default.
- **Ascend backend:** selected K/V blocks are gathered and evaluated with
  PyTorch SDPA. Several query blocks are folded into one SDPA batch to reduce
  NPU launch overhead. Routing is non-differentiable; gradients still flow
  through selected Q/K/V attention computation.
- **Training:** prompt-only DMD post-training uses the sparse LongLive generator,
  a dense real-score teacher, and the existing trainable fake-score critic.
- **Inference:** both normal cached attention and Ulysses SP cached attention
  accept the same sparse configuration.

The NPU backend is a correctness and training backend, not a fused equivalent
of Light Forcing's Triton/FA4 CUDA kernel. It performs real sparse attention
compute, but profiling is required before claiming end-to-end acceleration.

This is an adaptation rather than a bit-for-bit copy of the paper setup.
Upstream Light Forcing trains a causal Wan2.1 student against non-causal
Wan2.1-14B score models. This repository currently supports the Wan2.2-TI2V-5B
all-causal DMD stack, so the dense teacher and critic are Wan2.2-5B. It also
uses LongLive2's native 8-latent chunk and 44x80 latent resolution. The default
0.85/0.95 sparsity pair is deliberately more conservative than the paper's
short-video 0.88/0.98 pair for the first NPU run.

## Data

The training path uses text prompts only. It does not download videos or stored
VAE latents: the student starts from noise and creates its own intermediate
latent trajectory online, while DMD compares dense-teacher and fake-critic
scores on noisy versions of that trajectory.

Prepare the exact prompt corpus used by the upstream Self-Forcing/Light-Forcing
recipe:

```bash
bash scripts/prepare_hsa_training_data.sh
```

The script downloads `gdhe17/Self-Forcing/vidprom_filtered_extended.txt`, checks
its SHA256, removes duplicates, and removes normalized exact VBench overlaps.
The expected result is 248,217 prompts. See `data/train/README.md` for license
and checksum details.

## Train

Start with a short smoke run. The 12 processes are distributed training/FSDP
workers; this trainer does not use inference SP/DP layout variables.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 \
GRADIENT_ACCUMULATION_STEPS=1 \
MAX_ITERS=10 \
TRAIN_RUN_NAME=hsa_cag_smoke \
bash scripts/run_npu_hsa_cag_training.sh
```

Then run the 2,000-iteration post-training recipe:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 \
NPROC_PER_NODE=12 \
GRADIENT_ACCUMULATION_STEPS=6 \
MAX_ITERS=2000 \
TRAIN_RUN_NAME=hsa_cag_2k \
bash scripts/run_npu_hsa_cag_training.sh
```

Reusing `TRAIN_RUN_NAME=hsa_cag_2k` resumes from the latest checkpoint in that
log directory. The launcher validates paths, generates a resolved config,
chooses another local rendezvous port if the preferred one is occupied, and
reports the effective global batch (`world_size * accumulation * batch_size`).

## Merge And Evaluate

Training checkpoints contain generator and critic LoRA state. Merge the
generator LoRA before the unified inference scripts:

```bash
/mnt/share/r50063443/conda_envs/longlive/bin/python \
  scripts/merge_lora_generator.py \
  --config_path configs/train_dmd_hsa_cag_npu_bf16.yaml \
  --generator_ckpt /mnt/share/weight/LongLive/checkpoints/longlive2_5b/longlive2_merged_generator.pt \
  --lora_ckpt logs/training/hsa_cag_2k/checkpoint_model_002000/model.pt \
  --output_path checkpoints/longlive2_hsa_cag_2k_merged.pt \
  --device npu:0
```

Run a dense control and HSA on the same merged weights:

```bash
LONGLIVE_GENERATOR_CKPT=checkpoints/longlive2_hsa_cag_2k_merged.pt \
RUN_ID=hsa_cag_dense_control \
bash scripts/run_vbench.sh longlive2_standard_20pct

LONGLIVE_GENERATOR_CKPT=checkpoints/longlive2_hsa_cag_2k_merged.pt \
LONGLIVE_SPARSE_METHOD=hsa_cag \
RUN_ID=hsa_cag_sparse \
bash scripts/run_vbench.sh longlive2_standard_20pct
```

The same switch works with msprof:

```bash
LONGLIVE_GENERATOR_CKPT=checkpoints/longlive2_hsa_cag_2k_merged.pt \
LONGLIVE_SPARSE_METHOD=hsa_cag \
bash scripts/run_msprof.sh 32s
```

Compare VBench Total/Quality/Semantic, generation-only latency/FPS, peak HBM,
FlashAttention/SDPA operator time, and HCCL time. A useful acceptance gate is
quality loss below 0.5 VBench Total with a repeatable unprofiled speedup.

## Tuning

The main settings are under `sparse_config` in the training YAML and
`sparsity.options` in the inference YAML:

| Setting | Meaning | Default |
|---|---|---:|
| `sparsity` | Average later-chunk historical sparsity target | 0.85 |
| `sparsity_base` | Late-chunk CAG base sparsity | 0.95 |
| `block_q`, `block_k` | Router/attention token block sizes | 40 |
| `keep_frames` | Frames admitted to second-stage routing | 6 |
| `keep_sink` | Always eligible earliest frames | 1 |
| `keep_near` | Always eligible most recent history frames | 2 |
| `dense_current` | Keep all current chunk K/V blocks | true |
| `query_block_batch` | Query blocks folded into one SDPA batch | 2 |

For 44x80 latent resolution, each frame has 880 tokens, so block sizes must
divide 880. Larger `query_block_batch` reduces launches but raises temporary
gather memory. Start at 2 on Ascend and profile 1/2/4 before changing sparsity.

## Scope And Limitations

- Zero-shot HSA inference is supported, but DMD post-training is recommended to
  recover the distribution shift introduced by routing. DMD is not required to
  make sparse attention execute.
- The current chunk is dense for stability, so achieved end-to-end FLOP
  reduction has a floor. The configured sparsity applies to historical blocks.
- The router uses mean-pooled query/key similarity and `topk`; route decisions
  do not receive gradients, matching common sparse-routing practice.
- Training currently uses FSDP data parallelism, not Ulysses sequence-parallel
  training. Ulysses HSA is implemented for inference.
- A custom fused Ascend block-sparse attention operator is the next performance
  step if gather plus SDPA launch overhead dominates msprof results.
