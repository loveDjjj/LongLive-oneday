# LongLive2.0 Usage

This `npu` branch keeps the existing AR and DMD training paths unchanged and
consolidates Ascend BF16 inference into two maintained workflows. NVFP4 configs
are intentionally not shipped on this branch.

## Installation

Use the versions of PyTorch, TorchVision, `torch_npu`, and CANN supported by the
target Ascend server, then install the project dependencies:

```bash
pip install -r requirements_npu_bf16.txt
```

Do not replace the server's matched Ascend packages with CUDA wheels. Download
the Wan2.2-TI2V-5B components and update model paths before running.

## Training

Training configs and entrypoints retain their existing organization.

### AR diffusion training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config_path configs/train_ar.yaml \
  --logdir logs/test_train_ar \
  --wandb-save-dir wandb \
  --disable-wandb
```

### DMD distillation

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config_path configs/train_dmd.yaml \
  --logdir logs/test_train_dmd \
  --wandb-save-dir wandb \
  --disable-wandb
```

Relevant training controls remain in the training YAML files:

- `infra.sequence_parallel_size` controls the SP group size.
- `infra.vae_halo_latents` controls chunk-halo VAE preparation.
- `algorithm` selects AR or DMD objective settings.
- `training` contains optimizer, batch size, checkpoint, and loop settings.
- `adapter` enables LoRA when present.

## Ascend inference

Inference has exactly two public launchers:

```bash
bash scripts/run_msprof.sh 32s
bash scripts/run_vbench.sh longlive2_standard_20pct
```

Their source configs are:

```text
configs/inference/msprof.yaml
configs/inference/vbench.yaml
```

See [`NPU_BF16_RUN_GUIDE.md`](NPU_BF16_RUN_GUIDE.md) for the device layout,
all presets, dataset definitions, run IDs, profiler semantics, and resume flow.

## Utilities

Inspect SP VAE halo windows:

```bash
python scripts/compute_sp_vae_chunk_halo.py --config configs/train_ar.yaml
```

Decode saved VAE latents:

```bash
python scripts/decode_vae_latents.py --help
python scripts/decode_lightvae_latents.py --help
```
