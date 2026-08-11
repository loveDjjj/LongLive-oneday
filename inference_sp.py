# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import time
from math import gcd

import peft
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.io import write_video
from tqdm import tqdm

from pipeline.causal_diffusion_inference_sp import CausalDiffusionInferencePipelineSP
from utils.config import normalize_config, section_get
from utils.dataset import PromptDataset, prompt_collate_fn
from utils.device import distributed_backend, is_npu, set_device
from utils.inference_utils import load_generator_checkpoint, load_lora_state_dict
from utils.lora_utils import configure_lora_for_model
from utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
from utils.misc import set_seed


def synchronize_accelerator(device):
    accelerator = getattr(torch, device.type, None)
    synchronize = getattr(accelerator, "synchronize", None)
    if synchronize is not None:
        synchronize()


def reset_peak_memory(device):
    accelerator = getattr(torch, device.type, None)
    reset = getattr(accelerator, "reset_peak_memory_stats", None)
    if reset is not None:
        try:
            reset(device)
        except (RuntimeError, TypeError):
            try:
                reset()
            except (RuntimeError, TypeError):
                pass


def peak_memory_gb(device):
    accelerator = getattr(torch, device.type, None)
    max_memory = getattr(accelerator, "max_memory_allocated", None)
    if max_memory is None:
        return None
    try:
        return max_memory(device) / (1024 ** 3)
    except (RuntimeError, TypeError):
        try:
            return max_memory() / (1024 ** 3)
        except (RuntimeError, TypeError):
            return None


def save_prompts_to_txt(prompts_for_sample, prompt_txt_path: str, is_main_process: bool):
    try:
        with open(prompt_txt_path, "w", encoding="utf-8") as f:
            if len(prompts_for_sample) == 0:
                return
            current_prompt = prompts_for_sample[0]
            current_indices = [0]
            for seg_idx in range(1, len(prompts_for_sample)):
                prompt = prompts_for_sample[seg_idx]
                if prompt == current_prompt:
                    current_indices.append(seg_idx)
                else:
                    f.write(f"[{','.join(str(i) for i in current_indices)}] {current_prompt}\n")
                    current_prompt = prompt
                    current_indices = [seg_idx]
            f.write(f"[{','.join(str(i) for i in current_indices)}] {current_prompt}\n")
    except Exception as exc:
        if is_main_process:
            print(f"Warning: failed to save prompts to {prompt_txt_path}: {exc}")


def compute_group_specs(world_size, sp_size, dp_size, num_heads, num_frame_per_block,
                        auto_sp_remainder=False):
    """Compute DP groups whose ranks each form a Ulysses SP group."""
    valid_base = gcd(num_heads, num_frame_per_block)
    valid_sp_sizes = sorted(size for size in range(1, valid_base + 1) if valid_base % size == 0)
    if sp_size not in valid_sp_sizes:
        raise ValueError(
            f"sp_size={sp_size} must divide gcd(num_heads={num_heads}, "
            f"num_frame_per_block={num_frame_per_block})={valid_base}"
        )

    full_groups = min(world_size // max(sp_size, 1), dp_size)
    groups = [
        (sp_size, list(range(i * sp_size, (i + 1) * sp_size)))
        for i in range(full_groups)
    ]
    ranks_used = full_groups * sp_size
    if auto_sp_remainder:
        remaining_gpus = world_size - ranks_used
        remaining_quota = dp_size - full_groups
        while remaining_gpus > 0 and remaining_quota > 0:
            size = max((v for v in valid_sp_sizes if v <= remaining_gpus), default=1)
            groups.append((size, list(range(ranks_used, ranks_used + size))))
            ranks_used += size
            remaining_gpus -= size
            remaining_quota -= 1
    return groups


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the config YAML file")
args = parser.parse_args()

config = normalize_config(OmegaConf.load(args.config_path))
if getattr(config, "model_quant", False):
    raise NotImplementedError(
        "The supported inference path is BF16 only; set model_quant=false."
    )
if not hasattr(config, "sampling_steps") or config.sampling_steps is None:
    raise ValueError("sampling_steps must be defined in the SP inference config")
if not hasattr(config, "guidance_scale") or config.guidance_scale is None:
    config.guidance_scale = 1.0

config.use_ema = section_get(config, "inference", "use_ema", getattr(config, "use_ema", False))
config.output_folder = section_get(config, "inference", "output_folder", getattr(config, "output_folder", "videos/longlive2_sp"))
config.num_samples = section_get(config, "inference", "num_samples", getattr(config, "num_samples", 1))
config.num_output_frames = getattr(config, "num_output_frames", config.image_or_video_shape[1])
config.save_with_index = getattr(config, "save_with_index", False)
config.inference_iter = getattr(config, "inference_iter", -1)
if getattr(config, "i2v", False):
    raise NotImplementedError("I2V inference is not included in this SP release path.")
if getattr(config, "kv_quant", False):
    raise NotImplementedError("The supported inference path requires kv_quant=false.")

sparse_config = getattr(config, "sparse_config", None)
if sparse_config:
    from wan_5b.modules.sla_attention import SLAAttentionConfig

    parsed_sparse_config = SLAAttentionConfig.from_mapping(sparse_config)
    if parsed_sparse_config.enabled and parsed_sparse_config.backend == "mindiesd":
        from wan_5b.modules.sla_attention_mindiesd import (
            mindiesd_available,
            mindiesd_unavailable_reason,
        )

        if not mindiesd_available():
            raise RuntimeError(
                "SLA sparse inference requires MindIE-SD RainFusionAttention, but it could "
                "not be imported. Install a MindIE-SD build matching torch_npu/CANN "
                f"before loading the model. Detail: {mindiesd_unavailable_reason()}"
            )

sp_size = int(getattr(config, "sp_size", 1))
dp_size = int(getattr(config, "dp_size", 1))
auto_sp_remainder = bool(getattr(config, "auto_sp_remainder", False))
model_num_heads = int(getattr(config, "model_num_heads", 24))

sp_group = None
dp_rank = 0
sp_rank = 0
effective_sp_size = sp_size
total_dp_groups = 1

if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    device = set_device(local_rank)
    if distributed_backend() == "nccl":
        os.environ.setdefault("NCCL_CROSS_NIC", "1")
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("NCCL_TIMEOUT", "1800")
    if not dist.is_initialized():
        dist.init_process_group(backend=distributed_backend(), rank=rank, world_size=world_size)
    set_seed(config.seed + rank)
    config.distributed = True

    group_specs = compute_group_specs(
        world_size=world_size,
        sp_size=sp_size,
        dp_size=dp_size,
        num_heads=model_num_heads,
        num_frame_per_block=int(getattr(config, "num_frame_per_block", 8)),
        auto_sp_remainder=auto_sp_remainder,
    )
    assigned_count = sum(len(ranks) for _, ranks in group_specs)
    if assigned_count != world_size:
        raise ValueError(
            f"SP group layout assigns {assigned_count}/{world_size} ranks. "
            "Increase dp_size, lower sp_size, or enable auto_sp_remainder."
        )
    total_dp_groups = len(group_specs)
    sp_groups_all = [dist.new_group(ranks=ranks) for _, ranks in group_specs]
    for dp_i, (eff_sp, ranks) in enumerate(group_specs):
        if rank in ranks:
            dp_rank = dp_i
            sp_rank = ranks.index(rank)
            effective_sp_size = eff_sp
            sp_group = sp_groups_all[dp_i]
            break
    if rank == 0:
        print(
            f"[SP] Parallelism: {total_dp_groups} DP group(s), "
            f"sp_sizes={[size for size, _ in group_specs]}, assigned={assigned_count}/{world_size}"
        )
else:
    local_rank = 0
    world_size = 1
    rank = 0
    device = set_device(local_rank)
    set_seed(config.seed)
    config.distributed = False
    effective_sp_size = 1

is_main_process = rank == 0
use_effective_sp = effective_sp_size > 1 and dist.is_initialized()
use_multi_dp = total_dp_groups > 1

if use_effective_sp:
    from wan_5b.distributed.sp_ulysses_inference import init_sequence_parallel
    init_sequence_parallel(group=sp_group)
    if is_main_process:
        print(f"[SP] Ulysses mode enabled: sp_sizes={[size for size, _ in group_specs]}")
elif is_main_process:
    print("[SP] Running SP model with world_size=1")

torch.set_grad_enabled(False)
free_vram = get_cuda_free_memory_gb(device)
low_memory = free_vram < 40
if is_main_process:
    print(f"[SP] Free VRAM: {free_vram:.1f} GB, low_memory={low_memory}")

pipeline = CausalDiffusionInferencePipelineSP(
    config,
    device=device,
    sp_group=sp_group,
    dp_rank=dp_rank,
)

merge_lora = bool(getattr(config, "merge_lora", False))
has_lora_adapter = bool(getattr(config, "adapter", None) and configure_lora_for_model is not None)
generator_ckpt_path = getattr(config, "generator_ckpt", None)
if not generator_ckpt_path:
    raise ValueError("checkpoints.generator_ckpt is required for inference")
if is_main_process:
    print(f"[SP] Loading generator checkpoint: {generator_ckpt_path}")
load_generator_checkpoint(
    pipeline.generator,
    generator_ckpt_path,
    use_ema=bool(config.use_ema),
    strict=True,
)

pipeline.is_lora_enabled = False
pipeline.is_lora_merged = False
if has_lora_adapter:
    if is_main_process:
        print(f"[SP] Applying LoRA config: {config.adapter}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=is_main_process,
    )
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if not lora_ckpt_path:
        raise ValueError("An adapter config requires checkpoints.lora_ckpt")
    peft.set_peft_model_state_dict(
        pipeline.generator.model,
        load_lora_state_dict(lora_ckpt_path),
    )
    if merge_lora:
        pipeline.generator.model = pipeline.generator.model.merge_and_unload(safe_merge=True)
        pipeline.is_lora_merged = True
    else:
        pipeline.is_lora_enabled = True
elif merge_lora and is_main_process:
    print("merge_lora=True requested but no adapter config was found; continuing without LoRA merge")

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)

pipeline.generator.model.eval().requires_grad_(False)

vae_device_str = getattr(config, "vae_device", None)
use_dedicated_vae_device = bool(getattr(config, "streaming_vae", False)) and bool(vae_device_str)
decode_on_this_rank = not use_effective_sp or sp_rank == 0
dedicated_vae_device = None
if use_dedicated_vae_device and total_dp_groups != 1:
    raise ValueError(
        "A single inference.vae_device cannot serve multiple DP groups safely. "
        "Use dp_size=1, or launch one independent SP+VAE process per replica."
    )
if use_dedicated_vae_device and is_npu():
    requested_vae_device = torch.device(vae_device_str)
    if requested_vae_device.type != "npu" or requested_vae_device.index is None:
        raise ValueError(
            "Ascend asynchronous VAE requires an explicit device such as vae_device: npu:4."
        )
    if requested_vae_device.index < world_size:
        raise ValueError(
            f"vae_device={requested_vae_device} overlaps the {world_size} torchrun worker "
            "devices. Expose one extra NPU and use its logical index."
        )
    if requested_vae_device.index >= torch.npu.device_count():
        raise ValueError(
            f"vae_device={requested_vae_device} is unavailable; "
            f"ASCEND_RT_VISIBLE_DEVICES exposes {torch.npu.device_count()} logical NPUs."
        )
if use_dedicated_vae_device and decode_on_this_rank:
    vae_device = torch.device(vae_device_str)
    dedicated_vae_device = vae_device
    pipeline.vae.to(device="cpu")
    pipeline.vae.to(device=vae_device)
    if hasattr(pipeline.vae, "mean"):
        pipeline.vae.mean = pipeline.vae.mean.to(device=vae_device)
        pipeline.vae.std = pipeline.vae.std.to(device=vae_device)
    if is_main_process:
        print(f"[SP] Async VAE on {vae_device}, diffusion on {device}")
elif decode_on_this_rank:
    pipeline.vae.to(device=device)
else:
    # SP produces the same gathered latent on every rank. Only the group leader
    # needs a VAE copy on the accelerator because only that rank saves output.
    pipeline.vae.to(device="cpu")
if is_main_process and use_effective_sp:
    print("[SP] VAE decode enabled on one leader per SP group")

nfpb = getattr(config, "num_frame_per_block", 8)
num_blocks = config.num_output_frames // nfpb
dataset = PromptDataset(
    data_path=config.data_path,
    num_blocks=num_blocks,
)
if is_main_process:
    print(f"[data] data_path={config.data_path}, mode={dataset._mode}, num_blocks={num_blocks}")
num_prompts = len(dataset)
if use_multi_dp:
    # DP groups use independent SP collectives, so uneven prompt counts are safe.
    # A strided sampler avoids dropping or padding samples when len(dataset) is
    # not divisible by the number of DP groups (for example, VBench-mini has 43).
    sampler = range(dp_rank, num_prompts, total_dp_groups)
elif dist.is_initialized():
    sampler = SequentialSampler(dataset)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(
    dataset, batch_size=1, sampler=sampler, num_workers=0,
    drop_last=False, collate_fn=prompt_collate_fn,
)

if is_main_process:
    os.makedirs(config.output_folder, exist_ok=True)
if dist.is_initialized():
    dist.barrier()

save_latents_only = section_get(
    config,
    "inference",
    "save_latents_only",
    getattr(config, "save_latents_only", getattr(config, "save_latent_only", False)),
    aliases=("save_latent_only", "return_latents"),
)

for i, batch_data in tqdm(enumerate(dataloader), disable=not is_main_process):
    idx = batch_data["idx"].item()
    block_prompts = list(batch_data["prompts"][0])
    if len(block_prompts) < num_blocks:
        block_prompts += [block_prompts[-1]] * (num_blocks - len(block_prompts))
    elif len(block_prompts) > num_blocks:
        block_prompts = block_prompts[:num_blocks]
    prompt = block_prompts[0]
    prompts = [block_prompts] * config.num_samples

    shape = config.image_or_video_shape
    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, shape[2], shape[3], shape[4]],
        device=device,
        dtype=torch.bfloat16,
    )
    if use_effective_sp:
        src = dist.get_global_rank(sp_group, 0)
        dist.broadcast(sampled_noise, src=src, group=sp_group)

    if is_main_process:
        print(f"\n[SP] Generating video {idx}: {prompt[:60]}...")
    reset_peak_memory(device)
    if dedicated_vae_device is not None:
        reset_peak_memory(dedicated_vae_device)
    synchronize_accelerator(device)
    generation_started = time.perf_counter()
    generated = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=save_latents_only or not decode_on_this_rank,
    )
    synchronize_accelerator(device)
    generation_seconds = time.perf_counter() - generation_started
    generation_peak_memory_gb = peak_memory_gb(device)
    vae_peak_memory_gb = (
        peak_memory_gb(dedicated_vae_device)
        if dedicated_vae_device is not None
        else None
    )

    should_save = (sp_rank == 0) if use_effective_sp else True
    if idx < num_prompts and should_save:
        save_started = time.perf_counter()
        if getattr(pipeline, "is_lora_merged", False):
            model_type = "merged_lora"
        elif getattr(pipeline, "is_lora_enabled", False):
            model_type = "lora"
        elif getattr(config, "use_ema", False):
            model_type = "ema"
        else:
            model_type = "regular"
        mode = f"dp{dp_rank}_sp{effective_sp_size}" if use_multi_dp else f"sp{effective_sp_size}"
        if save_latents_only:
            latents = generated
        else:
            current_video = rearrange(generated, "b t c h w -> b t h w c").cpu()
            video = 255.0 * current_video
            if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
                pipeline.vae.model.clear_cache()

        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                base_name = f"rank{rank}-{idx}-{seed_idx}_{model_type}_{mode}"
            else:
                base_name = f"rank{rank}-{prompt[:100]}-{seed_idx}_{model_type}_{mode}"
            if save_latents_only:
                torch.save(latents[seed_idx].cpu(), os.path.join(config.output_folder, f"{base_name}.pt"))
            else:
                output_path = os.path.join(config.output_folder, f"{base_name}.mp4")
                fps = 24 if "5B" in config.model_kwargs.model_name else 16
                write_video(output_path, video[seed_idx], fps=fps)
                if is_main_process:
                    print(f"[SP] Saved: {output_path}")
            save_prompts_to_txt(
                prompts[seed_idx] if isinstance(prompts[seed_idx], list) else [prompts[seed_idx]],
                os.path.join(config.output_folder, f"{base_name}_prompts.txt"),
                is_main_process=is_main_process,
            )

        save_seconds = time.perf_counter() - save_started
        if save_latents_only:
            frame_count = 0
            video_seconds = 0.0
            generation_fps = 0.0
            real_time_factor = 0.0
        else:
            frame_count = int(generated.shape[1])
            output_fps = 24 if "5B" in config.model_kwargs.model_name else 16
            video_seconds = frame_count / output_fps
            generation_fps = frame_count / generation_seconds
            real_time_factor = generation_seconds / video_seconds
        peak_memory_text = (
            f"{generation_peak_memory_gb:.2f}"
            if generation_peak_memory_gb is not None
            else "n/a"
        )
        vae_peak_memory_text = (
            f"{vae_peak_memory_gb:.2f}" if vae_peak_memory_gb is not None else "n/a"
        )
        print(
            f"[benchmark] rank={rank} prompt_index={idx} "
            f"generation_seconds={generation_seconds:.3f} "
            f"save_seconds={save_seconds:.3f} pixel_frames={frame_count} "
            f"video_seconds={video_seconds:.3f} generation_fps={generation_fps:.3f} "
            f"rtf={real_time_factor:.3f} peak_memory_gb={peak_memory_text} "
            f"vae_peak_memory_gb={vae_peak_memory_text}"
        )

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
