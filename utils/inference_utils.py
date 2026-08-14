# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint and media helpers for the supported BF16 inference path."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import torch


def is_sla_linear_parameter(name: str) -> bool:
    """Return whether a parameter belongs to an SLA compensation projection."""
    return name.startswith("sla_linear.") or ".sla_linear." in name


def canonical_generator_parameter_name(name: str) -> str:
    """Remove FSDP, wrapper, and PEFT prefixes from a generator parameter name."""
    return (
        str(name)
        .replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_orig_mod.", "")
        .removeprefix("model.")
        .replace("base_model.model.", "")
    )


def configure_generator_linear_only(generator_model) -> list[str]:
    """Freeze a generator and enable gradients only for SLA compensation."""
    generator_model.requires_grad_(False)
    trainable = []
    for name, parameter in generator_model.named_parameters():
        if is_sla_linear_parameter(name):
            parameter.requires_grad_(True)
            trainable.append(name)
    if not trainable:
        raise ValueError("linear_only training found no sla_linear parameters")
    return trainable


def enable_generator_linear_parameters(generator_model) -> list[str]:
    """Enable raw SLA compensation parameters alongside an existing LoRA adapter."""
    trainable = []
    for name, parameter in generator_model.named_parameters():
        if is_sla_linear_parameter(name):
            parameter.requires_grad_(True)
            trainable.append(name)
    if not trainable:
        raise ValueError("lora_plus_linear training found no sla_linear parameters")
    return trainable


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def clean_fsdp_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove FSDP/checkpoint/compile wrapper prefixes from state dict keys."""
    return {
        str(key)
        .replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_orig_mod.", ""): value
        for key, value in state_dict.items()
    }


def extract_generator_state_dict(
    checkpoint: object,
    *,
    use_ema: bool = False,
) -> Mapping[str, torch.Tensor]:
    """Extract a generator state dict from supported LongLive layouts."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Generator checkpoint must be a mapping, got {type(checkpoint).__name__}")

    if use_ema:
        if "generator_ema" not in checkpoint:
            raise KeyError("use_ema=true, but the checkpoint has no 'generator_ema' entry")
        state_dict = checkpoint["generator_ema"]
    elif "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "generator_lora" in checkpoint:
        raise ValueError(
            "The selected checkpoint contains only LoRA weights. Set checkpoints.lora_ckpt "
            "and provide a full checkpoints.generator_ckpt."
        )
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Generator state dict is empty or has an unsupported layout")
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError(
            "Unrecognized checkpoint layout: expected a tensor state dict or one of "
            "'generator', 'generator_ema', and 'model'"
        )
    return clean_fsdp_state_dict_keys(state_dict) if use_ema else state_dict


def load_generator_checkpoint(
    generator,
    checkpoint_path: str,
    *,
    use_ema: bool = False,
    strict: bool = True,
):
    """Load a LongLive generator checkpoint into ``generator``."""
    checkpoint = _torch_load(checkpoint_path)
    state_dict = extract_generator_state_dict(checkpoint, use_ema=use_ema)
    return load_generator_state_dict(generator, state_dict, strict=strict)


def load_generator_state_dict(
    generator,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
):
    """Load a generator while initializing newly introduced SLA projections.

    Dense LongLive checkpoints predate SLA and therefore do not contain the
    per-layer ``sla_linear`` weights. Their mathematically neutral
    initialization is zero, so synthesize only those missing entries and keep
    strict loading for every other model parameter.
    """
    expected = generator.state_dict()
    prepared = dict(state_dict)
    for key, value in expected.items():
        if key not in prepared and (
            key.startswith("sla_linear.") or ".sla_linear." in key
        ):
            prepared[key] = torch.zeros_like(value)
    return generator.load_state_dict(prepared, strict=strict)


def load_lora_state_dict(
    lora_ckpt_path: str,
    *,
    expected_sparse_method: str | None = None,
) -> Mapping[str, torch.Tensor]:
    """Load a LoRA checkpoint, unwrapping ``generator_lora`` when present."""
    checkpoint = _torch_load(lora_ckpt_path)
    if isinstance(checkpoint, Mapping) and "generator_lora" in checkpoint:
        if expected_sparse_method is not None:
            from utils.training_state import validate_sparse_checkpoint_method

            validate_sparse_checkpoint_method(checkpoint, expected_sparse_method)
        return checkpoint["generator_lora"]
    if expected_sparse_method is not None and isinstance(checkpoint, Mapping):
        from utils.training_state import validate_sparse_checkpoint_method

        validate_sparse_checkpoint_method(checkpoint, expected_sparse_method)
        raise ValueError(
            "method-labeled LoRA checkpoints must contain generator_lora"
        )
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError(f"LoRA checkpoint has an unsupported layout: {lora_ckpt_path}")
    return checkpoint


def load_generator_linear_state_dict(
    generator_model, state_dict: Mapping[str, torch.Tensor]
) -> None:
    """Strictly load the sparse linear-compensation tensors into a generator."""
    expected = {
        canonical_generator_parameter_name(name): parameter
        for name, parameter in generator_model.named_parameters()
        if is_sla_linear_parameter(name)
    }
    prepared = {
        canonical_generator_parameter_name(name): value
        for name, value in state_dict.items()
    }
    missing = sorted(set(expected) - set(prepared))
    unexpected = sorted(set(prepared) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "generator_linear keys do not match model: "
            f"missing={missing[:4]} unexpected={unexpected[:4]}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(prepared[name].to(parameter))


def load_generator_linear_checkpoint(
    generator_model,
    checkpoint_path: str,
    *,
    expected_sparse_method: str,
) -> None:
    checkpoint = _torch_load(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("linear checkpoint must be a mapping")
    from utils.training_state import validate_sparse_checkpoint_method

    validate_sparse_checkpoint_method(checkpoint, expected_sparse_method)
    scope = checkpoint.get("generator_train_scope")
    if scope not in {"linear_only", "lora_plus_linear"}:
        raise ValueError(
            "checkpoint generator_train_scope must be linear_only or lora_plus_linear"
        )
    state_dict = checkpoint.get("generator_linear")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint is missing generator_linear")
    load_generator_linear_state_dict(generator_model, state_dict)


def cpu_state_dict(module) -> dict[str, torch.Tensor]:
    """Return a detached CPU state dict suitable for portable checkpoints."""
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def apply_and_merge_lora(
    pipeline,
    config,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
):
    """Load and merge an optional LoRA adapter into the BF16 generator."""
    adapter_cfg = getattr(config, "adapter", None)
    lora_ckpt = getattr(config, "lora_ckpt", None)
    if adapter_cfg is None or not lora_ckpt:
        return False

    from utils.lora_utils import (
        configure_lora_for_model,
        set_peft_model_state_dict_for_ulysses,
    )

    if device is not None:
        pipeline.generator.to(device=torch.device(device), dtype=dtype)
    else:
        pipeline.generator.to(dtype=dtype)

    if verbose:
        print(f"[LoRA] Wrapping generator with adapter config: {adapter_cfg}")
    sparse_config = getattr(getattr(config, "model_kwargs", None), "sparse_config", {})
    sparse_method = sparse_config.get(
        "method", "hsa_cag" if "keep_frames" in sparse_config else "sla_cag"
    )
    generator_train_scope = str(getattr(config, "generator_train_scope", "lora"))
    if generator_train_scope == "linear_only":
        load_generator_linear_checkpoint(
            pipeline.generator.model,
            lora_ckpt,
            expected_sparse_method=sparse_method,
        )
        pipeline.generator.to(dtype=dtype)
        pipeline.generator.eval().requires_grad_(False)
        pipeline.is_lora_enabled = False
        pipeline.is_lora_merged = False
        return True
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=adapter_cfg,
        is_main_process=verbose,
    )

    if verbose:
        print(f"[LoRA] Loading LoRA weights from: {lora_ckpt}")
    lora_state = load_lora_state_dict(
        lora_ckpt, expected_sparse_method=sparse_method
    )
    set_peft_model_state_dict_for_ulysses(pipeline.generator.model, lora_state)

    if verbose:
        print("[LoRA] Merging LoRA delta into base weights (merge_and_unload)...")
    pipeline.generator.model = pipeline.generator.model.merge_and_unload(safe_merge=True)
    if generator_train_scope == "lora_plus_linear":
        load_generator_linear_checkpoint(
            pipeline.generator.model,
            lora_ckpt,
            expected_sparse_method=sparse_method,
        )
    pipeline.generator.model.eval().requires_grad_(False)
    pipeline.is_lora_enabled = False
    pipeline.is_lora_merged = True
    return True


def place_vae_for_streaming(pipeline, config) -> torch.device | None:
    """Move ``pipeline.vae`` to ``config.vae_device`` for streaming-pipeline decode.

    Only acts when both ``streaming_vae`` and ``vae_device`` are set; otherwise
    leaves the VAE on whatever device the rest of the pipeline already uses.
    Mirrors the relocation done in ``inference.py`` so that quick-start scripts
    can opt in to the streaming-pipeline VAE simply by enabling those config
    fields.
    """
    if not bool(getattr(config, "streaming_vae", False)):
        return None
    vae_device_str = getattr(config, "vae_device", None)
    if not vae_device_str:
        return None

    vae_device = torch.device(vae_device_str)
    pipeline.vae.to(device="cpu")
    pipeline.vae.to(device=vae_device)
    if hasattr(pipeline.vae, "mean"):
        pipeline.vae.mean = pipeline.vae.mean.to(device=vae_device)
        pipeline.vae.std = pipeline.vae.std.to(device=vae_device)
    return vae_device


def prepare_single_prompt_inputs(
    config,
    prompt: str,
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
    generator: torch.Generator | None = None,
):
    """Create the per-block prompt list and latent noise for one text prompt."""
    num_frames = int(getattr(config, "num_output_frames", config.image_or_video_shape[1]))
    frames_per_block = int(getattr(config, "num_frame_per_block", 1))
    if num_frames % frames_per_block != 0:
        raise ValueError(f"num_frames={num_frames} must be divisible by num_frame_per_block={frames_per_block}")

    latent_shape = list(config.image_or_video_shape[2:])
    if len(latent_shape) != 3:
        raise ValueError(f"Expected latent shape [C, H, W], got {latent_shape}")

    num_blocks = num_frames // frames_per_block
    prompts = [[prompt] * num_blocks for _ in range(batch_size)]
    noise = torch.randn(
        [batch_size, num_frames, *latent_shape],
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return noise, prompts


def video_to_uint8(video: torch.Tensor) -> torch.Tensor:
    """Convert a generated video tensor from [T, C, H, W] or [1, T, C, H, W] to uint8 THWC."""
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("video_to_uint8 expects a single sample when a batch dimension is present.")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor with 4 dims, got shape={tuple(video.shape)}")
    if video.shape[1] in (1, 3):
        from einops import rearrange

        video = rearrange(video, "t c h w -> t h w c")
    return (255.0 * video.cpu()).clamp(0, 255).to(torch.uint8)


def save_video(video: torch.Tensor, output_path: str | os.PathLike, *, fps: int = 24) -> None:
    """Save a generated LongLive video tensor as an mp4 file."""
    from torchvision.io import write_video

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(output_path), video_to_uint8(video), fps=fps)
