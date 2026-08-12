# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf
from pathlib import Path


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


wan_default_config = {
    "Wan2.2-TI2V-5B": {
        "resolution": [1280, 704],
        "temporal_compression_ratio": 4,
        "spatial_compression_ratio": 16,
        "num_heads": 24,
        "head_dim": 128,
        "num_transformer_blocks": 30,
        "fps": 24,
    }
}


SECTION_KEYS = (
    "infra",
    "algorithm",
    "training",
    "data",
    "evaluation",
    "inference",
    "logging",
    "checkpoints",
)


def _set_once(config, key, value, source):
    if value is None:
        return
    if key in config and config[key] != value:
        raise ValueError(
            f"{key} is defined more than once with different values: "
            f"{config[key]} vs {value} from {source}."
        )
    config[key] = value


def section_get(config, section_key, key, default=None, aliases=()):
    """Read a grouped config value, falling back to legacy flat names."""
    section = config.get(section_key, None)
    candidate_keys = (key, *aliases)
    if section is not None:
        for candidate in candidate_keys:
            if candidate in section:
                return section[candidate]
    for candidate in candidate_keys:
        if candidate in config:
            return config[candidate]
    return default


def resolve_model_location(model_kwargs, default_name="Wan2.2-TI2V-5B"):
    """Return the shared model name/root used by DiT, T5, and VAE wrappers."""
    if model_kwargs is None:
        return default_name, None
    return (
        model_kwargs.get("model_name", default_name),
        model_kwargs.get("model_root", None),
    )


def normalize_config(config):
    """Expand grouped release configs into the flat runtime schema.

    The training and inference code historically reads fields such as
    ``config.batch_size`` and ``config.model_kwargs`` directly.  Release
    configs can group those fields for readability, then call this function at
    the entry point to preserve the existing runtime contract.
    """
    for section_key in SECTION_KEYS:
        section = config.get(section_key, None)
        if section is None:
            continue
        for key, value in section.items():
            config[key] = value

    evaluation = config.get("evaluation", None)
    if evaluation is not None:
        if "interval" in evaluation:
            _set_once(config, "generate_interval", evaluation.interval, "evaluation.interval")
            _set_once(config, "vis_interval", evaluation.interval, "evaluation.interval")
        if "num_frames" in evaluation:
            num_frames = evaluation.num_frames
            if isinstance(num_frames, (list, tuple)):
                vis_lengths = list(num_frames)
                inference_num_frames = vis_lengths[0] if vis_lengths else 0
            else:
                inference_num_frames = int(num_frames)
                vis_lengths = [inference_num_frames]
            _set_once(config, "inference_num_frames", inference_num_frames, "evaluation.num_frames")
            _set_once(config, "vis_video_lengths", vis_lengths, "evaluation.num_frames")
        if "use_ema" in evaluation:
            _set_once(config, "vis_ema", evaluation.use_ema, "evaluation.use_ema")

    model_section = config.get("model", None)
    base_model_kwargs = config.get("model_kwargs", None)
    model_kwargs = OmegaConf.create({})
    if base_model_kwargs is not None:
        model_kwargs = OmegaConf.merge(model_kwargs, base_model_kwargs)

    if model_section is not None:
        section_kwargs = model_section.get("kwargs", None)
        if section_kwargs is not None:
            model_kwargs = OmegaConf.merge(model_kwargs, section_kwargs)

        model_name = model_section.get("name", None)
        if model_name is not None:
            model_kwargs.model_name = model_name
            config.model_name = model_name

        _set_once(
            config,
            "num_frame_per_block",
            model_section.get("num_frame_per_block", None),
            "model.num_frame_per_block",
        )

    if "model_name" in config and "model_name" not in model_kwargs:
        model_kwargs.model_name = config.model_name
    if "timestep_shift" in config and "timestep_shift" not in model_kwargs:
        model_kwargs.timestep_shift = config.timestep_shift
    if "timestep_shift" in model_kwargs:
        _set_once(config, "timestep_shift", model_kwargs.timestep_shift, "model_kwargs.timestep_shift")

    model_num_frame_per_block = model_kwargs.get("num_frame_per_block", None)
    if model_num_frame_per_block is not None:
        _set_once(config, "num_frame_per_block", model_num_frame_per_block, "model_kwargs.num_frame_per_block")

    if len(model_kwargs) > 0:
        config.model_kwargs = model_kwargs

    if "wandb_host" not in config:
        config.wandb_host = "https://api.wandb.ai"

    if "negative_prompt" not in config:
        config.negative_prompt = DEFAULT_NEGATIVE_PROMPT

    if config.get("trainer", None) == "score_distillation":
        dmd_defaults = {
            "i2v": False,
            "teacher_forcing": False,
            "backward_simulation": True,
            "independent_first_frame": False,
            "num_train_timestep": 1000,
            "denoising_loss_type": "flow",
            "real_guidance_scale": 3.0,
            "fake_guidance_scale": 0.0,
        }
        for key, value in dmd_defaults.items():
            if key not in config:
                config[key] = value
        if "causal" not in config:
            config.causal = bool(config.get("all_causal", True))

    # Causal DMD uses the same Wan backbone for generator/teacher/critic unless
    # a role-specific override is explicitly provided.
    if getattr(config, "all_causal", False) and "model_kwargs" in config:
        for role_key in ("real_model_kwargs", "fake_model_kwargs"):
            if config.get(role_key, None) is None:
                config[role_key] = OmegaConf.create(
                    OmegaConf.to_container(config.model_kwargs, resolve=True)
                )

    return config


def validate_sparse_training_config(config):
    """Validate the maintained HSA/SLA sparse-training workflow."""
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    require(config.get("trainer") == "score_distillation", "algorithm.trainer must be score_distillation")
    require(config.get("distribution_loss") == "dmd", "algorithm.distribution_loss must be dmd")
    require(bool(config.get("all_causal", False)), "algorithm.all_causal must be true")
    require(not bool(config.get("i2v", False)), "current training is prompt-only; i2v must be false")
    require(bool(config.get("backward_simulation", True)), "backward_simulation must be true")
    require(bool(config.get("generator_is_causal", False)), "algorithm.generator_is_causal must be true")
    require(not bool(config.get("teacher_forcing", False)), "algorithm.teacher_forcing must be false")
    require(
        not bool(config.get("independent_first_frame", False)),
        "algorithm.independent_first_frame must be false",
    )
    require(int(config.get("batch_size", 0)) == 1, "training.batch_size must currently be 1")
    require(not config.get("real_score_ckpt"), "checkpoints.real_score_ckpt is not supported")
    require(not config.get("fake_score_ckpt"), "checkpoints.fake_score_ckpt is not supported")
    require(float(config.get("ema_weight", 0.0)) == 0.0, "training.ema_weight must be zero")
    require(config.get("denoising_loss_type") == "flow", "algorithm.denoising_loss_type must be flow")
    require(int(config.get("sampling_steps", 0)) == 4, "inference.sampling_steps must be 4")

    model_kwargs = config.get("model_kwargs", {})
    require(model_kwargs.get("model_name") == "Wan2.2-TI2V-5B", "only Wan2.2-TI2V-5B is supported")
    sparse = model_kwargs.get("sparse_config", {})
    require(bool(sparse.get("enabled", False)), "model_kwargs.sparse_config.enabled must be true")
    sparse_method = sparse.get("method")
    if sparse_method is None:
        sparse_method = "hsa_cag" if "keep_frames" in sparse else "sla_cag"
        sparse["method"] = sparse_method
    require(
        sparse_method in {"hsa_cag", "sla_cag"},
        "sparse method must be hsa_cag or sla_cag",
    )
    require(
        sparse.get("backend") in {"ascend_triton", "portable"},
        "sparse backend must be ascend_triton or portable",
    )
    sp_size = int(config.get("sequence_parallel_size", 0))
    # Patch embedding maps 44x80 latents to 22x40=880 tokens per frame.
    # Ulysses gathers the full sequence and shards heads, so SLA sees the same
    # chunk token count for every supported SP size.
    frame_tokens = 22 * 40
    block_frames = int(model_kwargs.get("num_frame_per_block", 0))
    chunk_tokens = frame_tokens * max(block_frames, 1)
    block_q = int(sparse.get("block_q", 0))
    block_k = int(sparse.get("block_k", 0))
    require(
        sp_size > 0 and chunk_tokens % sp_size == 0,
        f"sequence_parallel_size must divide {chunk_tokens} pre-exchange chunk tokens",
    )
    require(
        block_q > 0 and chunk_tokens % block_q == 0,
        f"sparse block_q must divide {chunk_tokens} tokens per chunk at SP{sp_size}",
    )
    require(
        block_k > 0 and chunk_tokens % block_k == 0,
        f"sparse block_k must divide {chunk_tokens} tokens per chunk at SP{sp_size}",
    )
    require(int(sparse.get("query_block_batch", 0)) > 0, "sparse query_block_batch must be positive")
    require(0.0 <= float(sparse.get("sparsity", -1.0)) < 1.0, "sparse sparsity must be in [0, 1)")
    require(0.0 <= float(sparse.get("sparsity_base", -1.0)) < 1.0, "sparse sparsity_base must be in [0, 1)")

    adapter = config.get("adapter", {})
    require(adapter.get("type") == "lora", "adapter.type must be lora")
    require(bool(adapter.get("apply_to_critic", False)), "adapter.apply_to_critic must be true")

    quant_flags = (
        "model_quant",
        "generator_quant",
        "real_score_quant",
        "fake_score_quant",
        "kv_quant",
    )
    require(
        not any(bool(config.get(key, False)) for key in quant_flags),
        "Ascend BF16 training requires all quantization flags to be false",
    )

    shape = list(config.get("image_or_video_shape", []))
    require(len(shape) == 5, "data.image_or_video_shape must contain [B,F,C,H,W]")
    if len(shape) == 5:
        latent_frames = int(shape[1])
        require(latent_frames == 32, "current training requires exactly 32 latent frames")
        require(int(shape[0]) == 1, "data.image_or_video_shape batch dimension must be 1")
        require(latent_frames == int(config.get("num_training_frames", -1)), "latent frames must equal training.num_training_frames")
        require(latent_frames == int(config.get("slice_last_frames", -1)), "latent frames must equal training.slice_last_frames")
        require(latent_frames == int(sparse.get("num_output_frames", -1)), "latent frames must equal sparse_config.num_output_frames")

    require(block_frames > 0 and 32 % block_frames == 0, "num_frame_per_block must divide 32 latent frames")
    require(int(sparse.get("num_frame_per_block", -1)) == block_frames, "sparse and model block sizes must match")
    require(int(sparse.get("local_attn_size", -1)) == int(model_kwargs.get("local_attn_size", -2)), "sparse and model local_attn_size must match")
    require(sp_size > 0 and 24 % sp_size == 0, "sequence_parallel_size must divide 24 attention heads")
    require(sp_size > 0 and block_frames % sp_size == 0, "sequence_parallel_size must divide num_frame_per_block")
    require(int(config.get("gradient_accumulation_steps", 0)) > 0, "training.gradient_accumulation_steps must be positive")
    require(int(config.get("max_iters", 0)) > 0, "training.max_iters must be positive")
    require(int(config.get("log_iters", 0)) > 0, "training.log_iters must be positive")

    for key, label in (
        ("data_path", "training prompt file"),
        ("eval_data_path", "validation prompt file"),
        ("generator_ckpt", "base generator checkpoint"),
    ):
        value = config.get(key)
        require(bool(value) and Path(str(value)).exists(), f"{label} does not exist: {value}")
    model_root = model_kwargs.get("model_root")
    require(bool(model_root) and Path(str(model_root)).is_dir(), f"model root does not exist: {model_root}")

    if errors:
        formatted = "\n".join(f"  - {message}" for message in errors)
        raise ValueError(f"Invalid sparse training configuration:\n{formatted}")
    return config


def validate_sla_cag_training_config(config):
    """Backward-compatible alias for older launchers and tests."""
    return validate_sparse_training_config(config)
