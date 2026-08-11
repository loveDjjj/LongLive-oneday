#!/usr/bin/env python3
"""Resolve compact inference presets into runtime configs and JSON metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return OmegaConf.load(path)


def _repo_file(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _prompt_count(path_value: str) -> int:
    path = _repo_file(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if count == 0:
        raise ValueError(f"prompt file is empty: {path}")
    return count


def _require_preset(config, name: str):
    if name not in config.presets:
        choices = ", ".join(config.presets.keys())
        raise KeyError(f"unknown preset {name!r}; choose one of: {choices}")
    return config.presets[name]


def _model_root(value: str) -> str:
    return os.environ.get("LONGLIVE_MODEL_ROOT", value)


def _generator_checkpoint(value: str) -> str:
    return os.environ.get("LONGLIVE_GENERATOR_CKPT", value)


def _save(config: dict, output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), output)


def _sparse_model_config(sparsity) -> dict | None:
    method_override = os.environ.get("LONGLIVE_SPARSE_METHOD", "").strip()
    enabled = bool(sparsity.enabled) or bool(method_override)
    if not enabled:
        return None
    method = method_override or str(sparsity.method)
    if method != "sla_cag":
        raise ValueError(f"unsupported sparse attention method={method!r}")
    options = OmegaConf.to_container(sparsity.options, resolve=True)
    return {"enabled": True, **options}


def _sparse_method(sparsity) -> str:
    method_override = os.environ.get("LONGLIVE_SPARSE_METHOD", "").strip()
    if method_override:
        return method_override
    return str(sparsity.method) if bool(sparsity.enabled) else "dense"


def resolve_msprof(args) -> dict:
    config = _load(args.config)
    preset = _require_preset(config, args.preset)
    runtime = config.runtime
    generation = config.generation
    measurement = config.measurement
    frames = int(preset.latent_frames)
    sp_size = int(runtime.sp_size)
    dp_size = int(runtime.dp_size)
    nproc = sp_size * dp_size
    vae_mode = str(runtime.vae_mode)
    if vae_mode not in {"sync", "async_dedicated"}:
        raise ValueError(f"unsupported vae_mode={vae_mode!r}")
    if vae_mode == "async_dedicated" and dp_size != 1:
        raise ValueError("async_dedicated VAE requires dp_size=1")

    num_prompts = int(measurement.num_prompts)
    available_prompts = _prompt_count(str(measurement.prompts))
    if not 1 <= num_prompts <= available_prompts:
        raise ValueError(
            f"measurement.num_prompts={num_prompts} but {measurement.prompts} "
            f"contains {available_prompts} prompts"
        )
    warmup = int(measurement.warmup_per_rank)
    if warmup >= num_prompts:
        raise ValueError("warmup_per_rank must be smaller than num_prompts")

    async_vae = vae_mode == "async_dedicated"
    vae_device = f"npu:{nproc}" if async_vae else None
    output_folder = args.output_folder or "videos/msprof"
    model = config.model
    model_kwargs = {
        "model_name": str(model.name),
        "model_root": _model_root(str(model.root)),
        "timestep_shift": float(model.timestep_shift),
        "num_frame_per_block": int(model.num_frame_per_block),
        "local_attn_size": int(model.local_attn_size),
    }
    sparse_config = _sparse_model_config(config.sparsity)
    if sparse_config is not None:
        model_kwargs["sparse_config"] = sparse_config
    resolved = {
        "model_kwargs": model_kwargs,
        "sp_size": sp_size,
        "dp_size": dp_size,
        "auto_sp_remainder": False,
        "model_num_heads": int(runtime.model_num_heads),
        "use_ema": False,
        "output_folder": output_folder,
        "num_samples": 1,
        "save_latents_only": False,
        "save_with_index": True,
        "inference_iter": num_prompts - 1,
        "num_output_frames": frames,
        "data": {
            "data_path": str(measurement.prompts),
            "image_or_video_shape": [
                1,
                frames,
                int(generation.latent_channels),
                int(generation.latent_height),
                int(generation.latent_width),
            ],
        },
        "inference": {
            "sampling_steps": int(generation.sampling_steps),
            "sink_size": int(generation.sink_size),
            "guidance_scale": float(generation.guidance_scale),
            "multi_shot_sink": bool(generation.multi_shot_sink),
            "multi_shot_rope_offset": int(generation.multi_shot_rope_offset),
            "kv_quant": False,
            "streaming_vae": async_vae,
            "async_vae": async_vae,
            "vae_type": "wan",
            "vae_device": vae_device,
        },
        "checkpoints": {
            "generator_ckpt": _generator_checkpoint(str(model.generator_checkpoint)),
            "lora_ckpt": None,
        },
        "model_quant": False,
        "sparsity": OmegaConf.to_container(config.sparsity, resolve=True),
        "logging": {"seed": int(measurement.seed)},
    }
    _save(resolved, args.output)

    pixel_frames = (frames - 1) * 4 + 1
    sparse_method = _sparse_method(config.sparsity)
    metadata = {
        "task": "msprof",
        "engine": "longlive2",
        "preset": args.preset,
        "latent_frames": frames,
        "pixel_frames": pixel_frames,
        "video_seconds": pixel_frames / 24.0,
        "sp_size": sp_size,
        "dp_size": dp_size,
        "nproc_per_node": nproc,
        "vae_mode": vae_mode,
        "required_devices": nproc + int(async_vae),
        "num_prompts": num_prompts,
        "warmup_per_rank": warmup,
        "sparsity_method": sparse_method,
        "run_tag": f"msprof-longlive2-{args.preset}-{vae_mode.replace('_dedicated', '')}-sp{sp_size}-dp{dp_size}-{sparse_method}",
        "msprof": OmegaConf.to_container(config.msprof, resolve=True),
    }
    return metadata


def resolve_vbench(args) -> dict:
    config = _load(args.config)
    preset = _require_preset(config, args.preset)
    defaults = config.defaults
    engine_name = str(preset.engine)
    category = str(preset.category)
    subset = str(preset.subset)
    engine = config.models[engine_name]
    dataset = config.datasets[category][subset]
    pixel_frames = int(preset.get("pixel_frames", defaults.pixel_frames))
    fps = int(preset.get("fps", defaults.fps))
    sp_size = int(preset.get("sp_size", defaults.sp_size))
    dp_size = int(preset.get("dp_size", defaults.dp_size))
    seeds = list(preset.get("seeds", defaults.seeds))
    nproc = sp_size * dp_size

    generation_prompts = str(dataset.generation_prompts)
    naming_prompts = str(dataset.naming_prompts)
    full_info = str(dataset.full_info)
    prompt_count = _prompt_count(generation_prompts)
    if _prompt_count(naming_prompts) != prompt_count:
        raise ValueError("generation and naming prompt counts differ")
    if not _repo_file(full_info).is_file():
        raise FileNotFoundError(_repo_file(full_info))

    seed = int(args.seed if args.seed is not None else seeds[0])
    if seed not in seeds:
        raise ValueError(f"seed={seed} is not part of preset seeds={seeds}")
    output_folder = args.output_folder or "videos/vbench"
    model_root = _model_root(str(engine.model_root))

    if engine_name == "longlive2":
        if (pixel_frames - 1) % 4 != 0:
            raise ValueError("LongLive pixel_frames must satisfy (pixel_frames - 1) % 4 == 0")
        latent_frames = (pixel_frames - 1) // 4 + 1
        model_kwargs = {
            "model_name": str(engine.model_name),
            "model_root": model_root,
            "timestep_shift": float(engine.timestep_shift),
            "num_frame_per_block": int(engine.num_frame_per_block),
            "local_attn_size": int(engine.local_attn_size),
        }
        sparse_config = _sparse_model_config(config.sparsity)
        if sparse_config is not None:
            model_kwargs["sparse_config"] = sparse_config
        resolved = {
            "model_kwargs": model_kwargs,
            "sp_size": sp_size,
            "dp_size": dp_size,
            "auto_sp_remainder": False,
            "model_num_heads": 24,
            "use_ema": False,
            "output_folder": output_folder,
            "num_samples": 1,
            "save_latents_only": False,
            "save_with_index": True,
            "inference_iter": -1,
            "num_output_frames": latent_frames,
            "data": {
                "data_path": generation_prompts,
                "image_or_video_shape": [1, latent_frames, 48, 44, 80],
            },
            "inference": {
                "sampling_steps": int(engine.sampling_steps),
                "sink_size": int(engine.sink_size),
                "guidance_scale": float(engine.guidance_scale),
                "multi_shot_sink": bool(engine.multi_shot_sink),
                "multi_shot_rope_offset": int(engine.multi_shot_rope_offset),
                "kv_quant": False,
                "streaming_vae": False,
                "async_vae": False,
                "vae_type": "wan",
                "vae_device": None,
            },
            "checkpoints": {
                "generator_ckpt": _generator_checkpoint(str(engine.generator_checkpoint)),
                "lora_ckpt": None,
            },
            "model_quant": False,
            "sparsity": OmegaConf.to_container(config.sparsity, resolve=True),
            "logging": {"seed": seed},
        }
    elif engine_name == "wan22":
        if _sparse_model_config(config.sparsity) is not None:
            raise ValueError("SLA+CAG is implemented for LongLive2 causal inference only")
        latent_frames = None
        resolved = {
            "model_kwargs": {
                "model_name": str(engine.model_name),
                "model_root": model_root,
                "timestep_shift": float(engine.timestep_shift),
            },
            "sp_size": sp_size,
            "dp_size": dp_size,
            "output_folder": output_folder,
            "num_samples": 1,
            "save_with_index": True,
            "inference_iter": -1,
            "num_output_frames": pixel_frames,
            "data": {
                "data_path": generation_prompts,
                "width": int(engine.width),
                "height": int(engine.height),
            },
            "inference": {
                "sampling_steps": int(engine.sampling_steps),
                "sample_solver": str(engine.sample_solver),
                "guidance_scale": float(engine.guidance_scale),
                "negative_prompt": None,
                "fps": fps,
                "t5_cpu": False,
                "offload_model": False,
            },
            "sparsity": OmegaConf.to_container(config.sparsity, resolve=True),
            "logging": {"seed": seed},
        }
    else:
        raise ValueError(f"unsupported VBench engine={engine_name!r}")

    _save(resolved, args.output)
    sparse_method = _sparse_method(config.sparsity)
    metadata = {
        "task": "vbench",
        "preset": args.preset,
        "engine": engine_name,
        "entrypoint": str(engine.entrypoint),
        "category": category,
        "subset": subset,
        "pixel_frames": pixel_frames,
        "latent_frames": latent_frames,
        "fps": fps,
        "sp_size": sp_size,
        "dp_size": dp_size,
        "nproc_per_node": nproc,
        "required_devices": nproc,
        "seeds": seeds,
        "prompt_count": prompt_count,
        "generation_prompts": generation_prompts,
        "naming_prompts": naming_prompts,
        "full_info": full_info,
        "sparsity_method": sparse_method,
        "run_tag": f"vbench-{engine_name}-{category}-{subset}-{pixel_frames}f-{len(seeds)}seed-sp{sp_size}-dp{dp_size}-{sparse_method}",
    }
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("msprof", "vbench"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-folder")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = resolve_msprof(args) if args.kind == "msprof" else resolve_vbench(args)
    print(json.dumps(metadata, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
