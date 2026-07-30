#!/usr/bin/env python3
"""Native Wan2.2-TI2V-5B BF16 inference with SP/DP parallelism."""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.io import write_video

from utils.device import distributed_backend, set_device, synchronize
from utils.misc import set_seed
from utils.parallel_layout import validate_sp_dp_layout
from wan_5b import WanTI2V
from wan_5b.configs import WAN_CONFIGS
from wan_5b.distributed.sp_training import set_sequence_parallel_group


def _peak_memory_gb(device: torch.device) -> float | None:
    accelerator = getattr(torch, device.type, None)
    getter = getattr(accelerator, "max_memory_allocated", None)
    if getter is None:
        return None
    try:
        return getter(device) / (1024**3)
    except (RuntimeError, TypeError):
        return None


def _reset_peak_memory(device: torch.device) -> None:
    accelerator = getattr(torch, device.type, None)
    reset = getattr(accelerator, "reset_peak_memory_stats", None)
    if reset is not None:
        try:
            reset(device)
        except (RuntimeError, TypeError):
            pass


def _load_prompts(path: str) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _create_parallel_groups(world_size: int, rank: int, sp_size: int, dp_size: int):
    validate_sp_dp_layout(
        world_size=world_size,
        sp_size=sp_size,
        dp_size=dp_size,
        num_heads=24,
    )

    groups = []
    selected = None
    for dp_rank in range(dp_size):
        ranks = list(range(dp_rank * sp_size, (dp_rank + 1) * sp_size))
        group = dist.new_group(ranks=ranks)
        groups.append(group)
        if rank in ranks:
            selected = (group, dp_rank, ranks.index(rank))
    if selected is None:
        raise RuntimeError(f"rank {rank} was not assigned to an SP group")
    return selected


def _video_to_uint8(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4 or video.shape[0] not in (1, 3):
        raise ValueError(f"Expected Wan video [C,T,H,W], got {tuple(video.shape)}")
    return (
        ((video.float().clamp(-1, 1) + 1.0) * 127.5)
        .permute(1, 2, 3, 0)
        .to(torch.uint8)
        .cpu()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=distributed_backend(), rank=rank, world_size=world_size
        )

    sp_size = int(config.sp_size)
    dp_size = int(config.dp_size)
    if world_size > 1:
        sp_group, dp_rank, sp_rank = _create_parallel_groups(
            world_size, rank, sp_size, dp_size
        )
    else:
        if sp_size != 1 or dp_size != 1:
            raise ValueError("single-process inference requires sp_size=1 and dp_size=1")
        sp_group, dp_rank, sp_rank = None, 0, 0
    set_sequence_parallel_group(sp_group)

    model_root = str(config.model_kwargs.model_root)
    model_name = str(config.model_kwargs.model_name)
    if model_name != "Wan2.2-TI2V-5B":
        raise ValueError(f"Native baseline only supports Wan2.2-TI2V-5B, got {model_name}")
    if not Path(model_root).is_dir():
        raise FileNotFoundError(f"Wan2.2 model_root does not exist: {model_root}")
    if int(config.get("num_samples", 1)) != 1:
        raise ValueError("Native Wan VBench inference requires num_samples=1")
    frame_num = int(config.num_output_frames)
    if frame_num <= 0 or (frame_num - 1) % 4 != 0:
        raise ValueError(f"num_output_frames must satisfy 4n+1, got {frame_num}")
    width = int(config.data.width)
    height = int(config.data.height)
    if width % 16 or height % 16:
        raise ValueError(f"width/height must be divisible by 16, got {width}x{height}")

    seed = int(config.logging.seed)
    set_seed(seed)
    torch.set_grad_enabled(False)
    wan_config = copy.deepcopy(WAN_CONFIGS["ti2v-5B"])
    wan_config.param_dtype = torch.bfloat16
    wan_config.t5_dtype = torch.bfloat16

    if rank == 0:
        print(
            f"[Wan2.2] world={world_size}, layout=SP{sp_size}xDP{dp_size}, "
            f"frames={frame_num}, size={width}x{height}, "
            f"steps={int(config.inference.sampling_steps)}, cfg={float(config.inference.guidance_scale)}"
        )

    pipeline = WanTI2V(
        config=wan_config,
        checkpoint_dir=model_root,
        device_id=local_rank,
        rank=sp_rank,
        use_sp=sp_size > 1,
        t5_cpu=bool(config.inference.t5_cpu),
        init_on_cpu=False,
        convert_model_dtype=True,
        device=device,
        load_vae=sp_rank == 0,
        process_group=sp_group,
    )

    prompts = _load_prompts(str(config.data.data_path))
    output_folder = Path(str(config.output_folder))
    if rank == 0:
        output_folder.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    assigned_indices = range(dp_rank, len(prompts), dp_size)
    inference_iter = int(config.get("inference_iter", -1))
    for local_index, prompt_index in enumerate(assigned_indices):
        prompt = prompts[prompt_index]
        if sp_rank == 0:
            print(f"[Wan2.2] Generating video {prompt_index}: {prompt[:80]}...", flush=True)
        _reset_peak_memory(device)
        synchronize(device)
        started = time.perf_counter()
        video = pipeline.generate(
            input_prompt=prompt,
            img=None,
            size=(width, height),
            frame_num=frame_num,
            shift=float(config.model_kwargs.timestep_shift),
            sample_solver=str(config.inference.sample_solver),
            sampling_steps=int(config.inference.sampling_steps),
            guide_scale=float(config.inference.guidance_scale),
            n_prompt=str(config.inference.get("negative_prompt", "") or ""),
            seed=seed,
            offload_model=bool(config.inference.offload_model),
        )
        synchronize(device)
        generation_seconds = time.perf_counter() - started

        if sp_rank == 0:
            output_path = output_folder / (
                f"rank{rank}-{prompt_index}-0_wan22_bf16_dp{dp_rank}_sp{sp_size}.mp4"
            )
            save_started = time.perf_counter()
            write_video(
                str(output_path),
                _video_to_uint8(video),
                fps=int(config.inference.fps),
            )
            save_seconds = time.perf_counter() - save_started
            video_seconds = frame_num / float(config.inference.fps)
            peak = _peak_memory_gb(device)
            peak_text = f"{peak:.2f}" if peak is not None else "n/a"
            print(
                f"[benchmark] rank={rank} prompt_index={prompt_index} "
                f"generation_seconds={generation_seconds:.3f} "
                f"save_seconds={save_seconds:.3f} pixel_frames={frame_num} "
                f"video_seconds={video_seconds:.3f} "
                f"generation_fps={frame_num / generation_seconds:.3f} "
                f"rtf={generation_seconds / video_seconds:.3f} "
                f"peak_memory_gb={peak_text}",
                flush=True,
            )
        if inference_iter >= 0 and local_index >= inference_iter:
            break

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
