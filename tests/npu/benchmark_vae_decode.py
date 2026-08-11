#!/usr/bin/env python3
"""Benchmark Wan2.2 VAE decode from a saved LongLive latent tensor."""

from __future__ import annotations

import argparse
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.device import create_event
from utils.wan_5b_wrapper import WanVAEWrapper


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _reset_peak_memory(device: torch.device) -> None:
    try:
        torch.npu.reset_peak_memory_stats(device)
    except TypeError:
        torch.npu.reset_peak_memory_stats()


def _peak_memory_gb(device: torch.device) -> float:
    try:
        value = torch.npu.max_memory_allocated(device)
    except TypeError:
        value = torch.npu.max_memory_allocated()
    return value / 1024**3


def _load_latent(path: Path) -> torch.Tensor:
    latent = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(latent):
        raise TypeError(f"expected a tensor in {path}, got {type(latent).__name__}")
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    if latent.ndim != 5:
        raise ValueError(
            "latent must have shape [T,C,H,W] or [B,T,C,H,W], "
            f"got {tuple(latent.shape)}"
        )
    if latent.shape[0] != 1 or latent.shape[2] != 48:
        raise ValueError(
            "Wan2.2 VAE benchmark requires batch 1 and 48 latent channels, "
            f"got {tuple(latent.shape)}"
        )
    return latent.permute(0, 2, 1, 3, 4).contiguous()


def _decode_cached_pipeline(
    model, latent, scale, chunk_frames: int
) -> tuple[torch.Tensor, float, float, float, float]:
    """Reproduce worker-side decode, pinned DtoH, and CPU postprocessing."""
    model.clear_cache()
    cpu_chunks = []
    vae_device_ms = 0.0
    dtoh_device_ms = 0.0
    decode_started = time.perf_counter()
    for start in range(0, latent.shape[2], chunk_frames):
        vae_start = create_event(latent.device, enable_timing=True)
        vae_end = create_event(latent.device, enable_timing=True)
        dtoh_end = create_event(latent.device, enable_timing=True)
        vae_start.record()
        decoded = model.cached_decode(
            latent[:, :, start : start + chunk_frames], scale
        ).float().clamp_(-1, 1)
        vae_end.record()
        pinned = torch.empty(
            decoded.shape,
            dtype=decoded.dtype,
            device="cpu",
            pin_memory=True,
        )
        pinned.copy_(decoded, non_blocking=True)
        dtoh_end.record()
        torch.npu.synchronize()
        vae_device_ms += vae_start.elapsed_time(vae_end)
        dtoh_device_ms += vae_end.elapsed_time(dtoh_end)
        cpu_chunks.append(pinned)
        del decoded
    decode_seconds = time.perf_counter() - decode_started

    post_started = time.perf_counter()
    output = torch.cat(cpu_chunks, dim=2).permute(0, 2, 1, 3, 4)
    output = (output * 0.5 + 0.5).clamp_(0, 1)
    post_seconds = time.perf_counter() - post_started
    model.clear_cache()
    return (
        output,
        decode_seconds,
        post_seconds,
        vae_device_ms / 1000.0,
        dtoh_device_ms / 1000.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        default=os.environ.get(
            "LONGLIVE_MODEL_ROOT", "/mnt/share/weight/Wan2.2-TI2V-5B"
        ),
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument(
        "--max-latent-frames",
        type=int,
        help="profile only the leading latent frames; defaults to the full tensor",
    )
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.chunk_frames <= 0 or args.iterations <= 0:
        raise ValueError("chunk-frames and iterations must be positive")
    if args.max_latent_frames is not None and args.max_latent_frames <= 0:
        raise ValueError("max-latent-frames must be positive")

    device = torch.device(args.device)
    torch.npu.set_device(device)
    latent = _load_latent(args.latent).to(device=device, dtype=torch.bfloat16)
    if args.max_latent_frames is not None:
        latent = latent[:, :, : args.max_latent_frames]
    if latent.shape[2] % args.chunk_frames:
        raise ValueError(
            f"latent frames {latent.shape[2]} are not divisible by "
            f"chunk-frames {args.chunk_frames}"
        )

    vae = WanVAEWrapper(model_root=args.model_root).eval().requires_grad_(False)
    vae.to(device=device, dtype=torch.bfloat16)
    scale = [
        vae.mean.to(device=device, dtype=latent.dtype),
        1.0 / vae.std.to(device=device, dtype=latent.dtype),
    ]
    print(
        f"torch={torch.__version__} torch_npu={_package_version('torch-npu')} "
        f"device={torch.npu.get_device_properties(device).name} "
        f"latent={tuple(latent.shape)} chunk_frames={args.chunk_frames}",
        flush=True,
    )

    with torch.no_grad():
        # Warm the first-chunk and steady-state decoder shapes without decoding
        # the full video twice.
        warmup_frames = min(latent.shape[2], 2 * args.chunk_frames)
        _decode_cached_pipeline(
            vae.model,
            latent[:, :, :warmup_frames],
            scale,
            args.chunk_frames,
        )
        torch.npu.synchronize()

        samples = []
        output = None
        for iteration in range(args.iterations):
            _reset_peak_memory(device)
            started = time.perf_counter()
            (
                output,
                decode_seconds,
                post_seconds,
                vae_device_seconds,
                dtoh_device_seconds,
            ) = _decode_cached_pipeline(
                vae.model, latent, scale, args.chunk_frames
            )
            elapsed = time.perf_counter() - started
            samples.append(elapsed)
            print(
                f"iteration={iteration} decode_seconds={elapsed:.3f} "
                f"decode_dtoh_seconds={decode_seconds:.3f} "
                f"cpu_post_seconds={post_seconds:.3f} "
                f"vae_device_seconds={vae_device_seconds:.3f} "
                f"dtoh_device_seconds={dtoh_device_seconds:.3f} "
                f"latent_fps={latent.shape[2] / elapsed:.3f} "
                f"pixel_fps={output.shape[1] / elapsed:.3f} "
                f"peak_memory_gb={_peak_memory_gb(device):.2f}",
                flush=True,
            )

    assert output is not None
    if not torch.isfinite(output).all().item():
        raise RuntimeError("VAE output contains non-finite values")
    ordered = sorted(samples)
    print(
        f"vae_cached_decode iterations={len(samples)} "
        f"median_seconds={ordered[len(ordered) // 2]:.3f} "
        f"min_seconds={min(samples):.3f} output_btchw={tuple(output.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
