#!/usr/bin/env python3
"""Benchmark dense attention and end-to-end HSA routing at the SP4 tail shape."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from wan_5b.modules.sparse_attention import hierarchical_sparse_attention
from wan_5b.modules.sparse_attention_ascend import (
    ascend_triton_available,
    ascend_triton_unavailable_reason,
)


def _parse_blocks(value: str) -> list[int]:
    blocks = [int(item) for item in value.split(",") if item.strip()]
    if not blocks or any(block <= 0 or 880 % block for block in blocks):
        raise argparse.ArgumentTypeError("every block must be positive and divide 880")
    return blocks


def _measure(operation, *, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        output = operation()
    torch.npu.synchronize()

    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        output = operation()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    if not torch.isfinite(output).all().item():
        raise RuntimeError("attention output contains non-finite values")
    return statistics.median(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--blocks", type=_parse_blocks, default=[40])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    if not ascend_triton_available():
        raise RuntimeError(ascend_triton_unavailable_reason())

    device = torch.device(args.device)
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu").manual_seed(11)
    shape_q = (1, 7040, 6, 128)
    shape_kv = (1, 28160, 6, 128)
    q = torch.randn(shape_q, generator=generator, dtype=torch.bfloat16).to(device)
    k = torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16).to(device)
    v = torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16).to(device)

    with torch.no_grad():
        dense_median, dense_min = _measure(
            lambda: F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(
            f"dense median_ms={dense_median:.3f} min_ms={dense_min:.3f} "
            f"q={shape_q} kv={shape_kv}",
            flush=True,
        )

        for block in args.blocks:
            config = {
                "enabled": True,
                "backend": "ascend_triton",
                "sparsity": 0.85,
                "sparsity_base": 0.95,
                "block_q": block,
                "block_k": block,
                "keep_frames": 6,
                "keep_sink": 1,
                "keep_near": 2,
                "dense_current": True,
                "min_sparse_history_frames": 2,
                "num_output_frames": 192,
                "num_frame_per_block": 8,
                "local_attn_size": 32,
            }
            hsa_median, hsa_min = _measure(
                lambda: hierarchical_sparse_attention(
                    q,
                    k,
                    v,
                    frame_seq=880,
                    chunk_id=23,
                    sparse_config=config,
                ),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            print(
                f"hsa block={block} median_ms={hsa_median:.3f} min_ms={hsa_min:.3f} "
                f"speedup={dense_median / hsa_median:.3f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
