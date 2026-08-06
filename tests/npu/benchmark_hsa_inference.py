#!/usr/bin/env python3
"""Benchmark dense attention and end-to-end HSA routing at the SP4 tail shape."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_5b.modules.sparse_attention import (
    SparseAttentionConfig,
    _history_block_indices,
    _portable_sparse_attention,
    hierarchical_sparse_attention,
    resolve_chunk_sparsity,
)
from wan_5b.modules.sparse_attention_ascend import (
    ascend_triton_available,
    ascend_triton_sparse_attention_blhd,
    ascend_triton_unavailable_reason,
)
from wan_5b.modules.sparse_attention_mindiesd import (
    mindiesd_available,
    mindiesd_sparse_attention_blhd,
    mindiesd_unavailable_reason,
)


def _parse_blocks(value: str) -> list[int]:
    blocks = [int(item) for item in value.split(",") if item.strip()]
    if not blocks or any(
        block <= 0 or 7040 % block or 28160 % block for block in blocks
    ):
        raise argparse.ArgumentTypeError(
            "every block must be positive and divide the 7040/28160 tail lengths"
        )
    return blocks


def _parse_positive_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _cann_version() -> str:
    roots = [
        os.environ.get("ASCEND_HOME_PATH"),
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/ascend-toolkit/latest",
    ]
    for root in filter(None, roots):
        for name in ("version.cfg", "version.info", "compiler/version.info"):
            try:
                lines = (Path(root) / name).read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if "version" in line.lower() and "=" in line:
                    return line.split("=", 1)[1].strip()
    return "unknown"


def _sparse_config(block: int, backend: str = "ascend_triton") -> dict:
    return {
        "enabled": True,
        "backend": backend,
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


def _build_lut(
    q: torch.Tensor, k: torch.Tensor, block: int, backend: str = "ascend_triton"
) -> torch.Tensor:
    config = SparseAttentionConfig.from_mapping(_sparse_config(block, backend))
    frame_seq = 880
    chunk_id = 23
    history_tokens = k.shape[1] - q.shape[1]
    history_frames = history_tokens // frame_seq
    b, _, heads, dim = q.shape
    q_count = q.shape[1] // block
    k_count = k.shape[1] // block
    history_block_count = history_tokens // block
    sparsity = resolve_chunk_sparsity(config, chunk_id)
    history_keep = max(1, math.ceil((1.0 - sparsity) * history_block_count))

    q_blocks = q.reshape(b, q_count, block, heads, dim).mean(dim=2)
    history_k = k[:, :history_tokens]
    k_block_means = history_k.reshape(
        b, history_block_count, block, heads, dim
    ).mean(dim=2).float()
    history_frame_keys = history_k.reshape(
        b, history_frames, frame_seq, heads, dim
    ).mean(dim=2).float()
    history_ids = _history_block_indices(
        q_blocks,
        k_block_means,
        history_k,
        history_frames,
        frame_seq,
        block,
        history_keep,
        config,
        history_frame_keys=history_frame_keys,
    )
    current_ids = torch.arange(history_block_count, k_count, device=q.device)
    current = current_ids.view(1, 1, 1, -1).expand(b, heads, q_count, -1)
    history_ids = torch.sort(history_ids, dim=-1).values
    return torch.cat([history_ids, current], dim=-1).contiguous()


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
    parser.add_argument(
        "--backend",
        choices=("mindiesd", "ascend_triton"),
        default="mindiesd",
    )
    parser.add_argument("--blocks", type=_parse_blocks, default=[128])
    parser.add_argument(
        "--native-query-batches",
        type=_parse_positive_list,
        default=[],
        help="also benchmark gathered native SDPA with these query-block batch sizes",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    if args.backend == "mindiesd":
        if not mindiesd_available():
            raise RuntimeError(mindiesd_unavailable_reason())
        if args.blocks != [128]:
            raise ValueError("MindIE-SD RainFusionAttention only supports block 128")
    elif not ascend_triton_available():
        raise RuntimeError(ascend_triton_unavailable_reason())

    device = torch.device(args.device)
    torch.npu.set_device(device)
    device_name = torch.npu.get_device_properties(device).name
    print(
        f"torch={torch.__version__} torch_npu={_package_version('torch-npu')} "
        f"mindiesd={_package_version('mindiesd')} cann={_cann_version()} "
        f"device={device_name}",
        flush=True,
    )
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
            config = _sparse_config(block, args.backend)
            try:
                route_median, route_min = _measure(
                    lambda: _build_lut(q, k, block, args.backend),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                block_lut = _build_lut(q, k, block, args.backend)
                selected_blocks = block_lut.shape[-1]
                kv_blocks = k.shape[1] // block
                effective_sparsity = 1.0 - selected_blocks / kv_blocks
                if args.backend == "mindiesd":
                    sparse_kernel = lambda: mindiesd_sparse_attention_blhd(
                        q, k, v, block_lut
                    )
                else:
                    sparse_kernel = lambda: ascend_triton_sparse_attention_blhd(
                        q,
                        k,
                        v,
                        block_lut,
                        block_q=block,
                        block_k=block,
                        validate_lut=False,
                    )
                kernel_median, kernel_min = _measure(
                    sparse_kernel,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                routing_cache = {}
                hsa_median, hsa_min = _measure(
                    lambda: hierarchical_sparse_attention(
                        q,
                        k,
                        v,
                        frame_seq=880,
                        chunk_id=23,
                        sparse_config=config,
                        routing_cache=routing_cache,
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                print(
                    f"hsa backend={args.backend} block={block} "
                    f"uncached_route_ms={route_median:.3f} "
                    f"uncached_route_min_ms={route_min:.3f} "
                    f"kernel_ms={kernel_median:.3f} kernel_min_ms={kernel_min:.3f} "
                    f"cached_full_ms={hsa_median:.3f} cached_full_min_ms={hsa_min:.3f} "
                    f"selected={selected_blocks}/{kv_blocks} "
                    f"effective_sparsity={effective_sparsity:.3f} "
                    f"speedup={dense_median / hsa_median:.3f}x",
                    flush=True,
                )
                for query_batch in args.native_query_batches:
                    native_median, native_min = _measure(
                        lambda: _portable_sparse_attention(
                            q,
                            k,
                            v,
                            block_lut,
                            block_q=block,
                            block_k=block,
                            query_block_batch=query_batch,
                            scale=None,
                        ),
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                    print(
                        f"native block={block} query_batch={query_batch} "
                        f"kernel_ms={native_median:.3f} min_ms={native_min:.3f} "
                        f"speedup={dense_median / (route_median + native_median):.3f}x",
                        flush=True,
                    )
            except Exception as error:
                message = str(error).splitlines()[0] if str(error) else "no detail"
                print(
                    f"hsa block={block} failed={type(error).__name__}: {message}",
                    flush=True,
                )
                torch.npu.empty_cache()


if __name__ == "__main__":
    main()
