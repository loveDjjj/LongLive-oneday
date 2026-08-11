#!/usr/bin/env python3
"""Benchmark dense attention and complete SLA+CAG at the SP4 tail shape."""

from __future__ import annotations

import argparse
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

from wan_5b.modules.sla_attention import (
    SLAAttentionConfig,
    _linear_attention,
    _projected_linear_attention,
    build_sla_block_lut,
    resolve_chunk_sparsity,
    sla_cag_attention,
)
from wan_5b.modules.sla_attention_ascend import (
    ascend_triton_available,
    ascend_triton_sparse_attention_blhd,
    ascend_triton_unavailable_reason,
)
from wan_5b.modules.sla_attention_mindiesd import (
    mindiesd_bsa_available,
    mindiesd_bsa_sparse_attention_blhd,
    mindiesd_bsa_unavailable_reason,
    mindiesd_available,
    mindiesd_sparse_attention_blhd,
    mindiesd_unavailable_reason,
)


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


def _config(backend: str) -> SLAAttentionConfig:
    return SLAAttentionConfig.from_mapping(
        {
            "enabled": True,
            "backend": backend,
            "sparsity": 0.95,
            "sparsity_base": 0.97,
            "block_q": 128,
            "block_k": 128,
            "feature_map": "softmax",
            "keep_sink_frames": 1,
            "keep_recent_frames": 1,
            "dense_current": False,
            "min_sparse_history_frames": 1,
            "linear_cache": True,
            "num_output_frames": 192,
            "num_frame_per_block": 8,
            "local_attn_size": 32,
        }
    )


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
        choices=("mindiesd", "mindiesd_bsa", "ascend_triton"),
        default="mindiesd",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    if args.backend == "mindiesd" and not mindiesd_available():
        raise RuntimeError(mindiesd_unavailable_reason())
    if args.backend == "mindiesd_bsa" and not mindiesd_bsa_available():
        raise RuntimeError(mindiesd_bsa_unavailable_reason())
    if args.backend == "ascend_triton" and not ascend_triton_available():
        raise RuntimeError(ascend_triton_unavailable_reason())

    device = torch.device(args.device)
    torch.npu.set_device(device)
    print(
        f"torch={torch.__version__} torch_npu={_package_version('torch-npu')} "
        f"mindiesd={_package_version('mindiesd')} cann={_cann_version()} "
        f"device={torch.npu.get_device_properties(device).name}",
        flush=True,
    )
    generator = torch.Generator(device="cpu").manual_seed(11)
    shape_q = (1, 7040, 6, 128)
    shape_kv = (1, 28160, 6, 128)
    q = torch.randn(shape_q, generator=generator, dtype=torch.bfloat16).to(device)
    k = torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16).to(device)
    v = torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16).to(device)
    projection = torch.nn.Linear(128, 128, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(128, device=device, dtype=torch.bfloat16))
        projection.bias.zero_()

    config = _config(args.backend)
    chunk_id = 23
    sparsity = resolve_chunk_sparsity(config, chunk_id)
    with torch.no_grad():
        dense_median, dense_min = _measure(
            lambda: F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(
            f"dense median_ms={dense_median:.3f} min_ms={dense_min:.3f} "
            f"q={shape_q} kv={shape_kv}",
            flush=True,
        )

        route_median, route_min = _measure(
            lambda: build_sla_block_lut(
                q, k, frame_seq=880, sparsity=sparsity, config=config
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        router_cache = {}
        cached_route_median, cached_route_min = _measure(
            lambda: build_sla_block_lut(
                q,
                k,
                frame_seq=880,
                sparsity=sparsity,
                config=config,
                cache=router_cache,
                cache_token=chunk_id,
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        lut = build_sla_block_lut(
            q, k, frame_seq=880, sparsity=sparsity, config=config
        )
        if args.backend == "mindiesd":
            sparse_operation = lambda: mindiesd_sparse_attention_blhd(q, k, v, lut)
        elif args.backend == "mindiesd_bsa":
            sparse_operation = lambda: mindiesd_bsa_sparse_attention_blhd(
                q, k, v, lut
            )
        else:
            sparse_operation = lambda: ascend_triton_sparse_attention_blhd(
                q, k, v, lut, block_q=128, block_k=128, validate_lut=False
            )
        kernel_median, kernel_min = _measure(
            sparse_operation, warmup=args.warmup, iterations=args.iterations
        )

        explicit_linear_cache = {}

        def explicit_linear_operation():
            return projection(
                _linear_attention(
                    q,
                    k,
                    v,
                    history_tokens=k.shape[1] - q.shape[1],
                    chunk_id=chunk_id,
                    config=config,
                    cache=explicit_linear_cache,
                )
            )

        explicit_linear_median, explicit_linear_min = _measure(
            explicit_linear_operation,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        folded_linear_cache = {}

        def folded_linear_operation():
            return _projected_linear_attention(
                q,
                k,
                v,
                history_tokens=k.shape[1] - q.shape[1],
                chunk_id=chunk_id,
                config=config,
                cache=folded_linear_cache,
                projection=projection,
            )

        folded_linear_median, folded_linear_min = _measure(
            folded_linear_operation,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        linear_delta = (
            explicit_linear_operation().float() - folded_linear_operation().float()
        ).abs()

        attention_cache = {}
        full_median, full_min = _measure(
            lambda: sla_cag_attention(
                q,
                k,
                v,
                frame_seq=880,
                chunk_id=chunk_id,
                sparse_config=config,
                linear_projection=projection,
                attention_cache=attention_cache,
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        selected = lut.shape[-1]
        key_blocks = k.shape[1] // 128
        print(
            f"sla backend={args.backend} block=128 cag_sparsity={sparsity:.3f} "
            f"uncached_route_ms={route_median:.3f} uncached_route_min_ms={route_min:.3f} "
            f"cached_route_ms={cached_route_median:.3f} cached_route_min_ms={cached_route_min:.3f} "
            f"sparse_kernel_ms={kernel_median:.3f} sparse_kernel_min_ms={kernel_min:.3f} "
            f"explicit_linear_ms={explicit_linear_median:.3f} "
            f"explicit_linear_min_ms={explicit_linear_min:.3f} "
            f"folded_linear_ms={folded_linear_median:.3f} "
            f"folded_linear_min_ms={folded_linear_min:.3f} "
            f"linear_projection_speedup={explicit_linear_median / folded_linear_median:.3f}x "
            f"linear_max_abs={linear_delta.max().item():.6f} "
            f"linear_mean_abs={linear_delta.mean().item():.6f} "
            f"cached_full_ms={full_median:.3f} cached_full_min_ms={full_min:.3f} "
            f"selected={selected}/{key_blocks} "
            f"effective_sparsity={1.0 - selected / key_blocks:.3f} "
            f"speedup={dense_median / full_median:.3f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
