#!/usr/bin/env python3
"""Benchmark dense, routing, kernel, and full sparse attention at SP4 tail shape."""

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

from wan_5b.modules.hsa_attention import (
    HSAAttentionConfig,
    build_hsa_block_lut,
    hsa_cag_attention,
)
from wan_5b.modules.hsa_sla_attention import (
    HSASLAAttentionConfig,
    build_hsa_sla_block_lut,
    hsa_sla_cag_attention,
)
from wan_5b.modules.sla_attention import (
    SLAAttentionConfig,
    _run_sparse_backend,
    build_sla_block_lut,
    resolve_chunk_sparsity,
    sla_cag_attention,
)
from wan_5b.modules.sla_attention_ascend import (
    ascend_triton_available,
    ascend_triton_unavailable_reason,
)
from wan_5b.modules.sla_attention_mindiesd import (
    mindiesd_available,
    mindiesd_bsa_available,
    mindiesd_bsa_unavailable_reason,
    mindiesd_unavailable_reason,
)
from wan_5b.modules.sparse_attention import calculate_chunk_sparsities


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


def _config(method: str, backend: str, latent_frames: int):
    common = {
        "enabled": True,
        "method": method,
        "backend": backend,
        "block_q": 128,
        "block_k": 128,
        "num_output_frames": latent_frames,
        "num_frame_per_block": 8,
        "local_attn_size": 32,
        "query_block_batch": 1,
    }
    if method == "hsa_cag":
        return HSAAttentionConfig.from_mapping(
            {
                **common,
                "sparsity": 0.85,
                "sparsity_base": 0.95,
                "keep_frames": 6,
                "keep_sink": 1,
                "keep_near": 2,
                "dense_current": True,
                "min_sparse_history_frames": 2,
            }
        )
    if method == "hsa_sla_cag":
        return HSASLAAttentionConfig.from_mapping(
            {
                **common,
                "sparsity": 0.90,
                "sparsity_base": 0.93,
                "feature_map": "softmax",
                "candidate_frames": 8,
                "keep_sink_frames": 1,
                "keep_recent_frames": 1,
                "dense_current": False,
                "min_sparse_history_frames": 2,
                "linear_cache": True,
            }
        )
    return SLAAttentionConfig.from_mapping(
        {
            **common,
            "sparsity": 0.95,
            "sparsity_base": 0.97,
            "feature_map": "softmax",
            "keep_sink_frames": 1,
            "keep_recent_frames": 1,
            "dense_current": False,
            "min_sparse_history_frames": 1,
            "linear_cache": True,
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
    parser.add_argument(
        "--method", choices=("hsa_cag", "sla_cag", "hsa_sla_cag"), required=True
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--backend", choices=("mindiesd", "mindiesd_bsa", "ascend_triton"),
        default="mindiesd",
    )
    parser.add_argument("--latent-frames", type=int, choices=(32, 192, 384), default=192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--check-linear-backward",
        action="store_true",
        help=(
            "for hsa_sla_cag, verify the local linear projection receives gradients; "
            "this does not validate sparse-kernel Q/K/V backward"
        ),
    )
    parser.add_argument(
        "--check-training-backward",
        action="store_true",
        help=(
            "for hsa_sla_cag with ascend_triton, verify real-shape Q/K/V and "
            "linear-projection backward required by multi-layer training"
        ),
    )
    args = parser.parse_args()
    if args.method in {"hsa_cag", "hsa_sla_cag"} and args.backend == "mindiesd_bsa":
        raise ValueError("HSA-based methods support mindiesd RainFusion or ascend_triton")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    if args.check_linear_backward and args.method != "hsa_sla_cag":
        raise ValueError("--check-linear-backward requires --method hsa_sla_cag")
    if args.check_training_backward and args.method != "hsa_sla_cag":
        raise ValueError("--check-training-backward requires --method hsa_sla_cag")
    if args.check_training_backward and args.backend != "ascend_triton":
        raise ValueError(
            "--check-training-backward requires --backend ascend_triton; "
            "MindIE-SD sparse operators are forward-only"
        )
    if args.backend == "mindiesd" and not mindiesd_available():
        raise RuntimeError("MindIE-SD RainFusion is unavailable: " + mindiesd_unavailable_reason())
    if args.backend == "mindiesd_bsa" and not mindiesd_bsa_available():
        raise RuntimeError("MindIE-SD BSA is unavailable: " + mindiesd_bsa_unavailable_reason())
    if args.backend == "ascend_triton" and not ascend_triton_available():
        raise RuntimeError(
            "Ascend Triton is unavailable: " + ascend_triton_unavailable_reason()
        )

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
    config = _config(args.method, args.backend, args.latent_frames)
    config_map = {**config.__dict__, "method": args.method}
    schedule = calculate_chunk_sparsities(
        args.latent_frames, 8, 32, config_map
    )
    chunk_id = args.latent_frames // 8 - 1
    sparsity = schedule[min(chunk_id, len(schedule) - 1)]
    projection = torch.nn.Linear(128, 128, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        projection.weight.zero_()
        projection.bias.zero_()

        dense_median, dense_min = _measure(
            lambda: F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        route_cache = {}
        if args.method == "hsa_cag":
            route = lambda: build_hsa_block_lut(
                q, k, frame_seq=880, chunk_id=chunk_id, config=config,
                sparsity=sparsity, attention_cache=route_cache,
            )
            full = lambda: hsa_cag_attention(
                q, k, v, frame_seq=880, chunk_id=chunk_id,
                sparse_config=config, attention_cache=route_cache,
            )
        elif args.method == "sla_cag":
            route = lambda: build_sla_block_lut(
                q, k, frame_seq=880, sparsity=sparsity, config=config,
                cache=route_cache, cache_token=chunk_id,
            )
            full_cache = {}
            full = lambda: sla_cag_attention(
                q, k, v, frame_seq=880, chunk_id=chunk_id,
                sparse_config=config, linear_projection=projection,
                attention_cache=full_cache,
            )
        else:
            route = lambda: build_hsa_sla_block_lut(
                q, k, frame_seq=880, sparsity=sparsity, config=config,
                cache=route_cache, cache_token=chunk_id,
            )
            full_cache = {}
            full = lambda: hsa_sla_cag_attention(
                q, k, v, frame_seq=880, chunk_id=chunk_id,
                sparse_config=config, linear_projection=projection,
                attention_cache=full_cache,
            )

        route_median, route_min = _measure(
            route, warmup=args.warmup, iterations=args.iterations
        )
        lut = route()
        kernel_median, kernel_min = _measure(
            lambda: _run_sparse_backend(q, k, v, lut, config),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        full_median, full_min = _measure(
            full, warmup=args.warmup, iterations=args.iterations
        )

    selected = lut.shape[-1]
    total = k.shape[1] // 128
    print(
        f"dense median_ms={dense_median:.3f} min_ms={dense_min:.3f} "
        f"q={shape_q} kv={shape_kv}"
    )
    print(
        f"sparse method={args.method} backend={args.backend} block=128 "
        f"route_ms={route_median:.3f} route_min_ms={route_min:.3f} "
        f"kernel_ms={kernel_median:.3f} kernel_min_ms={kernel_min:.3f} "
        f"full_ms={full_median:.3f} full_min_ms={full_min:.3f} "
        f"selected={selected}/{total} effective_sparsity={1.0-selected/total:.3f} "
        f"speedup={dense_median/full_median:.3f}x"
    )
    if args.check_linear_backward:
        projection.zero_grad(set_to_none=True)
        output = hsa_sla_cag_attention(
            q,
            k,
            v,
            frame_seq=880,
            chunk_id=chunk_id,
            sparse_config=config,
            linear_projection=projection,
            attention_cache=None,
        )
        output.float().square().mean().backward()
        weight_grad = projection.weight.grad
        bias_grad = projection.bias.grad
        if weight_grad is None or bias_grad is None:
            raise RuntimeError("hybrid linear projection did not receive gradients")
        if not torch.isfinite(weight_grad).all().item() or not torch.isfinite(bias_grad).all().item():
            raise RuntimeError("hybrid linear projection gradients are non-finite")
        grad_norm = weight_grad.float().norm().item()
        if grad_norm == 0.0:
            raise RuntimeError("hybrid linear projection weight gradient is zero")
        print(
            f"linear_backward=passed weight_grad_norm={grad_norm:.6f} "
            f"bias_grad_norm={bias_grad.float().norm().item():.6f} "
            f"q_grad={q.grad is not None} k_grad={k.grad is not None} "
            f"v_grad={v.grad is not None}"
        )
    if args.check_training_backward:
        projection.zero_grad(set_to_none=True)
        q_train = q.detach().requires_grad_(True)
        k_train = k.detach().requires_grad_(True)
        v_train = v.detach().requires_grad_(True)
        output = hsa_sla_cag_attention(
            q_train,
            k_train,
            v_train,
            frame_seq=880,
            chunk_id=chunk_id,
            sparse_config=config,
            linear_projection=projection,
            attention_cache=None,
        )
        output.float().square().mean().backward()
        gradients = {
            "q": q_train.grad,
            "k": k_train.grad,
            "v": v_train.grad,
            "weight": projection.weight.grad,
            "bias": projection.bias.grad,
        }
        for name, gradient in gradients.items():
            if gradient is None:
                raise RuntimeError(f"hybrid training backward produced no {name} gradient")
            if not torch.isfinite(gradient).all().item():
                raise RuntimeError(
                    f"hybrid training backward produced non-finite {name} gradient"
                )
        norms = {
            name: gradient.float().norm().item()
            for name, gradient in gradients.items()
        }
        if any(value == 0.0 for value in norms.values()):
            raise RuntimeError(f"hybrid training backward has zero gradients: {norms}")
        print(
            "training_backward=passed "
            + " ".join(f"{name}_grad_norm={value:.6f}" for name, value in norms.items())
        )


if __name__ == "__main__":
    main()
