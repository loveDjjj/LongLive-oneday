#!/usr/bin/env python3
"""Validate MindIE-SD RainFusionAttention against dense NPU attention."""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_5b.modules.sla_attention_mindiesd import (
    mindiesd_available,
    mindiesd_sparse_attention_blhd,
    mindiesd_unavailable_reason,
)
from wan_5b.modules.sla_attention import _portable_sparse_attention


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--q-tokens", type=int, default=1024)
    parser.add_argument("--kv-tokens", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--selected-blocks",
        type=int,
        default=8,
        help="KV blocks retained by the sparse-LUT correctness check",
    )
    args = parser.parse_args()
    if args.q_tokens % 128 or args.kv_tokens % 128:
        raise ValueError("q-tokens and kv-tokens must be divisible by 128")
    if not mindiesd_available():
        raise RuntimeError(mindiesd_unavailable_reason())

    device = torch.device(args.device)
    torch.npu.set_device(device)
    device_name = torch.npu.get_device_properties(device).name
    print(
        f"torch={torch.__version__} torch_npu={_package_version('torch-npu')} "
        f"mindiesd={_package_version('mindiesd')} cann={_cann_version()} "
        f"device={device_name}",
        flush=True,
    )
    generator = torch.Generator(device="cpu").manual_seed(17)
    shape_q = (1, args.q_tokens, args.heads, args.head_dim)
    shape_kv = (1, args.kv_tokens, args.heads, args.head_dim)
    q = torch.randn(shape_q, generator=generator, dtype=_dtype(args.dtype)).to(device)
    k = torch.randn(shape_kv, generator=generator, dtype=_dtype(args.dtype)).to(device)
    v = torch.randn(shape_kv, generator=generator, dtype=_dtype(args.dtype)).to(device)

    q_blocks = args.q_tokens // 128
    kv_blocks = args.kv_tokens // 128
    if not 0 < args.selected_blocks <= kv_blocks:
        raise ValueError("selected-blocks must be in [1, kv-tokens / 128]")
    full_lut = torch.arange(kv_blocks, device=device).view(1, 1, 1, -1)
    full_lut = full_lut.expand(1, args.heads, q_blocks, -1).contiguous()
    sparse_lut = full_lut[..., : args.selected_blocks].contiguous()

    with torch.no_grad():
        sparse = mindiesd_sparse_attention_blhd(q, k, v, full_lut)
        dense = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        sparse_selected = mindiesd_sparse_attention_blhd(q, k, v, sparse_lut)
        sparse_reference = _portable_sparse_attention(
            q,
            k,
            v,
            sparse_lut,
            block_q=128,
            block_k=128,
            query_block_batch=1,
            scale=None,
        )
        torch.npu.synchronize()

    difference = (sparse.float() - dense.float()).abs()
    max_abs = difference.max().item()
    mean_abs = difference.mean().item()
    tolerance = 3e-2 if args.dtype == "bf16" else 1e-2
    print(
        f"dtype={args.dtype} q={shape_q} kv={shape_kv} "
        f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
    )
    if max_abs > tolerance:
        raise AssertionError(
            f"MindIE-SD full-LUT output differs from dense attention: "
            f"{max_abs:.6f} > {tolerance:.6f}"
        )
    sparse_difference = (sparse_selected.float() - sparse_reference.float()).abs()
    sparse_max_abs = sparse_difference.max().item()
    sparse_mean_abs = sparse_difference.mean().item()
    print(
        f"selected_blocks={args.selected_blocks}/{kv_blocks} "
        f"sparse_max_abs={sparse_max_abs:.6f} "
        f"sparse_mean_abs={sparse_mean_abs:.6f}"
    )
    if sparse_max_abs > tolerance:
        raise AssertionError(
            "MindIE-SD sparse-LUT output differs from the portable reference: "
            f"{sparse_max_abs:.6f} > {tolerance:.6f}"
        )
    print("MindIE-SD SLA sparse smoke test passed")


if __name__ == "__main__":
    main()
