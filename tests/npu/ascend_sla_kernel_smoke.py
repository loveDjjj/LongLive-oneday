#!/usr/bin/env python3
"""Compare the Ascend Triton SLA sparse kernel with the portable PyTorch reference."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

print("[smoke] importing PyTorch", flush=True)
import torch
import torch.nn.functional as F
print(f"[smoke] PyTorch imported: {torch.__version__}", flush=True)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("[smoke] importing Ascend Triton backend", flush=True)
from wan_5b.modules.sla_attention_ascend import (
    ascend_triton_available,
    ascend_triton_sparse_attention,
    ascend_triton_sparse_attention_blhd,
    ascend_triton_unavailable_reason,
)
print("[smoke] Ascend Triton backend imported", flush=True)


def stage(message):
    print(f"[smoke] {message}", flush=True)


def reference_attention(q, k, v, lut, block_q, block_k):
    b, heads, lq, dim = q.shape
    q_blocks = lq // block_q
    k_blocks = k.shape[2] // block_k
    k_view = k.reshape(b, heads, k_blocks, block_k, dim)
    v_view = v.reshape(b, heads, k_blocks, block_k, dim)
    expanded_k = k_view.unsqueeze(2).expand(-1, -1, q_blocks, -1, -1, -1)
    expanded_v = v_view.unsqueeze(2).expand_as(expanded_k)
    indices = lut.unsqueeze(-1).unsqueeze(-1).expand(
        -1, -1, -1, -1, block_k, dim
    )
    selected_k = torch.gather(expanded_k, 3, indices).flatten(3, 4)
    selected_v = torch.gather(expanded_v, 3, indices).flatten(3, 4)
    q_view = q.reshape(b, heads, q_blocks, block_q, dim)
    merged = b * heads * q_blocks
    output = F.scaled_dot_product_attention(
        q_view.reshape(merged, block_q, dim),
        selected_k.reshape(merged, selected_k.shape[-2], dim),
        selected_v.reshape(merged, selected_v.shape[-2], dim),
    )
    return output.reshape(b, heads, lq, dim)


def error_stats(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return difference.max().item(), difference.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    stage("checking Ascend Triton availability")
    if not ascend_triton_available():
        raise RuntimeError(ascend_triton_unavailable_reason())
    stage(f"backend available; device={args.device}")

    device = torch.device(args.device)
    shape_q = (1, 2, 256, 128)
    shape_kv = (1, 2, 512, 128)
    block_q = block_k = 128
    stage("creating deterministic inputs on CPU")
    cpu_generator = torch.Generator(device="cpu").manual_seed(7)
    lut_cpu = torch.tensor(
        [[[[0, 2, 3], [1, 2, 3]], [[0, 1, 3], [0, 2, 3]]]],
        dtype=torch.long,
    )
    source_q_cpu = torch.randn(
        shape_q, generator=cpu_generator, dtype=torch.bfloat16
    )
    source_k_cpu = torch.randn(
        shape_kv, generator=cpu_generator, dtype=torch.bfloat16
    )
    source_v_cpu = torch.randn(
        shape_kv, generator=cpu_generator, dtype=torch.bfloat16
    )

    stage(f"initializing selected device {device}")
    torch.npu.set_device(device)
    torch.empty(1, device=device)
    torch.npu.synchronize()
    stage("selected NPU initialized")

    stage("copying LUT and BF16 inputs to NPU")
    lut = lut_cpu.to(device)
    source_q = source_q_cpu.to(device)
    source_k = source_k_cpu.to(device)
    source_v = source_v_cpu.to(device)
    triton_inputs = [
        x.detach().clone().requires_grad_(True) for x in (source_q, source_k, source_v)
    ]
    reference_inputs = [
        x.detach().clone().requires_grad_(True) for x in (source_q, source_k, source_v)
    ]
    torch.npu.synchronize()

    stage("launching Triton forward (first run compiles the kernel)")
    started_at = time.perf_counter()
    actual = ascend_triton_sparse_attention(
        *triton_inputs, lut, block_q=block_q, block_k=block_k
    )
    torch.npu.synchronize()
    stage(f"Triton forward finished in {time.perf_counter() - started_at:.2f}s")

    stage("running portable reference forward")
    started_at = time.perf_counter()
    expected = reference_attention(*reference_inputs, lut, block_q, block_k)
    torch.npu.synchronize()
    stage(f"reference forward finished in {time.perf_counter() - started_at:.2f}s")
    maximum, mean = error_stats(actual, expected)
    print(f"forward max_abs={maximum:.6f} mean_abs={mean:.6f}", flush=True)
    if maximum > 0.05 or mean > 0.01:
        raise AssertionError("forward error exceeds BF16 tolerance")

    stage("launching inference-only BLHD Triton forward")
    with torch.no_grad():
        actual_blhd = ascend_triton_sparse_attention_blhd(
            source_q.permute(0, 2, 1, 3).contiguous(),
            source_k.permute(0, 2, 1, 3).contiguous(),
            source_v.permute(0, 2, 1, 3).contiguous(),
            lut,
            block_q=block_q,
            block_k=block_k,
        )
    torch.npu.synchronize()
    maximum, mean = error_stats(
        actual_blhd,
        expected.permute(0, 2, 1, 3).contiguous(),
    )
    print(f"BLHD forward max_abs={maximum:.6f} mean_abs={mean:.6f}", flush=True)
    if maximum > 0.05 or mean > 0.01:
        raise AssertionError("BLHD forward error exceeds BF16 tolerance")

    if not args.forward_only:
        grad = torch.randn(
            actual.shape, generator=cpu_generator, dtype=actual.dtype
        ).to(device)
        stage("launching Triton backward (first run compiles backward kernels)")
        started_at = time.perf_counter()
        actual.backward(grad)
        torch.npu.synchronize()
        stage(f"Triton backward finished in {time.perf_counter() - started_at:.2f}s")

        stage("running portable reference backward")
        started_at = time.perf_counter()
        expected.backward(grad)
        torch.npu.synchronize()
        stage(f"reference backward finished in {time.perf_counter() - started_at:.2f}s")
        for name, actual_input, expected_input in zip(
            ("dq", "dk", "dv"), triton_inputs, reference_inputs
        ):
            maximum, mean = error_stats(actual_input.grad, expected_input.grad)
            print(f"{name} max_abs={maximum:.6f} mean_abs={mean:.6f}", flush=True)
            if maximum > 0.08 or mean > 0.015:
                raise AssertionError(f"{name} error exceeds BF16 tolerance")

    print("Ascend Triton SLA sparse smoke test passed", flush=True)


if __name__ == "__main__":
    main()
