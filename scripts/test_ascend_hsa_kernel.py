#!/usr/bin/env python3
"""Compare the Ascend Triton HSA kernel with the portable PyTorch reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_5b.modules.sparse_attention_ascend import (
    ascend_triton_available,
    ascend_triton_sparse_attention,
    ascend_triton_unavailable_reason,
)


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

    if not ascend_triton_available():
        raise RuntimeError(ascend_triton_unavailable_reason())

    torch.manual_seed(7)
    device = torch.device(args.device)
    shape_q = (1, 2, 80, 128)
    shape_kv = (1, 2, 160, 128)
    block_q = block_k = 40
    lut = torch.tensor(
        [[[[0, 2, 3], [1, 2, 3]], [[0, 1, 3], [0, 2, 3]]]],
        device=device,
        dtype=torch.long,
    )

    source_q = torch.randn(shape_q, device=device, dtype=torch.bfloat16)
    source_k = torch.randn(shape_kv, device=device, dtype=torch.bfloat16)
    source_v = torch.randn(shape_kv, device=device, dtype=torch.bfloat16)
    triton_inputs = [x.detach().clone().requires_grad_(True) for x in (source_q, source_k, source_v)]
    reference_inputs = [x.detach().clone().requires_grad_(True) for x in (source_q, source_k, source_v)]

    actual = ascend_triton_sparse_attention(
        *triton_inputs, lut, block_q=block_q, block_k=block_k
    )
    expected = reference_attention(*reference_inputs, lut, block_q, block_k)
    maximum, mean = error_stats(actual, expected)
    print(f"forward max_abs={maximum:.6f} mean_abs={mean:.6f}")
    if maximum > 0.05 or mean > 0.01:
        raise AssertionError("forward error exceeds BF16 tolerance")

    if not args.forward_only:
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
        for name, actual_input, expected_input in zip(
            ("dq", "dk", "dv"), triton_inputs, reference_inputs
        ):
            maximum, mean = error_stats(actual_input.grad, expected_input.grad)
            print(f"{name} max_abs={maximum:.6f} mean_abs={mean:.6f}")
            if maximum > 0.08 or mean > 0.015:
                raise AssertionError(f"{name} error exceeds BF16 tolerance")

    print("Ascend Triton HSA smoke test passed")


if __name__ == "__main__":
    main()
