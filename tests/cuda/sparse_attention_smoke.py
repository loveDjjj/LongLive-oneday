#!/usr/bin/env python3
"""CUDA 稀疏正确性与反向烟测；小形状 portable 也可在 CPU 运行。

--real-shape 使用 SP4 的 Q=7040、KV=28160、H=6、D=128。
本脚本不发布性能数据；CUDA/FlexAttention 首次编译可能耗时数分钟。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wan_5b.modules.sla_attention import _portable_sparse_attention
from wan_5b.modules.sla_attention_cuda import (
    _cuda_flex_kernel_options,
    cuda_flex_sparse_attention,
)
from wan_5b.modules.sparse_attention import sparse_attention, with_cag_schedule


def _portable(q, k, v, lut):
    return _portable_sparse_attention(
        q, k, v, lut, block_q=128, block_k=128,
        query_block_batch=1, scale=None,
    )


def _check_gradients(gradients, names):
    norms = {}
    for name, gradient in zip(names, gradients):
        if gradient is None or not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"{name} gradient is absent or non-finite")
        norm = gradient.float().norm().item()
        if norm == 0:
            raise RuntimeError(f"{name} gradient is zero")
        norms[name] = norm
    return norms


def _check_close(actual, expected, label):
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.025, rtol=0.05)
    print(f"{label}_max_abs={(actual.float() - expected.float()).abs().max().item():.6g}", flush=True)


def _check_softmax(q, k, v, backend, real_shape):
    b, query_tokens, heads, _ = q.shape
    q_blocks, k_blocks = query_tokens // 128, k.shape[1] // 128
    selected = 30 if real_shape else 2
    # Deliberately leave the last KV block unused by every row and head.
    scores = torch.rand(b, heads, q_blocks, k_blocks - 1, device=q.device)
    lut = scores.topk(selected, dim=-1).indices.sort(dim=-1).values
    run = _portable if backend == "portable" else cuda_flex_sparse_attention
    actual = run(q, k, v, lut)
    gradient = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), gradient)
    norms = _check_gradients(actual_grads, ("q", "k", "v"))
    if torch.count_nonzero(actual_grads[1][:, -128:]).item() or torch.count_nonzero(actual_grads[2][:, -128:]).item():
        raise RuntimeError("unselected KV block received sparse-softmax gradients")

    if real_shape:
        # A selected-KV reference avoids constructing a full token-level mask at
        # the 7040 x 28160 production shape. The small case below is independent.
        reference_inputs = [tensor.detach().requires_grad_() for tensor in (q, k, v)]
        reference = _portable(*reference_inputs, lut)
    else:
        reference_inputs = [tensor.detach().float().requires_grad_() for tensor in (q, k, v)]
        allowed = torch.zeros(b, heads, q_blocks, k_blocks, dtype=torch.bool, device=q.device)
        allowed.scatter_(-1, lut, True)
        token_mask = allowed.repeat_interleave(128, -2).repeat_interleave(128, -1)
        reference = F.scaled_dot_product_attention(
            *[tensor.transpose(1, 2) for tensor in reference_inputs], attn_mask=token_mask,
        ).transpose(1, 2)
    reference_grads = torch.autograd.grad(reference, reference_inputs, gradient.to(reference.dtype))
    _check_close(actual, reference, "softmax_output")
    for name, measured, expected in zip(("q", "k", "v"), actual_grads, reference_grads):
        _check_close(measured, expected, f"softmax_{name}_gradient")

    with torch.no_grad():
        baseline = run(q, k, v, lut)
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, -128:] += 5
        changed_v[:, -128:] += 10
        perturbed = run(q, changed_k, changed_v, lut)
        torch.testing.assert_close(perturbed, baseline, rtol=0, atol=0)
        # The dense positive control only needs one query block, including for
        # the real-shape test; it still sees the complete resident KV sequence.
        query = q[:, :128].transpose(1, 2)
        dense = F.scaled_dot_product_attention(query, k.transpose(1, 2), v.transpose(1, 2))
        changed_dense = F.scaled_dot_product_attention(
            query, changed_k.transpose(1, 2), changed_v.transpose(1, 2),
        )
        if (dense - changed_dense).abs().max().item() <= 0.01:
            raise RuntimeError("dense positive control did not respond to perturbed KV")
    print(f"softmax_backward=passed norms={norms} unselected_kv_audit=passed", flush=True)


def _check_method(q, k, v, method, backend, frame_seq):
    projection = torch.nn.Linear(q.shape[-1], q.shape[-1], device=q.device, dtype=q.dtype)
    config = with_cag_schedule(
        {"enabled": True, "method": method, "backend": backend},
        num_output_frames=32, num_frame_per_block=8, local_attn_size=32,
    )
    output = sparse_attention(
        q, k, v, frame_seq=frame_seq, chunk_id=3, sparse_config=config,
        linear_projection=projection,
    )
    inputs = (q, k, v)
    names = ("q", "k", "v")
    if method != "hsa_cag":
        inputs += (projection.weight, projection.bias)
        names += ("linear_weight", "linear_bias")
    gradients = torch.autograd.grad(output, inputs, torch.randn_like(output))
    norms = _check_gradients(gradients, names)
    if not torch.isfinite(output).all().item():
        raise RuntimeError(f"{method} produced a non-finite output")
    print(f"method={method} training_backward=passed norms={norms}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("portable", "cuda_flex"), default="portable")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--real-shape", action="store_true")
    parser.add_argument("--method", choices=("all", "hsa_cag", "sla_cag", "hsa_sla_cag"), default="all")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.backend == "cuda_flex" and device.type != "cuda":
        parser.error("cuda_flex requires --device cuda:N")
    if args.real_shape and device.type != "cuda":
        parser.error("--real-shape requires a CUDA device; use the small CPU portable smoke instead")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(device)}", flush=True)
    else:
        print(f"torch={torch.__version__} device={device}", flush=True)
    if args.backend == "cuda_flex":
        print(f"flex_kernel_options={_cuda_flex_kernel_options()}", flush=True)
    torch.manual_seed(17)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    query_tokens, key_tokens, heads, frame_seq = (7040, 28160, 6, 880) if args.real_shape else (256, 640, 2, 128)
    q, k, v = [
        (torch.randn(1, length, heads, 128, device=device, dtype=dtype) * 0.5).requires_grad_()
        for length in (query_tokens, key_tokens, key_tokens)
    ]
    print(f"backend={args.backend} dtype={dtype} q_shape={tuple(q.shape)} kv_shape={tuple(k.shape)}", flush=True)
    _check_softmax(q, k, v, args.backend, args.real_shape)
    methods = ("hsa_cag", "sla_cag", "hsa_sla_cag") if args.method == "all" else (args.method,)
    for method in methods:
        _check_method(q, k, v, method, args.backend, frame_seq)
    print("CUDA sparse attention smoke passed" if device.type == "cuda" else "CPU portable sparse attention smoke passed", flush=True)


if __name__ == "__main__":
    main()
