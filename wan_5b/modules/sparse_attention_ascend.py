# Copyright 2026 LongLive HSA contributors.
# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton backend for cached rectangular block-sparse attention.

The sparse softmax kernels are adapted from the Apache-2.0 MindSpeed-MM-SLA
implementation. Unlike that implementation, this backend accepts different
query and KV lengths and consumes HSA's precomputed block LUT directly.
"""

from __future__ import annotations

import math

import torch


_TRITON_IMPORT_ERROR: Exception | None = None
try:
    import torch_npu  # noqa: F401
    import triton
    import triton.language as tl
    import triton.backends.ascend  # noqa: F401
except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - host dependent
    _TRITON_IMPORT_ERROR = error
    triton = None
    tl = None


def ascend_triton_available() -> bool:
    """Return whether the current process can launch Ascend Triton kernels."""
    if _TRITON_IMPORT_ERROR is not None or triton is None:
        return False
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and npu.is_available())


def ascend_triton_unavailable_reason() -> str:
    if _TRITON_IMPORT_ERROR is not None:
        return f"Ascend Triton import failed: {_TRITON_IMPORT_ERROR}"
    if getattr(torch, "npu", None) is None:
        return "torch.npu is unavailable"
    if not torch.npu.is_available():
        return "no available Ascend NPU was detected"
    return "Ascend Triton is available"


if triton is not None:

    @triton.jit
    def _hsa_sparse_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        lut_ptr,
        out_ptr,
        lse_ptr,
        scale,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_l: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_l: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_h: tl.constexpr,
        v_stride_l: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_l: tl.constexpr,
        LQ: tl.constexpr,
        LKV: tl.constexpr,
        D: tl.constexpr,
        HEADS: tl.constexpr,
        Q_BLOCKS: tl.constexpr,
        SELECTED_BLOCKS: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_Q_PAD: tl.constexpr,
        BLOCK_K_PAD: tl.constexpr,
        STORE_LSE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bh = pid // Q_BLOCKS
        query_block = pid - bh * Q_BLOCKS
        batch_index = bh // HEADS
        head_index = bh - batch_index * HEADS

        q_offsets = tl.arange(0, BLOCK_Q_PAD)
        k_offsets = tl.arange(0, BLOCK_K_PAD)
        d_offsets = tl.arange(0, D)
        q_indices = query_block * BLOCK_Q + q_offsets
        q_mask = (q_offsets < BLOCK_Q) & (q_indices < LQ)

        q_base = batch_index * q_stride_b + head_index * q_stride_h
        k_base = batch_index * k_stride_b + head_index * k_stride_h
        v_base = batch_index * v_stride_b + head_index * v_stride_h
        q = tl.load(
            q_ptr + q_base + q_indices[:, None] * q_stride_l + d_offsets[None, :],
            mask=q_mask[:, None],
            other=0.0,
        )

        row_max = tl.full([BLOCK_Q_PAD], -float("inf"), tl.float32)
        row_sum = tl.zeros([BLOCK_Q_PAD], tl.float32)
        accumulator = tl.zeros([BLOCK_Q_PAD, D], tl.float32)
        lut_base = pid * SELECTED_BLOCKS

        for selected_index in tl.range(0, SELECTED_BLOCKS, num_stages=2):
            key_block = tl.load(lut_ptr + lut_base + selected_index)
            k_indices = key_block * BLOCK_K + k_offsets
            k_mask = (k_offsets < BLOCK_K) & (k_indices < LKV)
            k = tl.load(
                k_ptr + k_base + k_indices[:, None] * k_stride_l + d_offsets[None, :],
                mask=k_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * (scale * 1.4426950408889634)
            scores = tl.where(q_mask[:, None] & k_mask[None, :], scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, block_max)
            probabilities = tl.math.exp2(scores - new_max[:, None])
            alpha = tl.math.exp2(row_max - new_max)
            block_sum = tl.sum(probabilities, axis=1)

            v = tl.load(
                v_ptr + v_base + k_indices[:, None] * v_stride_l + d_offsets[None, :],
                mask=k_mask[:, None],
                other=0.0,
            )
            accumulator = accumulator * alpha[:, None]
            accumulator += tl.dot(probabilities.to(v.dtype), v)
            row_sum = row_sum * alpha + block_sum
            row_max = new_max

        output = accumulator / row_sum[:, None]
        out_base = batch_index * out_stride_b + head_index * out_stride_h
        out_offsets = out_base + q_indices[:, None] * out_stride_l + d_offsets[None, :]
        tl.store(out_ptr + out_offsets, output, mask=q_mask[:, None])
        if STORE_LSE:
            tl.store(
                lse_ptr + bh * LQ + q_indices,
                row_max + tl.math.log2(row_sum),
                mask=q_mask,
            )


    @triton.jit
    def _hsa_sparse_delta(
        out_ptr,
        grad_out_ptr,
        delta_ptr,
        ROWS: tl.constexpr,
        D: tl.constexpr,
    ):
        row = tl.program_id(0)
        d_offsets = tl.arange(0, D)
        mask = row < ROWS
        out = tl.load(out_ptr + row * D + d_offsets, mask=mask, other=0.0)
        grad_out = tl.load(grad_out_ptr + row * D + d_offsets, mask=mask, other=0.0)
        delta = tl.sum(out.to(tl.float32) * grad_out.to(tl.float32), axis=0)
        tl.store(delta_ptr + row, delta, mask=mask)


    @triton.jit
    def _hsa_sparse_bwd_dq(
        q_ptr,
        k_ptr,
        v_ptr,
        lut_ptr,
        lse_ptr,
        delta_ptr,
        grad_out_ptr,
        grad_q_ptr,
        scale,
        LQ: tl.constexpr,
        LKV: tl.constexpr,
        D: tl.constexpr,
        Q_BLOCKS: tl.constexpr,
        SELECTED_BLOCKS: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_Q_PAD: tl.constexpr,
        BLOCK_K_PAD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bh = pid // Q_BLOCKS
        query_block = pid - bh * Q_BLOCKS

        q_offsets = tl.arange(0, BLOCK_Q_PAD)
        k_offsets = tl.arange(0, BLOCK_K_PAD)
        d_offsets = tl.arange(0, D)
        q_indices = query_block * BLOCK_Q + q_offsets
        q_mask = (q_offsets < BLOCK_Q) & (q_indices < LQ)
        q_base = bh * LQ * D
        kv_base = bh * LKV * D

        q_linear_offsets = q_base + q_indices[:, None] * D + d_offsets[None, :]
        q_ptrs = q_ptr + q_linear_offsets
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        grad_out = tl.load(
            grad_out_ptr + q_base + q_indices[:, None] * D + d_offsets[None, :],
            mask=q_mask[:, None],
            other=0.0,
        )
        lse = tl.load(lse_ptr + bh * LQ + q_indices, mask=q_mask, other=float("inf"))
        delta = tl.load(delta_ptr + bh * LQ + q_indices, mask=q_mask, other=0.0)
        grad_q = tl.zeros([BLOCK_Q_PAD, D], tl.float32)
        lut_base = pid * SELECTED_BLOCKS

        for selected_index in tl.range(0, SELECTED_BLOCKS, num_stages=2):
            key_block = tl.load(lut_ptr + lut_base + selected_index)
            k_indices = key_block * BLOCK_K + k_offsets
            k_mask = (k_offsets < BLOCK_K) & (k_indices < LKV)
            k_ptrs = k_ptr + kv_base + k_indices[:, None] * D + d_offsets[None, :]
            v_ptrs = v_ptr + kv_base + k_indices[:, None] * D + d_offsets[None, :]
            k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

            scores = tl.dot(q, tl.trans(k)) * (scale * 1.4426950408889634)
            probabilities = tl.math.exp2(scores - lse[:, None])
            probabilities = tl.where(q_mask[:, None] & k_mask[None, :], probabilities, 0.0)
            grad_probabilities = tl.dot(grad_out, tl.trans(v)).to(tl.float32)
            grad_scores = probabilities * (grad_probabilities - delta[:, None])
            grad_q += tl.dot(grad_scores.to(k.dtype), k)

        tl.store(grad_q_ptr + q_linear_offsets, grad_q * scale, mask=q_mask[:, None])


    @triton.jit
    def _hsa_sparse_bwd_dk(
        q_ptr,
        k_ptr,
        v_ptr,
        sparse_map_ptr,
        lse_ptr,
        delta_ptr,
        grad_out_ptr,
        grad_k_ptr,
        scale,
        LQ: tl.constexpr,
        LKV: tl.constexpr,
        D: tl.constexpr,
        Q_BLOCKS: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_Q_PAD: tl.constexpr,
        BLOCK_K_PAD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bh = pid // K_BLOCKS
        key_block = pid - bh * K_BLOCKS

        q_offsets = tl.arange(0, BLOCK_Q_PAD)
        k_offsets = tl.arange(0, BLOCK_K_PAD)
        d_offsets = tl.arange(0, D)
        k_indices = key_block * BLOCK_K + k_offsets
        k_mask = (k_offsets < BLOCK_K) & (k_indices < LKV)
        q_base = bh * LQ * D
        kv_base = bh * LKV * D

        kv_linear_offsets = kv_base + k_indices[:, None] * D + d_offsets[None, :]
        k_ptrs = k_ptr + kv_linear_offsets
        v_ptrs = v_ptr + kv_linear_offsets
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        grad_k = tl.zeros([BLOCK_K_PAD, D], tl.float32)

        for query_block in tl.range(0, Q_BLOCKS, num_stages=1):
            selected = tl.load(
                sparse_map_ptr + (bh * Q_BLOCKS + query_block) * K_BLOCKS + key_block
            )
            q_indices = query_block * BLOCK_Q + q_offsets
            q_mask = (q_offsets < BLOCK_Q) & (q_indices < LQ)
            q_ptrs = q_ptr + q_base + q_indices[:, None] * D + d_offsets[None, :]
            grad_out_ptrs = (
                grad_out_ptr + q_base + q_indices[:, None] * D + d_offsets[None, :]
            )
            q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
            grad_out = tl.load(grad_out_ptrs, mask=q_mask[:, None], other=0.0)
            lse = tl.load(
                lse_ptr + bh * LQ + q_indices,
                mask=q_mask,
                other=float("inf"),
            )
            delta = tl.load(
                delta_ptr + bh * LQ + q_indices,
                mask=q_mask,
                other=0.0,
            )

            scores_t = tl.dot(k, tl.trans(q)) * (scale * 1.4426950408889634)
            probabilities_t = tl.math.exp2(scores_t - lse[None, :])
            probabilities_t = tl.where(
                (selected != 0) & k_mask[:, None] & q_mask[None, :],
                probabilities_t,
                0.0,
            )
            grad_probabilities_t = tl.dot(v, tl.trans(grad_out)).to(tl.float32)
            grad_scores_t = probabilities_t * (
                grad_probabilities_t - delta[None, :]
            )
            grad_k += tl.dot(grad_scores_t.to(q.dtype), q)

        tl.store(grad_k_ptr + kv_linear_offsets, grad_k * scale, mask=k_mask[:, None])


    @triton.jit
    def _hsa_sparse_bwd_dv(
        q_ptr,
        k_ptr,
        sparse_map_ptr,
        lse_ptr,
        grad_out_ptr,
        grad_v_ptr,
        scale,
        LQ: tl.constexpr,
        LKV: tl.constexpr,
        D: tl.constexpr,
        Q_BLOCKS: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_Q_PAD: tl.constexpr,
        BLOCK_K_PAD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bh = pid // K_BLOCKS
        key_block = pid - bh * K_BLOCKS

        q_offsets = tl.arange(0, BLOCK_Q_PAD)
        k_offsets = tl.arange(0, BLOCK_K_PAD)
        d_offsets = tl.arange(0, D)
        k_indices = key_block * BLOCK_K + k_offsets
        k_mask = (k_offsets < BLOCK_K) & (k_indices < LKV)
        q_base = bh * LQ * D
        kv_base = bh * LKV * D

        kv_linear_offsets = kv_base + k_indices[:, None] * D + d_offsets[None, :]
        k = tl.load(
            k_ptr + kv_linear_offsets,
            mask=k_mask[:, None],
            other=0.0,
        )
        grad_v = tl.zeros([BLOCK_K_PAD, D], tl.float32)

        for query_block in tl.range(0, Q_BLOCKS, num_stages=1):
            selected = tl.load(
                sparse_map_ptr + (bh * Q_BLOCKS + query_block) * K_BLOCKS + key_block
            )
            q_indices = query_block * BLOCK_Q + q_offsets
            q_mask = (q_offsets < BLOCK_Q) & (q_indices < LQ)
            q = tl.load(
                q_ptr + q_base + q_indices[:, None] * D + d_offsets[None, :],
                mask=q_mask[:, None],
                other=0.0,
            )
            grad_out = tl.load(
                grad_out_ptr + q_base + q_indices[:, None] * D + d_offsets[None, :],
                mask=q_mask[:, None],
                other=0.0,
            )
            lse = tl.load(
                lse_ptr + bh * LQ + q_indices,
                mask=q_mask,
                other=float("inf"),
            )

            scores_t = tl.dot(k, tl.trans(q)) * (scale * 1.4426950408889634)
            probabilities_t = tl.math.exp2(scores_t - lse[None, :])
            probabilities_t = tl.where(
                (selected != 0) & k_mask[:, None] & q_mask[None, :],
                probabilities_t,
                0.0,
            )
            grad_v += tl.dot(probabilities_t.to(grad_out.dtype), grad_out)

        tl.store(grad_v_ptr + kv_linear_offsets, grad_v, mask=k_mask[:, None])


class _AscendSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_lut, block_q, block_k, scale):
        if not ascend_triton_available():
            raise RuntimeError(ascend_triton_unavailable_reason())
        if q.device.type != "npu" or k.device.type != "npu" or v.device.type != "npu":
            raise ValueError("Ascend Triton HSA requires q/k/v on an NPU device.")
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(f"Ascend Triton HSA requires BF16/FP16, got {q.dtype}.")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise TypeError("q/k/v must use the same dtype.")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        lut_indices = block_lut.contiguous().to(torch.long)
        block_lut = lut_indices.to(torch.int32)

        b, heads, lq, dim = q.shape
        lkv = k.shape[2]
        q_blocks = math.ceil(lq / block_q)
        k_blocks = math.ceil(lkv / block_k)
        selected_blocks = block_lut.shape[-1]
        block_q_pad = triton.next_power_of_2(block_q)
        block_k_pad = triton.next_power_of_2(block_k)
        if dim not in (64, 128, 256):
            raise ValueError(f"unsupported HSA head dimension: {dim}")
        if block_q_pad > 128 or block_k_pad > 128:
            raise ValueError("Ascend Triton HSA supports block sizes up to 128.")

        output = torch.empty_like(q)
        lse = torch.empty((b, heads, lq), dtype=torch.float32, device=q.device)
        sparse_map = torch.zeros(
            (b, heads, q_blocks, k_blocks), dtype=torch.int8, device=q.device
        )
        sparse_map.scatter_(-1, lut_indices, 1)
        sparse_map = sparse_map.contiguous()

        grid = (b * heads * q_blocks,)
        _hsa_sparse_fwd[grid](
            q,
            k,
            v,
            block_lut,
            output,
            lse,
            scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            LQ=lq,
            LKV=lkv,
            D=dim,
            HEADS=heads,
            Q_BLOCKS=q_blocks,
            SELECTED_BLOCKS=selected_blocks,
            BLOCK_Q=block_q,
            BLOCK_K=block_k,
            BLOCK_Q_PAD=block_q_pad,
            BLOCK_K_PAD=block_k_pad,
            STORE_LSE=True,
        )

        ctx.save_for_backward(q, k, v, block_lut, sparse_map, lse, output)
        ctx.block_q = block_q
        ctx.block_k = block_k
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, block_lut, sparse_map, lse, output = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        b, heads, lq, dim = q.shape
        lkv = k.shape[2]
        block_q = ctx.block_q
        block_k = ctx.block_k
        q_blocks = math.ceil(lq / block_q)
        k_blocks = math.ceil(lkv / block_k)
        block_q_pad = triton.next_power_of_2(block_q)
        block_k_pad = triton.next_power_of_2(block_k)

        delta = torch.empty_like(lse)
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        _hsa_sparse_delta[(b * heads * lq,)](
            output,
            grad_output,
            delta,
            ROWS=b * heads * lq,
            D=dim,
        )
        _hsa_sparse_bwd_dq[(b * heads * q_blocks,)](
            q,
            k,
            v,
            block_lut,
            lse,
            delta,
            grad_output,
            grad_q,
            ctx.scale,
            LQ=lq,
            LKV=lkv,
            D=dim,
            Q_BLOCKS=q_blocks,
            SELECTED_BLOCKS=block_lut.shape[-1],
            BLOCK_Q=block_q,
            BLOCK_K=block_k,
            BLOCK_Q_PAD=block_q_pad,
            BLOCK_K_PAD=block_k_pad,
        )
        kv_grid = (b * heads * k_blocks,)
        _hsa_sparse_bwd_dk[kv_grid](
            q,
            k,
            v,
            sparse_map,
            lse,
            delta,
            grad_output,
            grad_k,
            ctx.scale,
            LQ=lq,
            LKV=lkv,
            D=dim,
            Q_BLOCKS=q_blocks,
            K_BLOCKS=k_blocks,
            BLOCK_Q=block_q,
            BLOCK_K=block_k,
            BLOCK_Q_PAD=block_q_pad,
            BLOCK_K_PAD=block_k_pad,
        )
        _hsa_sparse_bwd_dv[kv_grid](
            q,
            k,
            sparse_map,
            lse,
            grad_output,
            grad_v,
            ctx.scale,
            LQ=lq,
            LKV=lkv,
            D=dim,
            Q_BLOCKS=q_blocks,
            K_BLOCKS=k_blocks,
            BLOCK_Q=block_q,
            BLOCK_K=block_k,
            BLOCK_Q_PAD=block_q_pad,
            BLOCK_K_PAD=block_k_pad,
        )
        return grad_q, grad_k, grad_v, None, None, None, None


def _requires_autograd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v))


def _ascend_triton_sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: torch.Tensor,
    block_q: int,
    block_k: int,
    scale: float,
) -> torch.Tensor:
    """Launch the inference-only kernel without allocating backward state."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_lut = block_lut.contiguous().to(torch.int32)

    b, heads, lq, dim = q.shape
    lkv = k.shape[2]
    q_blocks = math.ceil(lq / block_q)
    block_q_pad = triton.next_power_of_2(block_q)
    block_k_pad = triton.next_power_of_2(block_k)
    output = torch.empty_like(q)

    _hsa_sparse_fwd[(b * heads * q_blocks,)](
        q,
        k,
        v,
        block_lut,
        output,
        output,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        LQ=lq,
        LKV=lkv,
        D=dim,
        HEADS=heads,
        Q_BLOCKS=q_blocks,
        SELECTED_BLOCKS=block_lut.shape[-1],
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        BLOCK_Q_PAD=block_q_pad,
        BLOCK_K_PAD=block_k_pad,
        STORE_LSE=False,
    )
    return output


def ascend_triton_sparse_attention_blhd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: torch.Tensor,
    *,
    block_q: int,
    block_k: int,
    scale: float | None = None,
    validate_lut: bool = True,
) -> torch.Tensor:
    """Run the inference-only HSA kernel directly on contiguous BLHD tensors."""
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v)):
        raise ValueError("BLHD Ascend HSA is inference-only; disable gradients")
    if not ascend_triton_available():
        raise RuntimeError(ascend_triton_unavailable_reason())
    if q.device.type != "npu" or k.device.type != "npu" or v.device.type != "npu":
        raise ValueError("Ascend Triton HSA requires q/k/v on an NPU device.")
    if q.ndim != 4 or k.shape != v.shape:
        raise ValueError("q/k/v must be BLHD tensors and k/v shapes must match.")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("q and k/v batch, head, and head dimensions must match.")
    if q.dtype not in (torch.bfloat16, torch.float16) or q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("Ascend Triton HSA requires matching BF16/FP16 q/k/v tensors.")

    b, lq, heads, dim = q.shape
    lkv = k.shape[1]
    q_blocks = math.ceil(lq / block_q)
    expected = (b, heads, q_blocks)
    if block_lut.ndim != 4 or block_lut.shape[:3] != expected:
        raise ValueError(f"block_lut must start with {expected}, got {tuple(block_lut.shape)}.")
    if block_lut.shape[-1] == 0:
        raise ValueError("block_lut must select at least one key block.")
    key_blocks = math.ceil(lkv / block_k)
    if validate_lut and (
        torch.any(block_lut < 0) or torch.any(block_lut >= key_blocks)
    ):
        raise ValueError("block_lut contains an out-of-range key block index.")

    block_lut = block_lut.contiguous().to(torch.int32)
    block_q_pad = triton.next_power_of_2(block_q)
    block_k_pad = triton.next_power_of_2(block_k)
    if dim not in (64, 128, 256):
        raise ValueError(f"unsupported HSA head dimension: {dim}")
    if block_q_pad > 128 or block_k_pad > 128:
        raise ValueError("Ascend Triton HSA supports block sizes up to 128.")

    output = torch.empty_like(q)
    attention_scale = float(scale) if scale is not None else dim ** -0.5
    _hsa_sparse_fwd[(b * heads * q_blocks,)](
        q,
        k,
        v,
        block_lut,
        output,
        output,
        attention_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        output.stride(0),
        output.stride(2),
        output.stride(1),
        LQ=lq,
        LKV=lkv,
        D=dim,
        HEADS=heads,
        Q_BLOCKS=q_blocks,
        SELECTED_BLOCKS=block_lut.shape[-1],
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        BLOCK_Q_PAD=block_q_pad,
        BLOCK_K_PAD=block_k_pad,
        STORE_LSE=False,
    )
    return output


def ascend_triton_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: torch.Tensor,
    *,
    block_q: int,
    block_k: int,
    scale: float | None = None,
    validate_lut: bool = True,
) -> torch.Tensor:
    """Run HSA sparse attention on BHLD tensors using an Ascend Triton LUT."""
    if not ascend_triton_available():
        raise RuntimeError(ascend_triton_unavailable_reason())
    if q.device.type != "npu" or k.device.type != "npu" or v.device.type != "npu":
        raise ValueError("Ascend Triton HSA requires q/k/v on an NPU device.")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Ascend Triton HSA requires BF16/FP16, got {q.dtype}.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q/k/v must use the same dtype.")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("q/k/v must be BHLD tensors and k/v shapes must match.")
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k/v batch, head, and head dimensions must match.")
    expected_q_blocks = math.ceil(q.shape[2] / block_q)
    expected = (q.shape[0], q.shape[1], expected_q_blocks)
    if block_lut.shape[:3] != expected or block_lut.ndim != 4:
        raise ValueError(
            f"block_lut must start with {expected}, got {tuple(block_lut.shape)}."
        )
    if block_lut.shape[-1] == 0:
        raise ValueError("block_lut must select at least one key block.")
    key_blocks = math.ceil(k.shape[2] / block_k)
    if validate_lut and (
        torch.any(block_lut < 0) or torch.any(block_lut >= key_blocks)
    ):
        raise ValueError("block_lut contains an out-of-range key block index.")
    attention_scale = float(scale) if scale is not None else q.shape[-1] ** -0.5
    if not _requires_autograd(q, k, v):
        return _ascend_triton_sparse_attention_forward(
            q,
            k,
            v,
            block_lut,
            int(block_q),
            int(block_k),
            attention_scale,
        )
    return _AscendSparseAttention.apply(
        q, k, v, block_lut, int(block_q), int(block_k), attention_scale
    )


__all__ = [
    "ascend_triton_available",
    "ascend_triton_sparse_attention",
    "ascend_triton_sparse_attention_blhd",
    "ascend_triton_unavailable_reason",
]
