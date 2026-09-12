# Copyright 2026 LongLive sparse attention contributors.
# SPDX-License-Identifier: Apache-2.0
"""CUDA FlexAttention execution of the shared rectangular sparse block LUT.

Uses the public PyTorch 2.9 BlockMask.from_kv_blocks interface. Only block-level
metadata is materialized; no Q-token by KV-token mask or gathered KV copies are
created. FlexAttention generates the Q/K/V backward using the transposed block
indices. CUDA correctness and performance require the hardware smoke tests.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F


def _flex_api():
    if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 9):
        raise RuntimeError("cuda_flex requires PyTorch >= 2.9 with FlexAttention")
    try:
        from torch.nn.attention.flex_attention import BlockMask, flex_attention
    except ImportError as error:
        raise RuntimeError("cuda_flex requires PyTorch >= 2.9 with FlexAttention") from error
    return BlockMask, flex_attention


def cuda_flex_available() -> bool:
    try:
        _flex_api()
        _cuda_flex_kernel_options()
    except RuntimeError:
        return False
    return torch.cuda.is_available()


def build_cuda_block_mask(
    block_lut: torch.Tensor,
    *,
    query_tokens: int,
    key_tokens: int,
    block_q: int = 128,
    block_k: int = 128,
    validate_lut: bool = True,
):
    """Convert [B,H,Q_blocks,K_selected] into a forward/backward BlockMask.

    All maintained routers produce unique whole-block indices. Padding the last
    axis to K_blocks is necessary: from_kv_blocks derives the transpose extent
    from this axis, not from the largest selected index. Metadata is therefore
    O(B*H*Q_blocks*K_blocks), even when the trailing KV blocks are never chosen.
    Content checks synchronize CUDA and can be disabled for trusted router LUTs.
    """
    if block_q != 128 or block_k != 128:
        raise ValueError("cuda_flex requires 128-token query and KV blocks")
    if query_tokens <= 0 or key_tokens <= 0:
        raise ValueError("cuda_flex requires nonempty query and KV sequences")
    if query_tokens % block_q or key_tokens % block_k:
        raise ValueError("cuda_flex requires query/KV lengths divisible by 128")
    if block_lut.ndim != 4 or min(block_lut.shape[:2]) <= 0:
        raise ValueError("block_lut must have shape [B,H,Q_blocks,K_selected]")
    if block_lut.dtype not in (torch.int32, torch.int64):
        raise TypeError("block_lut must contain int32 or int64 indices")
    q_blocks, k_blocks = query_tokens // block_q, key_tokens // block_k
    selected = block_lut.shape[-1]
    if block_lut.shape[-2] != q_blocks or not 0 < selected <= k_blocks:
        raise ValueError("block_lut shape is incompatible with query/KV block counts")
    if validate_lut:
        if torch.any(block_lut < 0).item() or torch.any(block_lut >= k_blocks).item():
            raise ValueError("block_lut contains an out-of-range KV block index")
        sorted_indices = block_lut.sort(dim=-1).values
        if torch.any(sorted_indices[..., 1:] == sorted_indices[..., :-1]).item():
            raise ValueError("block_lut must not contain duplicate KV blocks in a row")

    BlockMask, _ = _flex_api()
    full_indices = F.pad(block_lut.to(torch.int32), (0, k_blocks - selected)).contiguous()
    full_counts = torch.full(
        block_lut.shape[:-1], selected, dtype=torch.int32, device=block_lut.device
    )
    # Every chosen 128x128 block is fully visible. Zero partial-block counts and
    # full-block metadata let the kernel skip per-token mask evaluation.
    return BlockMask.from_kv_blocks(
        kv_num_blocks=torch.zeros_like(full_counts),
        kv_indices=torch.zeros_like(full_indices),
        full_kv_num_blocks=full_counts,
        full_kv_indices=full_indices,
        BLOCK_SIZE=(block_q, block_k),
    )


@lru_cache(maxsize=1)
def _cuda_flex_kernel_options() -> dict:
    from torch.nn.attention.flex_attention import FlexKernelOptions

    options = FlexKernelOptions.__annotations__
    # Newer PyTorch can select a CuTe/FLASH lowering that has had correctness
    # bugs with metadata-only sparse masks (pytorch/pytorch#188322). Explicitly
    # retain Triton on those versions; 2.9 uses the legacy forcing option.
    if "BACKEND" in options:
        return {"BACKEND": "TRITON"}
    if "FORCE_USE_FLEX_ATTENTION" in options:
        return {"FORCE_USE_FLEX_ATTENTION": True}
    raise RuntimeError("cuda_flex requires a PyTorch FlexAttention Triton backend selector")


@lru_cache(maxsize=1)
def _compiled_flex_attention():
    _, flex_attention = _flex_api()
    # Match the model's existing CUDA FlexAttention compilation mode, which
    # avoids CUDA graphs capturing mutable rolling-cache inputs.
    return torch.compile(
        flex_attention, fullgraph=True, dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )


def cuda_flex_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: torch.Tensor,
    *,
    block_q: int = 128,
    block_k: int = 128,
    scale: float | None = None,
    validate_lut: bool = True,
) -> torch.Tensor:
    """Execute BF16/FP16 BLHD attention, with Q/K/V autograd, on CUDA."""
    if q.device.type != "cuda":
        raise ValueError("cuda_flex requires q/k/v on a CUDA device")
    if any(t.device != q.device for t in (k, v, block_lut)):
        raise ValueError("q/k/v and block_lut must be on the same CUDA device")
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError("cuda_flex expects compatible BLHD q/k/v tensors")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("q/k/v batch, head count, and head dimension must match")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("cuda_flex requires matching BF16/FP16 q/k/v tensors")
    if q.shape[-1] not in (64, 128, 256):
        raise ValueError("cuda_flex supports head dimensions 64, 128, or 256")
    if block_lut.ndim != 4 or block_lut.shape[:2] != (q.shape[0], q.shape[2]):
        raise ValueError("block_lut batch/head dimensions must match q/k/v")
    mask = build_cuda_block_mask(
        block_lut, query_tokens=q.shape[1], key_tokens=k.shape[1],
        block_q=block_q, block_k=block_k, validate_lut=validate_lut,
    )
    output = _compiled_flex_attention()(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        block_mask=mask, scale=scale, kernel_options=_cuda_flex_kernel_options(),
    )
    return output.transpose(1, 2).contiguous()


__all__ = ["build_cuda_block_mask", "cuda_flex_available", "cuda_flex_sparse_attention"]
