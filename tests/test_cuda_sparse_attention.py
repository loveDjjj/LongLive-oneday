"""CPU coverage of CUDA block metadata plus conditional CUDA numerical checks."""

import pytest
import torch
import torch.nn.functional as F

from wan_5b.modules.sla_attention import _portable_sparse_attention, _run_sparse_backend
from wan_5b.modules.sla_attention_cuda import (
    _cuda_flex_kernel_options,
    build_cuda_block_mask,
    cuda_flex_sparse_attention,
)
from wan_5b.modules.sparse_attention import parse_sparse_config


def _lut():
    # Different rows and heads, with a trailing KV block that is never selected.
    return torch.tensor([[[[0, 2], [1, 3]], [[1, 2], [0, 3]]]], dtype=torch.int64)


def test_rectangular_mask_preserves_head_routes_and_unused_trailing_blocks():
    lut = _lut()
    mask = build_cuda_block_mask(lut, query_tokens=256, key_tokens=640)
    expected = torch.zeros(1, 2, 2, 5, dtype=torch.int32).scatter_(-1, lut, 1)
    assert mask.shape == (1, 2, 256, 640)
    torch.testing.assert_close(mask.to_dense(), expected)
    assert not mask.kv_num_blocks.any()
    assert mask.full_kv_indices.shape == (1, 2, 2, 5)
    assert torch.all(mask.full_kv_num_blocks == 2)
    # The reverse lists must keep all five KV columns, including zero-degree
    # blocks, so dK/dV receive the correct sparse transpose.
    transposed = torch.zeros(1, 2, 5, 2, dtype=torch.int32)
    for b in range(1):
        for h in range(2):
            for k in range(5):
                count = mask.full_q_num_blocks[b, h, k]
                transposed[b, h, k, mask.full_q_indices[b, h, k, :count]] = 1
    torch.testing.assert_close(transposed, expected.transpose(-1, -2))


def test_real_shape_mask_has_only_block_level_storage():
    lut = torch.arange(30).view(1, 1, 1, 30).expand(1, 6, 55, 30)
    mask = build_cuda_block_mask(lut, query_tokens=7040, key_tokens=28160)
    assert mask.full_kv_indices.shape == (1, 6, 55, 220)
    assert mask.full_q_indices.shape == (1, 6, 220, 55)
    assert mask.to_dense().numel() == 6 * 55 * 220


@pytest.mark.parametrize("method", ["hsa_cag", "sla_cag", "hsa_sla_cag"])
def test_cuda_backend_is_explicit_and_requires_128_blocks(method):
    config = parse_sparse_config({"enabled": True, "method": method, "backend": "cuda_flex"})
    assert config.backend == "cuda_flex"
    with pytest.raises(ValueError, match="128-token"):
        parse_sparse_config({
            "enabled": True, "method": method, "backend": "cuda_flex", "block_k": 64,
        })
    q = torch.zeros(1, 256, 2, 128)
    k = torch.zeros(1, 640, 2, 128)
    with pytest.raises(ValueError, match="CUDA device"):
        _run_sparse_backend(q, k, k, _lut(), config)


@pytest.mark.parametrize("kind", ["negative", "overflow", "duplicate", "dtype", "shape", "length"])
def test_invalid_lut_fails_before_kernel(kind):
    lut = _lut()
    key_tokens = 640
    if kind == "negative":
        lut[0, 0, 0, 0] = -1
    elif kind == "overflow":
        lut[0, 0, 0, 0] = 5
    elif kind == "duplicate":
        lut[0, 0, 0] = 2
    elif kind == "dtype":
        lut = lut.float()
    elif kind == "shape":
        lut = lut[:, :, :1]
    else:
        key_tokens = 639
    with pytest.raises((ValueError, TypeError)):
        build_cuda_block_mask(lut, query_tokens=256, key_tokens=key_tokens)


def test_flex_explicitly_selects_triton_for_installed_api():
    options = _cuda_flex_kernel_options()
    assert options in ({"BACKEND": "TRITON"}, {"FORCE_USE_FLEX_ATTENTION": True})


@pytest.mark.parametrize("annotations,expected", [
    ({"BACKEND": str, "FORCE_USE_FLEX_ATTENTION": bool}, {"BACKEND": "TRITON"}),
    ({"FORCE_USE_FLEX_ATTENTION": bool}, {"FORCE_USE_FLEX_ATTENTION": True}),
])
def test_flex_selector_compatibility_never_combines_new_and_legacy_options(monkeypatch, annotations, expected):
    from torch.nn.attention.flex_attention import FlexKernelOptions

    monkeypatch.setattr(FlexKernelOptions, "__annotations__", annotations)
    _cuda_flex_kernel_options.cache_clear()
    try:
        assert _cuda_flex_kernel_options() == expected
    finally:
        _cuda_flex_kernel_options.cache_clear()


def test_unrecognized_flex_selector_fails_instead_of_silently_selecting_auto(monkeypatch):
    from torch.nn.attention.flex_attention import FlexKernelOptions

    monkeypatch.setattr(FlexKernelOptions, "__annotations__", {})
    _cuda_flex_kernel_options.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="backend selector"):
            _cuda_flex_kernel_options()
    finally:
        _cuda_flex_kernel_options.cache_clear()


def test_unsupported_pytorch_version_has_an_explicit_error(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.8.0")
    with pytest.raises(RuntimeError, match="PyTorch >= 2.9"):
        build_cuda_block_mask(_lut(), query_tokens=256, key_tokens=640)


@pytest.mark.parametrize("query_block_batch", [1, 2])
def test_portable_four_dimensional_sdpa_matches_masked_reference_and_gradients(monkeypatch, query_block_batch):
    torch.manual_seed(72)
    q, k, v = (torch.randn(1, length, 2, 4, requires_grad=True) for length in (4, 10, 10))
    lut = _lut()
    original_sdpa = F.scaled_dot_product_attention
    calls = []

    def require_four_dims(*args, **kwargs):
        assert all(tensor.ndim == 4 for tensor in args[:3])
        calls.append(args[0].shape)
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", require_four_dims)
    actual = _portable_sparse_attention(
        q, k, v, lut, block_q=2, block_k=2,
        query_block_batch=query_block_batch, scale=0.3,
    )
    block_mask = torch.zeros(1, 2, 2, 5, dtype=torch.bool).scatter_(-1, lut, True)
    token_mask = block_mask.repeat_interleave(2, -2).repeat_interleave(2, -1)
    expected = original_sdpa(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=token_mask, scale=0.3,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e)
    assert len(calls) == 2 // query_block_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_flex_rectangular_forward_backward_and_unselected_kv(dtype):
    torch.manual_seed(72)
    q, k, v = (
        torch.randn(1, length, 2, 128, device="cuda", dtype=dtype, requires_grad=True)
        for length in (256, 640, 640)
    )
    lut = _lut().cuda()
    actual = cuda_flex_sparse_attention(q, k, v, lut, scale=0.08)
    # FP32 dense masked attention is used only for this small numerical oracle.
    block_mask = torch.zeros(1, 2, 2, 5, dtype=torch.bool, device="cuda").scatter_(-1, lut, True)
    token_mask = block_mask.repeat_interleave(128, -2).repeat_interleave(128, -1)
    q_ref, k_ref, v_ref = [t.detach().float().requires_grad_() for t in (q, k, v)]
    expected = F.scaled_dot_product_attention(
        q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2),
        attn_mask=token_mask, scale=0.08,
    ).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.03)
    gradient = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (q, k, v), gradient)
    expected_grads = torch.autograd.grad(expected, (q_ref, k_ref, v_ref), gradient.float())
    for a, e in zip(grads, expected_grads):
        assert torch.isfinite(a).all() and a.abs().sum() > 0
        torch.testing.assert_close(a.float(), e, atol=0.025, rtol=0.05)
    assert torch.count_nonzero(grads[1][:, -128:]) == 0
    assert torch.count_nonzero(grads[2][:, -128:]) == 0
    with torch.no_grad():
        baseline = cuda_flex_sparse_attention(q, k, v, lut, scale=0.08)
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, -128:] += 5
        changed_v[:, -128:] += 10
        perturbed = cuda_flex_sparse_attention(q, changed_k, changed_v, lut, scale=0.08)
        torch.testing.assert_close(perturbed, baseline, atol=0, rtol=0)
        dense = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        dense_changed = F.scaled_dot_product_attention(
            q.transpose(1, 2), changed_k.transpose(1, 2), changed_v.transpose(1, 2),
        )
        assert (dense - dense_changed).abs().max() > 0.01
