import torch

from wan_5b.modules.sla_attention import (
    SLAAttentionConfig,
    _portable_sparse_attention,
    build_sla_block_lut,
    calculate_chunk_sparsities,
    sla_cag_attention,
)
from wan_5b.modules.sla_attention_ascend import _requires_autograd
from wan_5b.modules.sla_attention_mindiesd import (
    _prepare_mindiesd_bsa_mask,
    _prepare_mindiesd_lut,
)


def _projection(dim: int, *, identity: bool = False) -> torch.nn.Linear:
    projection = torch.nn.Linear(dim, dim)
    with torch.no_grad():
        if identity:
            projection.weight.copy_(torch.eye(dim))
        else:
            projection.weight.zero_()
        projection.bias.zero_()
    return projection


def _config(**overrides) -> SLAAttentionConfig:
    values = {
        "enabled": True,
        "backend": "portable",
        "sparsity": 0.75,
        "sparsity_base": 0.875,
        "block_q": 2,
        "block_k": 2,
        "feature_map": "softmax",
        "keep_sink_frames": 1,
        "keep_recent_frames": 1,
        "dense_current": False,
        "min_sparse_history_frames": 1,
    }
    values.update(overrides)
    return SLAAttentionConfig(**values)


def test_cag_starts_dense_and_preserves_average_budget():
    config = _config(sparsity=0.75, sparsity_base=0.9)
    schedule = calculate_chunk_sparsities(32, 8, 32, config)

    assert len(schedule) == 4
    assert schedule[0] == 0.0
    assert schedule[1] < schedule[-1]
    kv_lengths = [16, 24, 32]
    kept = sum((1.0 - value) * length for value, length in zip(schedule[1:], kv_lengths))
    target = sum((1.0 - config.sparsity) * length for length in kv_lengths)
    assert abs(kept - target) < 1.0e-6


def test_topk_alias_maps_to_sparsity():
    config = SLAAttentionConfig.from_mapping({"topk": 0.05})
    assert config.sparsity == 0.95


def test_first_chunk_is_exact_dense_attention():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 2, 4)
    output = sla_cag_attention(
        q,
        q,
        q,
        frame_seq=2,
        chunk_id=0,
        sparse_config=_config(),
        linear_projection=_projection(4, identity=True),
    )
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2)
    ).transpose(1, 2)
    torch.testing.assert_close(output, reference)


def test_rectangular_lut_keeps_sink_and_recent_blocks():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    config = _config(sparsity=0.75)
    lut = build_sla_block_lut(
        q,
        k,
        frame_seq=2,
        sparsity=config.sparsity,
        config=config,
    )

    # Global SLA selects 25% of six KV blocks. Sink block 0 and latest block 5
    # are fixed and consume that budget; the current chunk is not all-dense.
    assert lut.shape == (1, 2, 2, 2)
    assert torch.all(lut[..., 0] == 0)
    assert torch.all(lut[..., 1] == 5)


def test_dense_current_is_optional_quality_mode():
    torch.manual_seed(11)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    config = _config(dense_current=True)
    lut = build_sla_block_lut(
        q, k, frame_seq=2, sparsity=config.sparsity, config=config
    )
    assert torch.all(lut[..., -2:] == torch.tensor([4, 5]))


def test_zero_linear_projection_matches_sparse_branch():
    torch.manual_seed(2)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    v = torch.randn_like(k)
    config = _config()
    lut = build_sla_block_lut(
        q, k, frame_seq=2, sparsity=config.sparsity, config=config
    )
    reference = _portable_sparse_attention(
        q,
        k,
        v,
        lut,
        block_q=2,
        block_k=2,
        query_block_batch=1,
        scale=None,
    )
    output = sla_cag_attention(
        q,
        k,
        v,
        frame_seq=2,
        chunk_id=1,
        sparse_config=config,
        linear_projection=_projection(4),
    )
    torch.testing.assert_close(output, reference)


def test_sla_is_differentiable_for_qkv_and_projection():
    torch.manual_seed(3)
    q = torch.randn(1, 4, 2, 4, requires_grad=True)
    k = torch.randn(1, 12, 2, 4, requires_grad=True)
    v = torch.randn(1, 12, 2, 4, requires_grad=True)
    projection = _projection(4)
    output = sla_cag_attention(
        q,
        k,
        v,
        frame_seq=2,
        chunk_id=1,
        sparse_config=_config(),
        linear_projection=projection,
    )
    output.square().mean().backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0


def test_cached_linear_history_preserves_output():
    torch.manual_seed(4)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    v = torch.randn_like(k)
    config = _config(linear_cache=True)
    projection = _projection(4, identity=True)
    cache = {}

    with torch.no_grad():
        first = sla_cag_attention(
            q,
            k,
            v,
            frame_seq=2,
            chunk_id=1,
            sparse_config=config,
            linear_projection=projection,
            attention_cache=cache,
        )
        history_kv = cache["linear"]["kv"]
        second = sla_cag_attention(
            q,
            k,
            v,
            frame_seq=2,
            chunk_id=1,
            sparse_config=config,
            linear_projection=projection,
            attention_cache=cache,
        )
        uncached = sla_cag_attention(
            q,
            k,
            v,
            frame_seq=2,
            chunk_id=1,
            sparse_config=config,
            linear_projection=projection,
        )

    assert cache["linear"]["kv"] is history_kv
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, uncached)


def test_router_cache_invalidates_when_rolling_chunk_changes():
    torch.manual_seed(5)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    config = _config()
    cache = {}

    with torch.no_grad():
        build_sla_block_lut(
            q,
            k,
            frame_seq=2,
            sparsity=config.sparsity,
            config=config,
            cache=cache,
            cache_token=3,
        )
        first = cache["history_block_keys"]
        build_sla_block_lut(
            q,
            k,
            frame_seq=2,
            sparsity=config.sparsity,
            config=config,
            cache=cache,
            cache_token=3,
        )
        assert cache["history_block_keys"] is first

        shifted = k.clone()
        shifted[:, :8].add_(10.0)
        build_sla_block_lut(
            q,
            shifted,
            frame_seq=2,
            sparsity=config.sparsity,
            config=config,
            cache=cache,
            cache_token=4,
        )

    assert cache["history_block_keys"] is not first
    assert not torch.equal(cache["history_block_keys"], first)


def test_backend_aliases_and_mindiesd_block_contract():
    assert SLAAttentionConfig.from_mapping({"backend": "triton"}).backend == "ascend_triton"
    assert SLAAttentionConfig.from_mapping({"backend": "torch"}).backend == "portable"
    assert SLAAttentionConfig.from_mapping({"backend": "rainfusion"}).backend == "mindiesd"
    assert SLAAttentionConfig.from_mapping({"backend": "bsa"}).backend == "mindiesd_bsa"
    try:
        SLAAttentionConfig.from_mapping(
            {"enabled": True, "backend": "mindiesd", "block_q": 64, "block_k": 64}
        )
    except ValueError as error:
        assert "128-token" in str(error)
    else:
        raise AssertionError("expected MindIE-SD block validation to fail")


def test_mindiesd_lut_stays_compact_and_uses_valid_counts():
    compact = torch.tensor([[[[0, 2], [1, 3]], [[1, 2], [0, 3]]]], dtype=torch.int64)
    select_idx, select_num = _prepare_mindiesd_lut(compact, k_blocks=4)
    assert select_idx.shape == (2, 2, 2)
    assert select_num.shape == (2, 2)
    assert torch.all(select_num == 2)


def test_mindiesd_bsa_mask_expands_compact_lut():
    compact = torch.tensor([[[[0, 2], [1, 3]]]], dtype=torch.int64)
    mask = _prepare_mindiesd_bsa_mask(compact, k_blocks=4)
    expected = torch.tensor([[[[1, 0, 1, 0], [0, 1, 0, 1]]]], dtype=torch.int8)
    torch.testing.assert_close(mask, expected)


def test_ascend_autograd_gate_detects_projection_inputs():
    q = torch.randn(1, 2, 4, 8)
    assert not _requires_autograd(q, q, q)
    assert _requires_autograd(q.requires_grad_(), q, q)
