import torch

from wan_5b.modules.sla_attention import (
    SLAAttentionConfig,
    _linear_attention,
    _portable_sparse_attention,
    _projected_linear_attention,
    build_sla_block_lut,
    calculate_chunk_sparsities,
    sla_cag_attention,
)
from wan_5b.modules.sla_attention_ascend import _kernel_tiles, _requires_autograd
from wan_5b.modules.sla_attention_mindiesd import (
    _prepare_mindiesd_bsa_mask,
    _prepare_mindiesd_lut,
)
from wan_5b.modules.sparse_attention import parse_sparse_config, sparse_method


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
        "hard_keep_sink_frames": 1,
        "hard_keep_recent_frames": 1,
        "dense_current_blocks": False,
        "first_chunk_dense": True,
    }
    values.update(overrides)
    return SLAAttentionConfig(**values)


def test_dispatcher_accepts_parsed_sla_config():
    config = _config()
    assert sparse_method(config) == "sla_cag"
    assert parse_sparse_config(config) is config


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


def test_dense_current_blocks_is_rejected():
    try:
        _config(dense_current_blocks=True).validate()
    except ValueError as error:
        assert "dense_current_blocks=false" in str(error)
    else:
        raise AssertionError("expected dense current blocks to be rejected")


def test_sla_global_block_ids_are_invariant_to_head_sharding():
    torch.manual_seed(23)
    q = torch.randn(1, 4, 4, 4)
    k = torch.randn(1, 12, 4, 4)
    config = _config(sparsity=0.5)
    full = build_sla_block_lut(
        q,
        k,
        frame_seq=2,
        sparsity=0.5,
        config=config,
    )
    shards = [
        build_sla_block_lut(
            q[:, :, start:start + 2],
            k[:, :, start:start + 2],
            frame_seq=2,
            sparsity=0.5,
            config=config,
        )
        for start in (0, 2)
    ]
    torch.testing.assert_close(torch.cat(shards, dim=1), full)


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


def test_folded_linear_projection_matches_explicit_projection_and_gradients():
    torch.manual_seed(31)
    config = _config(feature_map="elu", linear_cache=False)
    base_q = torch.randn(1, 4, 2, 4)
    base_k = torch.randn(1, 12, 2, 4)
    base_v = torch.randn_like(base_k)
    base_projection = torch.nn.Linear(4, 4)

    explicit_inputs = [
        tensor.clone().requires_grad_() for tensor in (base_q, base_k, base_v)
    ]
    folded_inputs = [
        tensor.clone().requires_grad_() for tensor in (base_q, base_k, base_v)
    ]
    explicit_projection = torch.nn.Linear(4, 4)
    folded_projection = torch.nn.Linear(4, 4)
    explicit_projection.load_state_dict(base_projection.state_dict())
    folded_projection.load_state_dict(base_projection.state_dict())

    explicit = explicit_projection(
        _linear_attention(
            *explicit_inputs,
            history_tokens=8,
            chunk_id=1,
            config=config,
            cache=None,
        )
    )
    folded = _projected_linear_attention(
        *folded_inputs,
        history_tokens=8,
        chunk_id=1,
        config=config,
        cache=None,
        projection=folded_projection,
    )
    torch.testing.assert_close(folded, explicit, rtol=2.0e-5, atol=2.0e-6)

    gradient = torch.randn_like(explicit)
    explicit.backward(gradient)
    folded.backward(gradient)
    for actual, expected in zip(folded_inputs, explicit_inputs):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(
        folded_projection.weight.grad,
        explicit_projection.weight.grad,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    torch.testing.assert_close(
        folded_projection.bias.grad,
        explicit_projection.bias.grad,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_parameterless_projection_wrapper_survives_checkpoint_recomputation():
    """模拟 FSDP 重计算时 PEFT wrapper 暂时不暴露 registered parameter。"""

    class ParameterlessProjection(torch.nn.Module):
        def __init__(self, linear):
            super().__init__()
            object.__setattr__(self, "linear", linear)

        def forward(self, value):
            return self.linear(value)

    torch.manual_seed(34)
    q = torch.randn(1, 4, 2, 4, requires_grad=True)
    k = torch.randn(1, 12, 2, 4, requires_grad=True)
    v = torch.randn(1, 12, 2, 4, requires_grad=True)
    linear = torch.nn.Linear(4, 4)
    projection = ParameterlessProjection(linear)
    assert list(projection.parameters()) == []

    def run(query, key, value):
        return _projected_linear_attention(
            query,
            key,
            value,
            history_tokens=8,
            chunk_id=1,
            config=_config(feature_map="elu", linear_cache=False),
            cache=None,
            projection=projection,
        )

    output = torch.utils.checkpoint.checkpoint(
        run, q, k, v, use_reentrant=False
    )
    output.square().mean().backward()

    assert all(tensor.grad is not None for tensor in (q, k, v))
    assert linear.weight.grad is not None and linear.weight.grad.abs().sum() > 0


def test_linear_attention_matches_upstream_sla_formula():
    torch.manual_seed(33)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    v = torch.randn_like(k)
    config = _config(feature_map="softmax", linear_cache=False, linear_eps=1.0e-5)

    output = _linear_attention(
        q,
        k,
        v,
        history_tokens=8,
        chunk_id=1,
        config=config,
        cache=None,
    )
    q_heads = torch.softmax(q, dim=-1).permute(0, 2, 1, 3)
    k_heads = torch.softmax(k, dim=-1).permute(0, 2, 1, 3)
    v_heads = v.permute(0, 2, 1, 3)
    kv_sum = k_heads.transpose(-1, -2) @ v_heads
    key_sum = k_heads.sum(dim=-2, keepdim=True)
    reference = (q_heads @ kv_sum) / (
        config.linear_eps + (q_heads * key_sum).sum(dim=-1, keepdim=True)
    )

    torch.testing.assert_close(output.permute(0, 2, 1, 3), reference)


def test_folded_linear_projection_stays_within_bf16_rounding_error():
    for feature_map in ("softmax", "elu", "relu"):
        torch.manual_seed(32)
        q = torch.randn(1, 16, 2, 8, dtype=torch.bfloat16)
        k = torch.randn(1, 64, 2, 8, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        projection = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
        config = _config(feature_map=feature_map, linear_cache=False)

        explicit = projection(
            _linear_attention(
                q,
                k,
                v,
                history_tokens=48,
                chunk_id=1,
                config=config,
                cache=None,
            )
        )
        folded = _projected_linear_attention(
            q,
            k,
            v,
            history_tokens=48,
            chunk_id=1,
            config=config,
            cache=None,
            projection=projection,
        )

        torch.testing.assert_close(folded, explicit, rtol=0.02, atol=0.002)


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


def test_router_matches_smooth_k_and_is_invariant_to_global_key_shift():
    torch.manual_seed(51)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    config = _config(
        sparsity=0.5,
        hard_keep_sink_frames=0,
        hard_keep_recent_frames=0,
    )

    actual = build_sla_block_lut(
        q,
        k,
        frame_seq=2,
        sparsity=config.sparsity,
        config=config,
    )
    q_blocks = q.reshape(1, 2, 2, 2, 4).mean(dim=2).permute(0, 2, 1, 3)
    smooth_k = k - k.mean(dim=1, keepdim=True)
    k_blocks = smooth_k.reshape(1, 6, 2, 2, 4).mean(dim=2).permute(0, 2, 1, 3)
    scores = q_blocks @ k_blocks.transpose(-1, -2)
    expected = torch.topk(scores, 3, dim=-1, sorted=False).indices.sort(dim=-1).values
    torch.testing.assert_close(actual, expected)

    shifted = build_sla_block_lut(
        q,
        k + torch.randn(1, 1, 2, 4),
        frame_seq=2,
        sparsity=config.sparsity,
        config=config,
    )
    torch.testing.assert_close(shifted, actual)


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


def test_ascend_kernel_tiles_fit_a2_a3_ub_budget():
    assert _kernel_tiles(40, 40) == (64, 64, 1, 1)
    assert _kernel_tiles(128, 128) == (64, 64, 2, 2)
    assert _kernel_tiles(65, 97) == (64, 64, 2, 2)


def test_ascend_kernel_tiles_reject_invalid_blocks():
    for block_q, block_k in ((0, 128), (128, 0), (129, 128), (128, 129)):
        try:
            _kernel_tiles(block_q, block_k)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid block sizes: {block_q}, {block_k}")
