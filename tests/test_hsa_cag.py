import torch

from wan_5b.modules.sparse_attention import (
    SparseAttentionConfig,
    _history_block_indices,
    calculate_chunk_sparsities,
    hierarchical_sparse_attention,
)
from wan_5b.modules.sparse_attention_ascend import _requires_autograd
from wan_5b.modules.sparse_attention_mindiesd import _prepare_mindiesd_lut


def test_cag_starts_dense_and_preserves_average_budget():
    config = SparseAttentionConfig(
        enabled=True,
        sparsity=0.85,
        sparsity_base=0.95,
    )
    schedule = calculate_chunk_sparsities(32, 4, 32, config)

    assert schedule[0] == 0.0
    assert len(schedule) == 8
    assert all(0.0 <= value < 1.0 for value in schedule)
    assert schedule[1] < schedule[-1]

    frame_counts = list(range(8, 33, 4))
    kv_lengths = [min(count, 32) for count in frame_counts]
    scheduled = sum((1.0 - value) * length for value, length in zip(schedule[1:], kv_lengths))
    target = sum((1.0 - config.sparsity) * length for length in kv_lengths)
    assert abs(scheduled - target) < 1e-6


def test_hsa_first_chunk_matches_dense_attention():
    torch.manual_seed(0)
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn(1, 8, 2, 4)
    v = torch.randn(1, 8, 2, 4)
    config = {
        "enabled": True,
        "sparsity": 0.75,
        "block_q": 2,
        "block_k": 2,
        "keep_frames": 1,
        "keep_sink": 1,
        "keep_near": 0,
    }

    actual = hierarchical_sparse_attention(
        q, k, v, frame_seq=4, chunk_id=0, sparse_config=config
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_hsa_is_differentiable_and_keeps_shape():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 2, 4, requires_grad=True)
    k = torch.randn(1, 12, 2, 4, requires_grad=True)
    v = torch.randn(1, 12, 2, 4, requires_grad=True)
    config = {
        "enabled": True,
        "sparsity": 0.5,
        "sparsity_base": 0.75,
        "block_q": 2,
        "block_k": 2,
        "keep_frames": 2,
        "keep_sink": 1,
        "keep_near": 1,
        "dense_current": True,
        "num_output_frames": 6,
        "num_frame_per_block": 1,
    }

    output = hierarchical_sparse_attention(
        q, k, v, frame_seq=4, chunk_id=2, sparse_config=config
    )
    assert output.shape == q.shape
    output.square().mean().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_hsa_query_block_batching_preserves_output():
    torch.manual_seed(2)
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn(1, 24, 2, 4)
    v = torch.randn(1, 24, 2, 4)
    config = {
        "enabled": True,
        "sparsity": 0.5,
        "block_q": 2,
        "block_k": 2,
        "keep_frames": 2,
        "keep_sink": 1,
        "keep_near": 1,
        "dense_current": True,
    }

    unbatched = hierarchical_sparse_attention(
        q,
        k,
        v,
        frame_seq=4,
        chunk_id=2,
        sparse_config={**config, "query_block_batch": 1},
    )
    batched = hierarchical_sparse_attention(
        q,
        k,
        v,
        frame_seq=4,
        chunk_id=2,
        sparse_config={**config, "query_block_batch": 2},
    )
    torch.testing.assert_close(batched, unbatched)


def test_hsa_auto_backend_falls_back_to_portable_on_cpu():
    torch.manual_seed(3)
    q = torch.randn(1, 4, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    v = torch.randn(1, 12, 2, 4)
    config = {
        "enabled": True,
        "sparsity": 0.5,
        "block_q": 2,
        "block_k": 2,
        "keep_frames": 2,
        "keep_sink": 1,
        "keep_near": 1,
    }
    portable = hierarchical_sparse_attention(
        q, k, v, frame_seq=4, chunk_id=2, sparse_config=config
    )
    automatic = hierarchical_sparse_attention(
        q, k, v, frame_seq=4, chunk_id=2, sparse_config={**config, "backend": "auto"}
    )
    torch.testing.assert_close(automatic, portable)


def test_hsa_explicit_ascend_backend_rejects_cpu():
    q = torch.randn(1, 4, 1, 4)
    k = torch.randn(1, 8, 1, 4)
    v = torch.randn(1, 8, 1, 4)
    config = {
        "enabled": True,
        "backend": "ascend_triton",
        "sparsity": 0.5,
        "block_q": 2,
        "block_k": 2,
        "keep_frames": 1,
        "keep_sink": 1,
        "keep_near": 0,
    }
    try:
        hierarchical_sparse_attention(
            q, k, v, frame_seq=4, chunk_id=2, sparse_config=config
        )
    except RuntimeError as error:
        assert "not npu" in str(error)
    else:
        raise AssertionError("Expected explicit Ascend backend to reject CPU tensors")


def test_hsa_backend_aliases_are_normalized():
    assert SparseAttentionConfig.from_mapping({"backend": "triton"}).backend == "ascend_triton"
    assert SparseAttentionConfig.from_mapping({"backend": "torch"}).backend == "portable"
    assert SparseAttentionConfig.from_mapping({"backend": "rainfusion"}).backend == "mindiesd"


def test_enabled_mindiesd_backend_requires_128_blocks():
    try:
        SparseAttentionConfig.from_mapping({
            "enabled": True,
            "backend": "mindiesd",
            "block_q": 40,
            "block_k": 40,
        })
    except ValueError as error:
        assert "block_q=block_k=128" in str(error)
    else:
        raise AssertionError("Expected MindIE-SD block-size validation to fail")


def test_hsa_rejects_misaligned_frame_blocks():
    q = torch.randn(1, 4, 1, 2)
    k = torch.randn(1, 8, 1, 2)
    v = torch.randn(1, 8, 1, 2)
    config = {"enabled": True, "block_k": 3}

    try:
        hierarchical_sparse_attention(
            q, k, v, frame_seq=4, chunk_id=1, sparse_config=config
        )
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Expected frame/block alignment validation to fail")


def test_hsa_accepts_execution_blocks_that_do_not_divide_a_frame():
    torch.manual_seed(4)
    q = torch.randn(1, 6, 1, 4)
    k = torch.randn(1, 12, 1, 4)
    v = torch.randn(1, 12, 1, 4)
    config = {
        "enabled": True,
        "sparsity": 0.5,
        "block_q": 2,
        "block_k": 2,
        "keep_frames": 1,
        "keep_sink": 1,
        "keep_near": 0,
    }

    output = hierarchical_sparse_attention(
        q, k, v, frame_seq=3, chunk_id=1, sparse_config=config
    )

    assert output.shape == q.shape
    assert torch.isfinite(output).all()


def test_hsa_builds_unique_128_block_lut_for_sp4_tail_shape():
    torch.manual_seed(5)
    frame_seq = 880
    block = 128
    q = torch.randn(1, 7040, 1, 2)
    history_k = torch.randn(1, 21120, 1, 2)
    full_k = torch.cat([history_k, q], dim=1)
    q_blocks = q.reshape(1, 55, block, 1, 2).mean(dim=2)
    k_blocks = full_k.reshape(1, 220, block, 1, 2).mean(dim=2)
    config = SparseAttentionConfig(
        enabled=True,
        block_q=block,
        block_k=block,
        keep_frames=6,
        keep_sink=1,
        keep_near=2,
    )

    selected = _history_block_indices(
        q_blocks,
        k_blocks,
        history_k,
        history_frames=24,
        frame_seq=frame_seq,
        block_k=block,
        history_keep_blocks=10,
        config=config,
    )

    assert selected.shape == (1, 1, 55, 10)
    assert selected.min() >= 0
    assert selected.max() < 165
    assert all(row.unique().numel() == 10 for row in selected.flatten(0, 2))


def test_explicit_mindiesd_backend_requires_npu():
    q = torch.randn(1, 256, 1, 4)
    k = torch.randn(1, 512, 1, 4)
    v = torch.randn(1, 512, 1, 4)
    config = {
        "enabled": True,
        "backend": "mindiesd",
        "sparsity": 0.5,
        "block_q": 128,
        "block_k": 128,
        "keep_frames": 1,
        "keep_sink": 1,
        "keep_near": 0,
    }

    with torch.no_grad():
        try:
            hierarchical_sparse_attention(
                q, k, v, frame_seq=128, chunk_id=1, sparse_config=config
            )
        except RuntimeError as error:
            assert "not npu" in str(error)
        else:
            raise AssertionError("Expected explicit MindIE-SD backend to reject CPU tensors")


def test_mindiesd_lut_stays_compact_and_uses_valid_counts():
    compact = torch.tensor([[[[0, 3], [1, 2]], [[2, 3], [0, 1]]]])

    select_idx, select_num = _prepare_mindiesd_lut(compact, k_blocks=4)

    assert select_idx.shape == (2, 2, 2)
    assert select_idx.dtype == torch.int64
    assert torch.equal(select_num, torch.full((2, 2), 2, dtype=torch.int64))
    assert torch.equal(select_idx, compact[0].permute(1, 0, 2))


def test_mindiesd_lut_rejects_more_selected_than_available_blocks():
    invalid = torch.zeros((1, 1, 1, 3), dtype=torch.int64)

    try:
        _prepare_mindiesd_lut(invalid, k_blocks=2)
    except ValueError as error:
        assert "selects 3 blocks" in str(error)
    else:
        raise AssertionError("Expected oversized MindIE-SD LUT to fail")


def test_mindiesd_lut_rejects_non_integral_indices():
    invalid = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

    try:
        _prepare_mindiesd_lut(invalid, k_blocks=1)
    except TypeError as error:
        assert "int32 or int64" in str(error)
    else:
        raise AssertionError("Expected non-integral MindIE-SD LUT to fail")


def test_hsa_rejects_sparse_current_chunk():
    try:
        SparseAttentionConfig.from_mapping(
            {"enabled": True, "dense_current": False, "keep_frames": 1}
        )
    except ValueError as error:
        assert "dense_current=true" in str(error)
    else:
        raise AssertionError("Expected sparse current-chunk routing to be rejected")


def test_ascend_hsa_uses_forward_only_path_without_gradients():
    tensor = torch.randn(2, requires_grad=True)

    assert _requires_autograd(tensor, tensor, tensor)
    with torch.no_grad():
        assert not _requires_autograd(tensor, tensor, tensor)
    assert not _requires_autograd(tensor.detach(), tensor.detach(), tensor.detach())
