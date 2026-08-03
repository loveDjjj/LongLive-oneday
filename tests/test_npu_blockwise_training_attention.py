import torch

from wan_5b.modules import causal_model


def _mean_value_attention(q, _k, v):
    value = v.mean(dim=1, keepdim=True)
    return value.expand(-1, q.shape[1], -1, -1)


def test_npu_causal_attention_uses_chunk_history(monkeypatch):
    monkeypatch.setattr(causal_model, "attention", _mean_value_attention)
    q = torch.zeros(1, 4, 1, 1)
    k = torch.zeros_like(q)
    v = torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(1, 4, 1, 1)
    mask = causal_model._NpuBlockwiseMask("causal", 4, 1, 2)

    output = causal_model._npu_blockwise_attention(q, k, v, mask)

    expected = torch.tensor([1.5, 1.5, 3.75, 3.75]).reshape(1, 4, 1, 1)
    torch.testing.assert_close(output, expected)


def test_npu_i2v_attention_keeps_first_frame_independent(monkeypatch):
    monkeypatch.setattr(causal_model, "attention", _mean_value_attention)
    q = torch.zeros(1, 5, 1, 1)
    k = torch.zeros_like(q)
    v = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0]).reshape(1, 5, 1, 1)
    mask = causal_model._NpuBlockwiseMask("i2v", 5, 1, 2)

    output = causal_model._npu_blockwise_attention(q, k, v, mask)

    expected = torch.tensor([
        1.0, 7.0 / 3.0, 7.0 / 3.0, 31.0 / 5.0, 31.0 / 5.0
    ]).reshape(1, 5, 1, 1)
    torch.testing.assert_close(output, expected)


def test_npu_teacher_attention_separates_clean_and_noisy_history(monkeypatch):
    monkeypatch.setattr(causal_model, "attention", _mean_value_attention)
    q = torch.zeros(1, 8, 1, 1)
    k = torch.zeros_like(q)
    v = torch.tensor([
        1.0, 2.0, 4.0, 8.0,
        16.0, 32.0, 64.0, 128.0,
    ]).reshape(1, 8, 1, 1)
    mask = causal_model._NpuBlockwiseMask("teacher", 4, 1, 2)

    output = causal_model._npu_blockwise_attention(q, k, v, mask)

    expected = torch.tensor([
        1.5, 1.5, 3.75, 3.75,
        24.0, 24.0, 48.75, 48.75,
    ]).reshape(1, 8, 1, 1)
    torch.testing.assert_close(output, expected)


def test_npu_mask_creation_does_not_call_flex_attention(monkeypatch):
    def fail_create_block_mask(*_args, **_kwargs):
        raise AssertionError("NPU mask creation must not invoke FlexAttention")

    monkeypatch.setattr(causal_model, "create_block_mask", fail_create_block_mask)
    mask = causal_model.CausalWanModel._prepare_blockwise_causal_attn_mask(
        "npu:0", num_frames=16, frame_seqlen=880, num_frame_per_block=8
    )

    assert mask == causal_model._NpuBlockwiseMask("causal", 16, 880, 8)
