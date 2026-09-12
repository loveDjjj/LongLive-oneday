"""CUDA extension selection can be checked without allocating GPU tensors."""

import importlib
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F


attention_module = importlib.import_module("wan_5b.modules.attention")


def test_disabled_fa3_uses_sdpa_when_fa2_is_missing():
    # Only the public input device is a stand-in. Transposes return real CPU
    # tensors, so the fallback itself executes and is compared numerically.
    torch.manual_seed(61)
    tensors = [torch.randn(1, length, 2, 8) for length in (4, 8, 8)]
    q, k, v = tensors
    cuda_q = SimpleNamespace(device=torch.device("cuda:0"), transpose=q.transpose)
    with mock.patch.multiple(
        attention_module,
        FLASH_ATTN_2_AVAILABLE=False,
        FLASH_ATTN_3_AVAILABLE=True,
        _USE_FA3=False,
        _USE_FA4=False,
        _USE_TE_ATTN=False,
    ), mock.patch.object(
        attention_module, "flash_attention", side_effect=AssertionError("FA3 is disabled")
    ):
        actual = attention_module.attention(
            cuda_q, k, v, dtype=torch.float32,
            softmax_scale=0.25, q_scale=0.5,
        )
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2) * 0.5, k.transpose(1, 2), v.transpose(1, 2),
        scale=0.25,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_explicit_fa3_request_reaches_flash_dispatch():
    q = SimpleNamespace(device=torch.device("cuda:0"))
    with mock.patch.multiple(
        attention_module,
        FLASH_ATTN_2_AVAILABLE=False,
        FLASH_ATTN_3_AVAILABLE=True,
        _USE_FA3=False,
        _USE_FA4=False,
        _USE_TE_ATTN=False,
    ), mock.patch.object(attention_module, "flash_attention", return_value="fa3") as run:
        assert attention_module.attention(q, None, None, fa_version=3) == "fa3"
    assert run.call_args.kwargs["version"] == 3
