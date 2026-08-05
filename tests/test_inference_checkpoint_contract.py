import pytest
import torch

from utils.inference_utils import (
    clean_fsdp_state_dict_keys,
    extract_generator_state_dict,
    load_lora_state_dict,
)


def test_extract_generator_prefers_explicit_generator():
    state = {"weight": torch.ones(1)}
    assert extract_generator_state_dict({"generator": state}) is state


def test_extract_generator_requires_requested_ema():
    with pytest.raises(KeyError, match="generator_ema"):
        extract_generator_state_dict({"generator": {"weight": torch.ones(1)}}, use_ema=True)


def test_extract_generator_rejects_lora_only_checkpoint():
    with pytest.raises(ValueError, match="only LoRA"):
        extract_generator_state_dict({"generator_lora": {"weight": torch.ones(1)}})


def test_extract_generator_rejects_training_metadata_as_state_dict():
    with pytest.raises(ValueError, match="Unrecognized checkpoint layout"):
        extract_generator_state_dict({"step": 100, "optimizer": {}})


def test_clean_fsdp_state_dict_keys():
    state = {"_fsdp_wrapped_module.model.weight": torch.ones(1)}
    assert list(clean_fsdp_state_dict_keys(state)) == ["model.weight"]


def test_load_lora_state_dict_unwraps_generator_lora(tmp_path):
    path = tmp_path / "train_state.pt"
    expected = {"adapter.weight": torch.ones(1)}
    torch.save({"generator_lora": expected, "step": 10}, path)

    actual = load_lora_state_dict(str(path))

    assert actual.keys() == expected.keys()
    assert torch.equal(actual["adapter.weight"], expected["adapter.weight"])
