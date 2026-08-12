import pytest
import torch

from utils.inference_utils import (
    clean_fsdp_state_dict_keys,
    configure_generator_linear_only,
    extract_generator_state_dict,
    load_generator_state_dict,
    load_generator_linear_checkpoint,
    load_generator_linear_state_dict,
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


def test_load_lora_state_dict_validates_sparse_method(tmp_path):
    path = tmp_path / "train_state.pt"
    torch.save(
        {
            "generator_lora": {"adapter.weight": torch.ones(1)},
            "sparse_method": "hsa_cag",
        },
        path,
    )

    with pytest.raises(ValueError, match="expected sla_cag"):
        load_lora_state_dict(str(path), expected_sparse_method="sla_cag")


def test_dense_checkpoint_synthesizes_zero_sla_projection():
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(2, 2)
            self.sla_linear = torch.nn.Linear(2, 2)

    generator = Generator()
    dense_state = {
        "dense.weight": torch.ones_like(generator.dense.weight),
        "dense.bias": torch.ones_like(generator.dense.bias),
    }
    load_generator_state_dict(generator, dense_state, strict=True)

    assert torch.count_nonzero(generator.sla_linear.weight) == 0
    assert torch.count_nonzero(generator.sla_linear.bias) == 0


def test_generator_linear_state_loads_only_compensation_parameters():
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(2, 2)
            self.sla_linear = torch.nn.Linear(2, 2)

    generator = Generator()
    dense_before = generator.dense.weight.detach().clone()
    state = {
        "sla_linear.weight": torch.ones_like(generator.sla_linear.weight),
        "sla_linear.bias": torch.ones_like(generator.sla_linear.bias),
    }

    load_generator_linear_state_dict(generator, state)

    assert torch.equal(generator.sla_linear.weight, state["sla_linear.weight"])
    assert torch.equal(generator.dense.weight, dense_before)


def test_generator_linear_state_accepts_fsdp_and_model_prefixes():
    generator = torch.nn.Module()
    generator.sla_linear = torch.nn.Linear(2, 2)
    state = {
        "_fsdp_wrapped_module.model.sla_linear.weight": torch.ones(2, 2),
        "_fsdp_wrapped_module.model.sla_linear.bias": torch.ones(2),
    }

    load_generator_linear_state_dict(generator, state)

    assert torch.count_nonzero(generator.sla_linear.weight) == 4
    assert torch.count_nonzero(generator.sla_linear.bias) == 2


def test_generator_linear_checkpoint_validates_method_and_scope(tmp_path):
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sla_linear = torch.nn.Linear(2, 2)

    generator = Generator()
    state = {
        "sla_linear.weight": torch.ones_like(generator.sla_linear.weight),
        "sla_linear.bias": torch.ones_like(generator.sla_linear.bias),
    }
    path = tmp_path / "linear.pt"
    torch.save(
        {
            "sparse_method": "hsa_sla_cag",
            "generator_train_scope": "linear_only",
            "generator_linear": state,
        },
        path,
    )

    load_generator_linear_checkpoint(
        generator, str(path), expected_sparse_method="hsa_sla_cag"
    )
    with pytest.raises(ValueError, match="expected sla_cag"):
        load_generator_linear_checkpoint(
            generator, str(path), expected_sparse_method="sla_cag"
        )


def test_generator_linear_state_rejects_missing_keys():
    generator = torch.nn.Module()
    generator.sla_linear = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match="keys do not match"):
        load_generator_linear_state_dict(
            generator, {"sla_linear.weight": torch.ones(2, 2)}
        )


def test_linear_only_scope_freezes_every_non_compensation_parameter():
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(2, 2)
            self.sla_linear = torch.nn.Linear(2, 2)

    generator = torch.nn.Sequential(Attention(), Attention())

    selected = configure_generator_linear_only(generator)
    actual = {
        name for name, parameter in generator.named_parameters()
        if parameter.requires_grad
    }

    assert actual == set(selected)
    assert actual == {
        "0.sla_linear.weight", "0.sla_linear.bias",
        "1.sla_linear.weight", "1.sla_linear.bias",
    }


def test_linear_only_scope_rejects_model_without_compensation():
    with pytest.raises(ValueError, match="no sla_linear parameters"):
        configure_generator_linear_only(torch.nn.Linear(2, 2))
