import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.checkpoints.validate_linear_checkpoint import validate_checkpoint


def _linear_state(layers=3, value=1.0):
    state = {}
    for layer in range(layers):
        prefix = f"blocks.{layer}.self_attn.sla_linear"
        state[f"{prefix}.weight"] = torch.full((4, 4), value)
        state[f"{prefix}.bias"] = torch.full((4,), value)
    return state


def _checkpoint(**overrides):
    checkpoint = {
        "generator_linear": _linear_state(),
        "step": 1000,
        "checkpoint_format": "longlive_generator_linear_v1",
        "generator_train_scope": "linear_only",
        "generator_trainable_parameters": [
            f"model.blocks.{layer}.self_attn.sla_linear.{kind}"
            for layer in range(3)
            for kind in ("weight", "bias")
        ],
        "sparse_method": "hsa_sla_cag",
        "world_size": 12,
        "sequence_parallel_size": 4,
        "data_parallel_size": 3,
    }
    checkpoint.update(overrides)
    return checkpoint


def test_validates_linear_sidecar():
    result = validate_checkpoint(
        _checkpoint(),
        expected_method="hsa_sla_cag",
        expected_layers=3,
        expected_step=1000,
        allow_zero=False,
        require_resume_state=False,
    )

    assert result["layers"] == 3
    assert result["tensors"] == 6
    assert result["parameters"] == 60
    assert result["nonzero_parameters"] == 60


def test_validates_full_resume_checkpoint():
    checkpoint = _checkpoint(
        checkpoint_format_version=4,
        critic_lora={"x": torch.ones(1)},
        generator_optimizer={"state": {}},
        critic_optimizer={"state": {}},
        rng_states=[{}],
        global_samples_consumed=3000,
        gradient_accumulation_steps=1,
        batch_size=1,
    )

    result = validate_checkpoint(
        checkpoint,
        expected_method="hsa_sla_cag",
        expected_layers=3,
        expected_step=None,
        allow_zero=False,
        require_resume_state=True,
    )

    assert result["resume_state_validated"] is True


def test_validates_lora_plus_linear_adapter_sidecar():
    checkpoint = _checkpoint(
        checkpoint_format="longlive_generator_adapter_v1",
        generator_train_scope="lora_plus_linear",
        generator_lora={"lora_A.weight": torch.ones(2, 2)},
        generator_trainable_parameters=[
            "model.blocks.0.self_attn.sla_linear.weight",
            "model.blocks.0.self_attn.q.lora_A.default.weight",
        ],
    )

    result = validate_checkpoint(
        checkpoint,
        expected_method="hsa_sla_cag",
        expected_layers=3,
        expected_step=1000,
        allow_zero=False,
        require_resume_state=False,
    )

    assert result["generator_train_scope"] == "lora_plus_linear"


def test_cli_validates_saved_sidecar_and_writes_json(tmp_path):
    checkpoint_path = tmp_path / "generator_linear.pt"
    output_path = tmp_path / "validation.json"
    torch.save(_checkpoint(), checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/checkpoints/validate_linear_checkpoint.py"
            ),
            str(checkpoint_path),
            "--expected-layers",
            "3",
            "--expected-step",
            "1000",
            "--json-output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "linear_checkpoint=passed" in result.stdout
    assert json.loads(output_path.read_text(encoding="utf-8"))["layers"] == 3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sparse_method": "sla_cag"}, "sparse_method"),
        ({"generator_train_scope": "lora"}, "linear_only"),
        ({"checkpoint_format": "unknown"}, "unsupported"),
        ({"step": 999}, "expected 1000"),
        ({"generator_linear": _linear_state(layers=2)}, "layer ids mismatch"),
        ({"generator_linear": _linear_state(value=0.0)}, "all generator_linear"),
    ],
)
def test_rejects_invalid_linear_checkpoint(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_checkpoint(
            _checkpoint(**overrides),
            expected_method="hsa_sla_cag",
            expected_layers=3,
            expected_step=1000,
            allow_zero=False,
            require_resume_state=False,
        )


def test_rejects_nonfinite_linear_weight():
    state = _linear_state()
    state["blocks.1.self_attn.sla_linear.weight"][0, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        validate_checkpoint(
            _checkpoint(generator_linear=state),
            expected_method="hsa_sla_cag",
            expected_layers=3,
            expected_step=None,
            allow_zero=False,
            require_resume_state=False,
        )


def test_allow_zero_is_explicit():
    result = validate_checkpoint(
        _checkpoint(generator_linear=_linear_state(value=0.0)),
        expected_method="hsa_sla_cag",
        expected_layers=3,
        expected_step=None,
        allow_zero=True,
        require_resume_state=False,
    )

    assert result["nonzero_fraction"] == 0.0
