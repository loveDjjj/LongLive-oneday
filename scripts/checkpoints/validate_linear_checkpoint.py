#!/usr/bin/env python3
"""Validate a LongLive SLA raw-linear training checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

import torch


LINEAR_KEY = re.compile(
    r"(?:^|\.)blocks\.(?P<layer>\d+)\.self_attn\.sla_linear\."
    r"(?P<kind>weight|bias)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-method", default="hsa_sla_cag")
    parser.add_argument("--expected-layers", type=int, default=30)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument(
        "--allow-zero",
        action="store_true",
        help="allow a pristine zero-initialized projection (normally rejected)",
    )
    parser.add_argument(
        "--require-resume-state",
        action="store_true",
        help="also require critic, optimizer, RNG, and cursor state",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_checkpoint(
    checkpoint: object,
    *,
    expected_method: str,
    expected_layers: int,
    expected_step: int | None,
    allow_zero: bool,
    require_resume_state: bool,
) -> dict:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if checkpoint.get("sparse_method") != expected_method:
        raise ValueError(
            f"sparse_method={checkpoint.get('sparse_method')!r}, "
            f"expected {expected_method!r}"
        )
    scope = checkpoint.get("generator_train_scope")
    if scope not in {"linear_only", "lora_plus_linear"}:
        raise ValueError("generator_train_scope must be linear_only or lora_plus_linear")
    sidecar_format = checkpoint.get("checkpoint_format")
    expected_sidecar_formats = {
        "linear_only": "longlive_generator_linear_v1",
        "lora_plus_linear": "longlive_generator_adapter_v1",
    }
    if sidecar_format is not None and sidecar_format != expected_sidecar_formats[scope]:
        raise ValueError(f"unsupported generator linear checkpoint format: {sidecar_format!r}")
    if scope == "lora_plus_linear":
        generator_lora = checkpoint.get("generator_lora")
        if not isinstance(generator_lora, Mapping) or not generator_lora:
            raise ValueError("lora_plus_linear checkpoint is missing generator_lora")
    step = checkpoint.get("step")
    if not isinstance(step, int) or step < 0:
        raise ValueError(f"invalid checkpoint step: {step!r}")
    if expected_step is not None and step != expected_step:
        raise ValueError(f"checkpoint step={step}, expected {expected_step}")

    state = checkpoint.get("generator_linear")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint is missing generator_linear tensors")
    layers: dict[int, dict[str, torch.Tensor]] = {}
    unexpected = []
    for raw_name, tensor in state.items():
        name = str(raw_name).replace("_fsdp_wrapped_module.", "")
        match = LINEAR_KEY.search(name)
        if match is None or not isinstance(tensor, torch.Tensor):
            unexpected.append(str(raw_name))
            continue
        layer = int(match.group("layer"))
        layers.setdefault(layer, {})[match.group("kind")] = tensor
    if unexpected:
        raise ValueError(f"unexpected generator_linear entries: {unexpected[:4]}")
    expected_ids = set(range(expected_layers))
    if set(layers) != expected_ids:
        raise ValueError(
            f"linear layer ids mismatch: found={sorted(layers)} "
            f"expected={sorted(expected_ids)}"
        )

    tensor_count = 0
    parameter_count = 0
    nonzero_count = 0
    squared_norm = 0.0
    dtypes = set()
    for layer, tensors in sorted(layers.items()):
        if set(tensors) != {"weight", "bias"}:
            raise ValueError(f"layer {layer} must contain weight and bias")
        weight, bias = tensors["weight"], tensors["bias"]
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError(f"layer {layer} weight must be square, got {tuple(weight.shape)}")
        if bias.shape != (weight.shape[0],):
            raise ValueError(
                f"layer {layer} bias shape {tuple(bias.shape)} does not match "
                f"weight {tuple(weight.shape)}"
            )
        for tensor in (weight, bias):
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"layer {layer} contains non-finite values")
            values = tensor.float()
            tensor_count += 1
            parameter_count += tensor.numel()
            nonzero_count += torch.count_nonzero(tensor).item()
            squared_norm += values.square().sum().item()
            dtypes.add(str(tensor.dtype).removeprefix("torch."))
    if nonzero_count == 0 and not allow_zero:
        raise ValueError(
            "all generator_linear tensors are zero; no trained update was detected"
        )

    trainable = checkpoint.get("generator_trainable_parameters")
    if trainable is not None:
        if not isinstance(trainable, (list, tuple)) or not trainable:
            raise ValueError("generator_trainable_parameters must be a non-empty list")
        invalid = [
            name for name in trainable
            if ".sla_linear." not in str(name) and "lora_" not in str(name)
        ]
        if invalid:
            raise ValueError(f"non-linear trainable generator parameters: {invalid[:4]}")

    if require_resume_state:
        if checkpoint.get("checkpoint_format_version") != 4:
            raise ValueError(
                "full resume checkpoint must use checkpoint_format_version=4"
            )
        required = {
            "critic_lora",
            "generator_optimizer",
            "critic_optimizer",
            "rng_states",
            "global_samples_consumed",
            "gradient_accumulation_steps",
            "batch_size",
            "world_size",
            "sequence_parallel_size",
            "data_parallel_size",
        }
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(f"checkpoint is missing resume state: {missing}")

    return {
        "schema_version": 1,
        "step": step,
        "sparse_method": expected_method,
        "generator_train_scope": scope,
        "layers": len(layers),
        "tensors": tensor_count,
        "parameters": parameter_count,
        "nonzero_parameters": nonzero_count,
        "nonzero_fraction": nonzero_count / parameter_count,
        "l2_norm": math.sqrt(squared_norm),
        "dtypes": sorted(dtypes),
        "resume_state_validated": require_resume_state,
    }


def main() -> None:
    args = parse_args()
    result = validate_checkpoint(
        _load(args.checkpoint),
        expected_method=args.expected_method,
        expected_layers=args.expected_layers,
        expected_step=args.expected_step,
        allow_zero=args.allow_zero,
        require_resume_state=args.require_resume_state,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        "linear_checkpoint=passed "
        f"step={result['step']} layers={result['layers']} "
        f"tensors={result['tensors']} parameters={result['parameters']} "
        f"nonzero_fraction={result['nonzero_fraction']:.6f} "
        f"l2_norm={result['l2_norm']:.6f} "
        f"resume_state={result['resume_state_validated']}"
    )


if __name__ == "__main__":
    main()
