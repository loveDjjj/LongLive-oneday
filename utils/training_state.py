from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from utils.inference_utils import is_sla_linear_parameter


def build_generator_linear_sidecar(checkpoint):
    """Extract the portable hybrid generator state from a full train checkpoint."""
    required = {
        "generator_linear",
        "step",
        "generator_train_scope",
        "generator_trainable_parameters",
        "sparse_method",
        "world_size",
        "sequence_parallel_size",
        "data_parallel_size",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"cannot build generator linear sidecar; missing {missing}")
    if checkpoint["generator_train_scope"] != "linear_only":
        raise ValueError("generator linear sidecar requires linear_only scope")
    return {
        "generator_linear": checkpoint["generator_linear"],
        "step": checkpoint["step"],
        "checkpoint_format": "longlive_generator_linear_v1",
        "generator_train_scope": checkpoint["generator_train_scope"],
        "generator_trainable_parameters": checkpoint[
            "generator_trainable_parameters"
        ],
        "sparse_method": checkpoint["sparse_method"],
        "world_size": checkpoint["world_size"],
        "sequence_parallel_size": checkpoint["sequence_parallel_size"],
        "data_parallel_size": checkpoint["data_parallel_size"],
    }


def build_generator_adapter_sidecar(checkpoint):
    """Extract LoRA and raw SLA compensation tensors for hybrid inference."""
    required = {
        "generator_lora",
        "generator_linear",
        "step",
        "generator_train_scope",
        "generator_trainable_parameters",
        "sparse_method",
        "world_size",
        "sequence_parallel_size",
        "data_parallel_size",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"cannot build generator adapter sidecar; missing {missing}")
    if checkpoint["generator_train_scope"] != "lora_plus_linear":
        raise ValueError("generator adapter sidecar requires lora_plus_linear scope")
    return {
        "generator_lora": checkpoint["generator_lora"],
        "generator_linear": checkpoint["generator_linear"],
        "step": checkpoint["step"],
        "checkpoint_format": "longlive_generator_adapter_v1",
        "generator_train_scope": checkpoint["generator_train_scope"],
        "generator_trainable_parameters": checkpoint[
            "generator_trainable_parameters"
        ],
        "sparse_method": checkpoint["sparse_method"],
        "world_size": checkpoint["world_size"],
        "sequence_parallel_size": checkpoint["sequence_parallel_size"],
        "data_parallel_size": checkpoint["data_parallel_size"],
    }


def validate_sparse_checkpoint_method(checkpoint, expected_method):
    """Prevent resuming optimizer/LoRA state from another sparse algorithm."""
    actual_method = checkpoint.get("sparse_method")
    if actual_method != expected_method:
        actual_label = actual_method if actual_method is not None else "unlabeled/legacy"
        raise ValueError(
            f"checkpoint sparse method is {actual_label}, expected {expected_method}; "
            "start a new run directory or provide a matching checkpoint"
        )


def list_training_checkpoints(output_dir):
    """Return one checkpoint per step, preferring the current layout."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []

    checkpoints = {}
    for checkpoint_file in output_dir.glob("checkpoint_model_*/model.pt"):
        try:
            step = int(checkpoint_file.parent.name.removeprefix("checkpoint_model_"))
        except ValueError:
            continue
        checkpoints[step] = (
            step,
            str(checkpoint_file.parent),
            checkpoint_file.parent.name,
            str(checkpoint_file),
        )

    for checkpoint_file in (output_dir / "checkpoints").glob("step_*/train_state.pt"):
        try:
            step = int(checkpoint_file.parent.name.removeprefix("step_"))
        except ValueError:
            continue
        checkpoints[step] = (
            step,
            str(checkpoint_file.parent),
            checkpoint_file.parent.name,
            str(checkpoint_file),
        )

    return [checkpoints[step] for step in sorted(checkpoints)]


def find_latest_training_checkpoint(output_dir):
    checkpoints = list_training_checkpoints(output_dir)
    return checkpoints[-1][3] if checkpoints else None


def should_save_final_checkpoint(*, start_step, final_step, save_interval, no_save):
    """Return whether a completed run needs a checkpoint outside its cadence."""
    return (
        not no_save
        and final_step > start_step
        and final_step % save_interval != 0
    )


def build_generator_optimizer_parameters(
    named_parameters, *, scope, lora_lr, linear_lr=None
):
    """Build AdamW parameters, separating raw SLA compensation from LoRA."""
    trainable = [
        (str(name), parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    if scope != "lora_plus_linear":
        return [parameter for _, parameter in trainable]
    if linear_lr is None or float(linear_lr) <= 0.0:
        raise ValueError("lora_plus_linear requires a positive linear_lr")
    linear = [
        parameter for name, parameter in trainable
        if is_sla_linear_parameter(name)
    ]
    linear_ids = {id(parameter) for parameter in linear}
    lora = [parameter for _, parameter in trainable if id(parameter) not in linear_ids]
    if not linear or not lora:
        raise ValueError("lora_plus_linear requires both LoRA and sla_linear parameters")
    return [
        {"params": lora, "lr": float(lora_lr)},
        {"params": linear, "lr": float(linear_lr)},
    ]


def linear_gradient_statistics(named_parameters, *, reduce_tensors=None):
    """Summarize sharded SLA-linear gradients using a caller-provided reducer."""
    selected = sorted(
        (str(name), parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad and ".sla_linear." in f".{name}"
    )
    if not selected:
        raise ValueError("linear gradient diagnostics found no trainable sla_linear tensors")

    device = selected[0][1].device
    present = torch.zeros(len(selected), dtype=torch.float32, device=device)
    nonfinite = torch.zeros_like(present)
    squared_norm = torch.zeros_like(present)
    for index, (_, parameter) in enumerate(selected):
        gradient = parameter.grad
        if gradient is None or gradient.numel() == 0:
            continue
        present[index] = 1.0
        finite = torch.isfinite(gradient)
        nonfinite[index] = (~finite).any().to(torch.float32)
        squared_norm[index] = torch.where(
            finite, gradient.float(), torch.zeros_like(gradient, dtype=torch.float32)
        ).square().sum()

    if reduce_tensors is not None:
        reduce_tensors(present, nonfinite, squared_norm)

    norms = squared_norm.sqrt()
    missing_indices = torch.nonzero(present == 0, as_tuple=False).flatten().cpu().tolist()
    nonfinite_indices = torch.nonzero(nonfinite > 0, as_tuple=False).flatten().cpu().tolist()
    zero_indices = torch.nonzero(
        (present > 0) & (nonfinite == 0) & (squared_norm == 0), as_tuple=False
    ).flatten().cpu().tolist()
    active_norms = norms[(present > 0) & (nonfinite == 0)]
    return {
        "tensor_count": len(selected),
        "with_gradient": int((present > 0).sum().item()),
        "nonzero": int(
            ((present > 0) & (nonfinite == 0) & (squared_norm > 0)).sum().item()
        ),
        "finite": int(((present > 0) & (nonfinite == 0)).sum().item()),
        "min_aggregated_l2": (
            float(active_norms.min().item()) if active_norms.numel() else 0.0
        ),
        "max_aggregated_l2": (
            float(active_norms.max().item()) if active_norms.numel() else 0.0
        ),
        "missing": [selected[index][0] for index in missing_indices],
        "nonfinite_names": [selected[index][0] for index in nonfinite_indices],
        "zero": [selected[index][0] for index in zero_indices],
    }


def resume_samples_per_rank(
    checkpoint,
    *,
    step,
    current_data_parallel_size,
    current_sequence_parallel_size,
    current_batch_size,
    current_accumulation_steps,
):
    """Convert a global checkpoint cursor into samples consumed by each DP rank."""
    saved_world_size = int(
        checkpoint.get(
            "world_size",
            current_data_parallel_size * current_sequence_parallel_size,
        )
    )
    inferred_sp_size = "sequence_parallel_size" not in checkpoint
    saved_sp_size = int(
        checkpoint.get("sequence_parallel_size", current_sequence_parallel_size)
    )
    saved_dp_size = int(
        checkpoint.get(
            "data_parallel_size",
            saved_world_size // max(saved_sp_size, 1),
        )
    )
    saved_batch_size = int(checkpoint.get("batch_size", current_batch_size))
    saved_accumulation = int(
        checkpoint.get(
            "gradient_accumulation_steps", current_accumulation_steps
        )
    )

    # Version-2 checkpoints counted SP workers as independent samples. Rebuild
    # their cursor from the saved logical DP layout instead of trusting it.
    if "data_parallel_size" in checkpoint:
        global_samples = int(
            checkpoint.get(
                "global_samples_consumed",
                step * saved_dp_size * saved_batch_size * saved_accumulation,
            )
        )
    else:
        global_samples = (
            step * saved_dp_size * saved_batch_size * saved_accumulation
        )

    denominator = current_data_parallel_size * current_batch_size
    samples, remainder = divmod(global_samples, denominator)
    metadata = {
        "global_samples": global_samples,
        "remainder": remainder,
        "saved_world_size": saved_world_size,
        "saved_data_parallel_size": saved_dp_size,
        "saved_sequence_parallel_size": saved_sp_size,
        "inferred_sequence_parallel_size": inferred_sp_size,
        "saved_batch_size": saved_batch_size,
        "saved_accumulation_steps": saved_accumulation,
    }
    return samples * current_batch_size, metadata


def restore_fsdp_optimizer_state(fsdp_cls, model, optimizer, full_state_dict):
    local_state_dict = fsdp_cls.optim_state_dict_to_load(
        model, optimizer, full_state_dict
    )
    optimizer.load_state_dict(local_state_dict)


def capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state().cpu(),
    }
    if hasattr(torch, "npu") and torch.npu.is_available():
        state["accelerator"] = torch.npu.get_rng_state().cpu()
        state["accelerator_type"] = "npu"
    elif torch.cuda.is_available():
        state["accelerator"] = torch.cuda.get_rng_state().cpu()
        state["accelerator_type"] = "cuda"
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        numpy_state["keys"].cpu().numpy(),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"])

    accelerator_state = state.get("accelerator")
    if accelerator_state is None:
        return
    if state.get("accelerator_type") == "npu" and hasattr(torch, "npu"):
        torch.npu.set_rng_state(accelerator_state)
    elif state.get("accelerator_type") == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(accelerator_state)
