#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Merge a LongLive generator checkpoint with LoRA weights for simple inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True, help="Inference yaml containing model, checkpoint, and adapter settings.")
    parser.add_argument("--output_path", required=True, help="Path to save the merged generator checkpoint.")
    parser.add_argument("--generator_ckpt", default=None, help="Override checkpoints.generator_ckpt from the yaml.")
    parser.add_argument("--lora_ckpt", default=None, help="Override checkpoints.lora_ckpt from the yaml.")
    parser.add_argument(
        "--generator_train_scope",
        choices=("lora", "linear_only", "lora_plus_linear"),
        help="Override training.generator_train_scope for adapter export.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for merging, e.g. cuda:0 or cpu.")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", help="Save merged weights in this dtype.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from omegaconf import OmegaConf

    from utils.config import normalize_config
    from utils.inference_utils import (
        cpu_state_dict,
        load_generator_checkpoint,
        load_generator_linear_checkpoint,
        load_lora_state_dict,
    )
    from utils.lora_utils import (
        configure_lora_for_model,
        set_peft_model_state_dict_for_ulysses,
    )
    from utils.wan_5b_wrapper import WanDiffusionWrapper

    config = normalize_config(OmegaConf.load(args.config_path))
    sparse_config = getattr(config.model_kwargs, "sparse_config", {})
    sparse_method = sparse_config.get(
        "method", "hsa_cag" if "keep_frames" in sparse_config else "sla_cag"
    )
    generator_train_scope = str(
        args.generator_train_scope
        or getattr(config, "generator_train_scope", "lora")
    )
    generator_ckpt = args.generator_ckpt or getattr(config, "generator_ckpt", None)
    lora_ckpt = args.lora_ckpt or getattr(config, "lora_ckpt", None)
    if not generator_ckpt:
        raise ValueError("Missing generator checkpoint. Set checkpoints.generator_ckpt or pass --generator_ckpt.")
    if not lora_ckpt:
        raise ValueError("Missing LoRA checkpoint. Set checkpoints.lora_ckpt or pass --lora_ckpt.")
    if not getattr(config, "adapter", None):
        raise ValueError("Missing adapter config. The merge script needs the LoRA rank/alpha/dropout settings.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"Building generator: {config.model_kwargs}")
    generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)
    generator.eval().requires_grad_(False)

    print(f"Loading generator checkpoint: {generator_ckpt}")
    incompatible = load_generator_checkpoint(
        generator,
        generator_ckpt,
        use_ema=bool(getattr(config, "use_ema", False)),
    )
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing:
        print(f"[Warning] Missing generator keys: {missing[:8]} ...")
    if unexpected:
        print(f"[Warning] Unexpected generator keys: {unexpected[:8]} ...")

    if generator_train_scope == "linear_only":
        print(f"Loading generator linear checkpoint: {lora_ckpt}")
        load_generator_linear_checkpoint(
            generator.model,
            lora_ckpt,
            expected_sparse_method=sparse_method,
        )
        generator.to(device=device, dtype=dtype)
    else:
        print(f"Applying LoRA config: {config.adapter}")
        generator.model = configure_lora_for_model(
            generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=True,
        )

        print(f"Loading LoRA checkpoint: {lora_ckpt}")
        set_peft_model_state_dict_for_ulysses(
            generator.model,
            load_lora_state_dict(
                lora_ckpt, expected_sparse_method=sparse_method
            ),
        )

        print(f"Merging LoRA on {device} in {dtype}...")
        generator.to(device=device, dtype=dtype)
        generator.model = generator.model.merge_and_unload(safe_merge=True)
        if generator_train_scope == "lora_plus_linear":
            print(f"Loading raw SLA linear checkpoint: {lora_ckpt}")
            load_generator_linear_checkpoint(
                generator.model,
                lora_ckpt,
                expected_sparse_method=sparse_method,
            )
    generator.eval().requires_grad_(False)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "generator": cpu_state_dict(generator),
        "checkpoint_format": "longlive_generator_merged_lora",
        "source_generator_ckpt": str(generator_ckpt),
        "source_lora_ckpt": str(lora_ckpt),
        "model_name": getattr(config.model_kwargs, "model_name", None),
        "dtype": str(dtype).replace("torch.", ""),
        "merged_lora": generator_train_scope in {"lora", "lora_plus_linear"},
        "generator_train_scope": generator_train_scope,
        "sparse_method": sparse_method,
    }
    torch.save(checkpoint, output_path)
    size_gib = os.path.getsize(output_path) / (1024 ** 3)
    print(f"Saved merged generator to {output_path} ({size_gib:.2f} GiB).")
    print("Use this file as checkpoints.generator_ckpt for inference and remove adapter/lora_ckpt from the inference config.")


if __name__ == "__main__":
    main()
