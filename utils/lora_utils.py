# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig
)


def set_peft_model_state_dict_for_ulysses(lora_model, lora_state_dict):
    """Load PEFT weights without treating Ulysses metadata as HF tensor parallelism.

    PEFT 0.19 enters its Hugging Face TP sharding path whenever distributed is
    initialized and LoRA base layers expose ``_hf_tp_plan`` and
    ``_hf_device_mesh``. LongLive uses Ulysses SP instead of HF TP, so those
    attributes must not affect adapter loading. Preserve and restore them to
    avoid changing the model after this compatibility boundary.
    """
    import peft

    metadata = []
    for module in lora_model.modules():
        saved = {}
        for attribute in ("_hf_tp_plan", "_hf_device_mesh"):
            if hasattr(module, attribute):
                saved[attribute] = getattr(module, attribute)
                setattr(module, attribute, None)
        if saved:
            metadata.append((module, saved))

    try:
        return peft.set_peft_model_state_dict(lora_model, lora_state_dict)
    finally:
        for module, saved in metadata:
            for attribute, value in saved.items():
                setattr(module, attribute, value)


def lora_target_linear_modules(transformer, adapter_target_modules):
    """返回 LoRA 目标层，原始 SLA 补偿层始终由独立优化器参数组训练。"""
    target_linear_modules = set()
    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if (
                    isinstance(submodule, torch.nn.Linear)
                    and not full_submodule_name.endswith(".sla_linear")
                ):
                    target_linear_modules.add(full_submodule_name)
    return sorted(target_linear_modules)


def configure_lora_for_model(
    transformer,
    model_name,
    lora_config,
    is_main_process=True,
    all_causal=False,
):
    """Configure LoRA for a WanDiffusionWrapper model
    
    Args:
        transformer: The transformer model to apply LoRA to
        model_name: 'generator' or 'fake_score'
        lora_config: LoRA configuration
        is_main_process: Whether this is the main process (for logging)
        all_causal: Whether all models use causal attention blocks
    
    Returns:
        lora_model: The LoRA-wrapped model
    """
    import peft

    if model_name == 'generator':
        adapter_target_modules = ['CausalWanAttentionBlock']
    elif model_name == 'fake_score':
        adapter_target_modules = ['CausalWanAttentionBlock'] if all_causal else ['WanAttentionBlock']
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    target_linear_modules = lora_target_linear_modules(
        transformer, adapter_target_modules
    )
    
    if is_main_process:
        print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
        if getattr(lora_config, 'verbose', False):
            for module_name in sorted(target_linear_modules):
                print(f"  - {module_name}")
    
    # Create LoRA config
    adapter_type = lora_config.get('type', 'lora')
    if adapter_type == 'lora':
        peft_config = peft.LoraConfig(
            r=lora_config.get('rank', 16),
            lora_alpha=lora_config.get('alpha', None) or lora_config.get('rank', 16),
            lora_dropout=lora_config.get('dropout', 0.0),
            target_modules=target_linear_modules,
        )
    else:
        raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
    
    # Apply LoRA to the transformer
    lora_model = peft.get_peft_model(transformer, peft_config)

    if is_main_process:
        print('peft_config', peft_config)
        lora_model.print_trainable_parameters()
    
    return lora_model


def gather_lora_state_dict(lora_model):
    from peft import get_peft_model_state_dict

    with FSDP.state_dict_type(
        lora_model,                  
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
    ):
        full = lora_model.state_dict()
    return get_peft_model_state_dict(lora_model, state_dict=full)


def load_lora_checkpoint(lora_model, lora_state_dict, model_name, is_main_process=True):
    """Load LoRA weights from state dict
    
    Args:
        lora_model: The LoRA-wrapped model
        lora_state_dict: LoRA state dict to load
        model_name: 'generator' or 'critic'
        is_main_process: Whether this is the main process (for logging)
    """
    if is_main_process:
        print(f"Loading LoRA {model_name} weights: {len(lora_state_dict)} keys in checkpoint")
    
    set_peft_model_state_dict_for_ulysses(lora_model, lora_state_dict)
    
    if is_main_process:
        print(f"LoRA {model_name} weights loaded successfully")
