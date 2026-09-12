"""CPU contracts for CUDA launch resolution and mixed-storage FSDP construction."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from functools import partial
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.distributed.fsdp.wrap import _recursive_wrap, size_based_auto_wrap_policy

from utils.config import normalize_config, validate_sparse_training_config, validate_training_resume_contract
from utils.distributed import mixed_storage_wrap_policy, promote_trainable_parameters


ROOT = Path(__file__).resolve().parents[1]


def _launch(tmp_path, entry="run_hsa_sla_cag.sh", device="cuda", overrides=None, previous_config=None, expected_error=None):
    env_root = tmp_path / "env"
    (env_root / "bin").mkdir(parents=True)
    (env_root / "bin/python").symlink_to(sys.executable)
    capture = tmp_path / "torchrun.json"
    torchrun = env_root / "bin/torchrun"
    torchrun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['TRAIN_CAPTURE']).write_text(json.dumps({"
        "'args': sys.argv[1:], 'device': os.environ['LLV2_DEVICE']}))\n"
    )
    torchrun.chmod(0o755)
    model = tmp_path / "model"
    (model / "google/umt5-xxl").mkdir(parents=True)
    for name in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth"):
        (model / name).touch()
    checkpoint = tmp_path / "generator.pt"
    checkpoint.touch()
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("A test prompt\n")
    manifest_dir = tmp_path / "scripts/training"
    manifest_dir.mkdir(parents=True)
    shutil.copy(ROOT / "scripts/training/write_run_manifest.py", manifest_dir)
    # Exercise the NPU launcher's backend preflight without importing hardware.
    module_dir = tmp_path / "wan_5b/modules"
    module_dir.mkdir(parents=True)
    (module_dir.parent / "__init__.py").touch()
    (module_dir / "__init__.py").touch()
    (module_dir / "sla_attention_ascend.py").write_text(
        "def ascend_triton_available(): return True\n"
        "def ascend_triton_unavailable_reason(): return ''\n"
    )
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("LONGLIVE_", "LLV2_", "SPARSE_", "SLA_"))
        and key not in {
            "ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "NPROC_PER_NODE",
            "NNODES", "NODE_RANK", "SP_SIZE", "GRADIENT_ACCUMULATION_STEPS",
            "MODEL_LOAD_DTYPE", "GENERATOR_TRAIN_SCOPE", "GENERATOR_LR", "LINEAR_LR",
        }
    }
    method = "hsa_cag" if entry == "run_hsa_cag.sh" else "sla_cag" if entry == "run_sla_cag.sh" else "hsa_sla_cag"
    env.update({
        "LONGLIVE_ROOT": str(tmp_path), "GENERATION_ENV": str(env_root),
        "MODEL_ROOT": str(model), "GENERATOR_CKPT": str(checkpoint),
        "TRAIN_PROMPTS": str(prompts), "TRAIN_RUN_NAME": "test",
        "TRAIN_CAPTURE": str(capture), "VALIDATE_LINEAR_CHECKPOINT": "0",
        "DRY_RUN": "0",
        "CONFIG_PATH": str(ROOT / f"configs/train/{method}.yaml"),
        "MAX_ITERS": "1", "VIS_INTERVAL": "0",
    })
    if device is not None:
        env["LLV2_DEVICE"] = device
    env.update(overrides or {})
    run = tmp_path / "runs/training/test"
    if previous_config is not None:
        checkpoint_dir = run / "checkpoints/step_0000001"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "train_state.pt").touch()
        OmegaConf.save(previous_config, run / "config.resolved.yaml")
        (run / "config.source.yaml").write_text("original source\n")
        previous_bytes = (run / "config.resolved.yaml").read_bytes()
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/training" / entry)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    if expected_error:
        assert result.returncode != 0
        assert expected_error in result.stdout + result.stderr
        assert (run / "config.resolved.yaml").read_bytes() == previous_bytes
        assert (run / "config.source.yaml").read_text() == "original source\n"
        assert not capture.exists()
        return None
    assert result.returncode == 0, result.stdout + result.stderr
    return (
        OmegaConf.load(run / "config.resolved.yaml"),
        json.loads(capture.read_text()), json.loads((run / "manifest.json").read_text()),
    )


@pytest.mark.parametrize("entry", [
    "run_hsa_cag.sh", "run_sla_cag.sh", "run_hsa_sla_cag.sh",
    "run_hsa_sla_cag_linear_only.sh",
])
def test_cuda_launch_defaults_keep_effective_batch_and_training_semantics(tmp_path, entry):
    config, invocation, manifest = _launch(tmp_path, entry=entry)
    assert config.infra.device_type == "cuda"
    assert config.infra.distributed_backend == "nccl"
    assert config.infra.sequence_parallel_size == 4
    assert config.training.gradient_accumulation_steps == 8
    assert config.infra.model_load_dtype == "bfloat16"
    assert config.infra.trainable_parameter_dtype == "float32"
    assert config.model_kwargs.sparse_config.backend == "portable"
    assert list(config.data.image_or_video_shape) == [1, 32, 48, 44, 80]
    assert config.model_kwargs.local_attn_size == 32
    assert config.model_kwargs.sparse_config.sparsity == 0.85
    assert "--standalone" in invocation["args"]
    assert "--nproc_per_node=4" in invocation["args"]
    assert manifest["distributed"]["effective_batch_size"] == 8
    assert manifest["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    expected_scope = "linear_only" if "linear_only" in entry else "lora" if entry == "run_hsa_cag.sh" else "lora_plus_linear"
    assert config.training.generator_train_scope == expected_scope


def test_unset_device_keeps_npu_training_defaults(tmp_path):
    config, invocation, manifest = _launch(tmp_path, device=None)
    assert config.infra.device_type == "npu"
    assert config.infra.distributed_backend == "hccl"
    assert config.infra.model_load_dtype == "float32"
    assert config.model_kwargs.sparse_config.backend == "ascend_triton"
    assert config.infra.sequence_parallel_size == 8
    assert config.training.gradient_accumulation_steps == 4
    assert "--nproc_per_node=16" in invocation["args"]
    assert manifest["distributed"]["effective_batch_size"] == 8


@pytest.mark.parametrize("sp,accumulation,effective_batch", [(4, 1, 1), (2, 4, 8)])
def test_cuda_overrides_support_smoke_and_sp2_dp2(tmp_path, sp, accumulation, effective_batch):
    config, _, manifest = _launch(tmp_path, overrides={
        "LONGLIVE_SP_SIZE": str(sp), "GRADIENT_ACCUMULATION_STEPS": str(accumulation),
        "SPARSE_BACKEND": "cuda_flex", "MODEL_LOAD_DTYPE": "float32",
    })
    config.data.eval_data_path = config.data.data_path
    validate_sparse_training_config(normalize_config(config))
    assert config.model_kwargs.sparse_config.backend == "cuda_flex"
    assert manifest["distributed"]["effective_batch_size"] == effective_batch


def test_training_config_rejects_backend_for_wrong_hardware(tmp_path):
    config, _, _ = _launch(tmp_path)
    config.data.eval_data_path = config.data.data_path
    config.model_kwargs.sparse_config.backend = "ascend_triton"
    with pytest.raises(ValueError, match="CUDA training cannot use ascend_triton"):
        validate_sparse_training_config(normalize_config(config))


def test_cuda_dry_run_needs_no_models_and_refuses_overwriting(tmp_path):
    env = dict(os.environ)
    env.update({
        "LLV2_DEVICE": "cuda", "DRY_RUN": "1", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "LONGLIVE_ROOT": str(tmp_path), "GENERATION_ENV": sys.prefix,
        "CONFIG_PATH": str(ROOT / "configs/train/hsa_sla_cag.yaml"),
        "MODEL_ROOT": str(tmp_path / "missing-model"),
        "GENERATOR_CKPT": str(tmp_path / "missing-checkpoint"),
        "TRAIN_PROMPTS": str(tmp_path / "missing-prompts"), "TRAIN_RUN_NAME": "dry",
        "NPROC_PER_NODE": "4", "NNODES": "1", "NODE_RANK": "0",
        "LONGLIVE_SP_SIZE": "4", "GRADIENT_ACCUMULATION_STEPS": "8",
    })
    command = ["bash", str(ROOT / "scripts/training/run_hsa_sla_cag.sh")]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "torchrun not started" in result.stdout
    config_path = tmp_path / "runs/training/dry/config.resolved.yaml"
    before = config_path.read_bytes()
    config = OmegaConf.load(config_path)
    assert config.infra.device_type == "cuda"
    assert config.infra.model_load_dtype == "bfloat16"
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "refuses to overwrite" in result.stderr
    assert config_path.read_bytes() == before


@pytest.mark.parametrize("field,value", [
    ("infra.device_type", "cuda"), ("infra.model_load_dtype", "bfloat16"),
    ("model_kwargs.sparse_config.method", "sla_cag"),
    ("model_kwargs.sparse_config.backend", "portable"),
    ("infra.sequence_parallel_size", 4), ("training.generator_train_scope", "linear_only"),
    ("model_kwargs.local_attn_size", 16), ("data.image_or_video_shape", [1, 16, 48, 44, 80]),
    ("checkpoints.generator_ckpt", "/different/base.pt"),
])
def test_resume_rejects_changed_training_contract(field, value):
    previous = OmegaConf.load(ROOT / "configs/train/hsa_sla_cag.yaml")
    current = OmegaConf.create(OmegaConf.to_container(previous))
    OmegaConf.update(current, field, value)
    with pytest.raises(ValueError, match="different run contract"):
        validate_training_resume_contract(previous, current)


def test_resume_accepts_legacy_npu_defaults_and_new_training_intervals():
    previous = OmegaConf.load(ROOT / "configs/train/hsa_sla_cag.yaml")
    current = OmegaConf.create(OmegaConf.to_container(previous))
    current.infra.device_type = "npu"
    current.infra.model_load_dtype = "float32"
    current.training.max_iters = 400
    current.training.log_iters = 50
    current.evaluation.interval = 200
    validate_training_resume_contract(previous, current)


def test_incompatible_resume_preserves_resolved_and_source_files(tmp_path):
    previous = OmegaConf.load(ROOT / "configs/train/hsa_sla_cag.yaml")
    _launch(tmp_path, previous_config=previous, expected_error="different run contract")


def test_formal_launch_rejects_existing_directory_without_checkpoint(tmp_path):
    run = tmp_path / "runs/training/incomplete"
    run.mkdir(parents=True)
    evidence = run / "config.resolved.yaml"
    evidence.write_text("previous evidence\n")
    env = dict(os.environ)
    env.update({
        "LLV2_DEVICE": "cuda", "DRY_RUN": "0", "NNODES": "1", "NODE_RANK": "0",
        "LONGLIVE_ROOT": str(tmp_path), "TRAIN_RUN_NAME": "incomplete",
    })
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/training/run_hsa_sla_cag.sh")],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "existing training directory has no train_state.pt" in result.stderr
    assert evidence.read_text() == "previous evidence\n"


class _StorageCheckedWrapper(torch.nn.Module):
    """Stand-in for FSDP's homogeneous dtype flattening, without a process group."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.owned_parameters = []

        def collect(child):
            if isinstance(child, _StorageCheckedWrapper):
                return
            self.owned_parameters.extend(child.parameters(recurse=False))
            for nested in child.children():
                collect(nested)

        collect(module)
        assert len({p.dtype for p in self.owned_parameters}) <= 1


def test_mixed_storage_fsdp_isolates_lora_and_raw_linear_without_changing_scope():
    module = torch.nn.Module()
    module.base_layer = torch.nn.Linear(16, 16, dtype=torch.bfloat16).requires_grad_(False)
    module.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(16, 2, bias=False, dtype=torch.bfloat16)})
    module.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 16, bias=False, dtype=torch.bfloat16)})
    module.sla_linear = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
    original_scope = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    frozen_weights = module.base_layer.weight.detach().clone()
    promote_trainable_parameters(module)
    assert {name for name, p in module.named_parameters() if p.requires_grad} == original_scope
    torch.testing.assert_close(module.base_layer.weight, frozen_weights)
    fp32_modules = {module.lora_A["default"], module.lora_B["default"], module.sla_linear}
    policy = partial(
        mixed_storage_wrap_policy,
        base_policy=partial(size_based_auto_wrap_policy, min_num_params=50_000_000),
        fp32_modules=fp32_modules,
    )
    wrapped, _ = _recursive_wrap(module, policy, _StorageCheckedWrapper, set(), set())
    root = _StorageCheckedWrapper(wrapped)
    assert root.owned_parameters
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in root.owned_parameters)
    for name in ("lora_A", "lora_B"):
        child = getattr(module, name)["default"]
        assert isinstance(child, _StorageCheckedWrapper)
        assert all(p.dtype == torch.float32 and p.requires_grad for p in child.owned_parameters)
    assert isinstance(module.sla_linear, _StorageCheckedWrapper)


def test_fsdp_uses_separate_fp32_leaves_and_casts_adapter_inputs():
    from utils.distributed import fsdp_wrap

    module = torch.nn.Module()
    module.base = torch.nn.Linear(4, 4, dtype=torch.bfloat16).requires_grad_(False)
    module.sla_linear = torch.nn.Linear(4, 4, dtype=torch.float32)
    with patch("utils.distributed.FSDP", return_value=module) as fsdp, \
         patch("utils.distributed.current_device", return_value=0), \
         patch("utils.distributed.distributed_backend", return_value="nccl"):
        fsdp_wrap(module, mixed_precision=True, separate_trainable_parameters=True)
    kwargs = fsdp.call_args.kwargs
    assert kwargs["mixed_precision"].cast_forward_inputs
    policy = kwargs["auto_wrap_policy"]
    assert policy(module=module.sla_linear, recurse=False, nonwrapped_numel=20)
    assert not policy(module=module.base, recurse=False, nonwrapped_numel=20)


def test_raw_linear_resume_preserves_sub_bfloat16_updates():
    from utils.inference_utils import load_generator_linear_state_dict

    module = torch.nn.Module()
    module.sla_linear = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
    promote_trainable_parameters(module)
    state = {
        "sla_linear.weight": torch.full((4, 4), 1.000002, dtype=torch.float32),
        "sla_linear.bias": torch.full((4,), 0.500002, dtype=torch.float32),
    }
    load_generator_linear_state_dict(module, state)
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("load_dtype,expected", [(None, None), ("float32", torch.float32), ("bfloat16", torch.bfloat16)])
def test_wrapper_passes_storage_dtype_to_checkpoint_loader_and_new_linear(load_dtype, expected):
    from utils.wan_5b_wrapper import WanDiffusionWrapper

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(4, 4)

        def initialize_sla_linear(self):
            self.sla_linear = torch.nn.Linear(4, 4)

        def register_to_config(self, **kwargs):
            pass

    tiny = TinyModel()
    with patch("utils.wan_5b_wrapper.CausalWanModel.from_pretrained", return_value=tiny) as loader:
        wrapper = WanDiffusionWrapper(is_causal=True, load_dtype=load_dtype)
    if expected is None:
        assert "torch_dtype" not in loader.call_args.kwargs
    else:
        assert loader.call_args.kwargs["torch_dtype"] == expected
    assert all(p.dtype == (expected or torch.float32) for p in wrapper.parameters())
