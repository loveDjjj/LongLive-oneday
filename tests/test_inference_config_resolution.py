from types import SimpleNamespace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.evaluation.resolve_config import (
    resolve_benchmark,
    resolve_msprof,
    resolve_vbench,
)


ROOT = Path(__file__).resolve().parents[1]
VBENCH_CONFIG = ROOT / "configs" / "inference" / "vbench.yaml"
MSPROF_CONFIG = ROOT / "configs" / "inference" / "msprof.yaml"


def _args(config, preset, output, *, seed=None):
    return SimpleNamespace(
        config=config,
        preset=preset,
        output=output,
        output_folder=None,
        seed=seed,
        num_prompts=None,
        warmup_per_rank=None,
        save_latents_only=False,
    )


def test_vbench_dense_does_not_inject_sparse_model_config(tmp_path, monkeypatch):
    monkeypatch.delenv("LONGLIVE_SPARSE_METHOD", raising=False)
    output = tmp_path / "dense.yaml"

    metadata = resolve_vbench(
        _args(VBENCH_CONFIG, "longlive2_standard_5pct", output, seed=0)
    )
    resolved = OmegaConf.load(output)

    assert "sparse_config" not in resolved.model_kwargs
    assert metadata["sparsity_method"] == "dense"
    assert metadata["dp_size"] == 8
    assert metadata["nproc_per_node"] == 16
    assert metadata["required_devices"] == 16


def test_vbench_hsa_uses_required_fused_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")
    output = tmp_path / "hsa.yaml"

    metadata = resolve_vbench(
        _args(VBENCH_CONFIG, "longlive2_standard_5pct", output, seed=0)
    )
    sparse = OmegaConf.load(output).model_kwargs.sparse_config

    assert sparse.enabled is True
    assert sparse.backend == "mindiesd"
    assert sparse.block_q == 128
    assert sparse.block_k == 128
    assert metadata["sparsity_method"] == "hsa_cag"


def test_explicit_dense_override_disables_configured_sparsity(tmp_path, monkeypatch):
    config = OmegaConf.load(VBENCH_CONFIG)
    config.sparsity.enabled = True
    config.sparsity.method = "hsa_cag"
    config_path = tmp_path / "sparse_default.yaml"
    OmegaConf.save(config, config_path)
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "dense")

    metadata = resolve_vbench(
        _args(config_path, "longlive2_standard_5pct", tmp_path / "dense.yaml", seed=0)
    )

    resolved = OmegaConf.load(tmp_path / "dense.yaml")
    assert "sparse_config" not in resolved.model_kwargs
    assert metadata["sparsity_method"] == "dense"


def test_vbench_rejects_hsa_for_native_wan22(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")

    with pytest.raises(ValueError, match="LongLive2 causal inference only"):
        resolve_vbench(
            _args(VBENCH_CONFIG, "wan22_standard_5pct", tmp_path / "wan22.yaml", seed=0)
        )


def test_msprof_hsa_uses_required_fused_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")
    output = tmp_path / "msprof.yaml"

    metadata = resolve_msprof(_args(MSPROF_CONFIG, "32s", output))
    sparse = OmegaConf.load(output).model_kwargs.sparse_config

    assert sparse.enabled is True
    assert sparse.backend == "mindiesd"
    assert sparse.block_q == 128
    assert sparse.block_k == 128
    assert metadata["sparsity_method"] == "hsa_cag"
    assert metadata["sparsity_backend"] == "mindiesd"
    assert metadata["msprof"]["ai_core"] is True


def test_benchmark_overrides_repetition_counts_without_profiler_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")
    args = _args(MSPROF_CONFIG, "32s", tmp_path / "benchmark.yaml")
    args.num_prompts = 4
    args.warmup_per_rank = 1

    metadata = resolve_benchmark(args)

    assert metadata["task"] == "benchmark"
    assert metadata["num_prompts"] == 4
    assert metadata["warmup_per_rank"] == 1
    assert metadata["run_tag"].startswith("benchmark-")
    assert "msprof" not in metadata
    assert OmegaConf.load(args.output).inference_iter == 3


def test_latent_only_benchmark_disables_dedicated_vae(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")
    args = _args(MSPROF_CONFIG, "32s", tmp_path / "latent_only.yaml")
    args.num_prompts = 4
    args.warmup_per_rank = 1
    args.save_latents_only = True

    metadata = resolve_benchmark(args)
    resolved = OmegaConf.load(args.output)

    assert resolved.save_latents_only is True
    assert resolved.inference.streaming_vae is False
    assert resolved.inference.async_vae is False
    assert resolved.inference.vae_device is None
    assert metadata["vae_mode"] == "disabled"
    assert metadata["required_devices"] == 4
