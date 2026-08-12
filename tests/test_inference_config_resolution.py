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
        vae_mode=None,
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


def test_explicit_dense_override_disables_configured_sparsity(tmp_path, monkeypatch):
    config = OmegaConf.load(VBENCH_CONFIG)
    config.sparsity.enabled = True
    config.sparsity.method = "sla_cag"
    config_path = tmp_path / "sparse_default.yaml"
    OmegaConf.save(config, config_path)
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "dense")
    output = tmp_path / "explicit_dense.yaml"

    metadata = resolve_vbench(
        _args(config_path, "longlive2_standard_5pct", output, seed=0)
    )

    resolved = OmegaConf.load(output)
    assert "sparse_config" not in resolved.model_kwargs
    assert resolved.sparsity.enabled is False
    assert resolved.sparsity.method == "dense"
    assert metadata["sparsity_method"] == "dense"
    assert metadata["sparsity_backend"] == "dense"


def test_vbench_sla_uses_required_fused_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "sla_cag")
    monkeypatch.delenv("LONGLIVE_SLA_BACKEND", raising=False)
    output = tmp_path / "sla.yaml"

    metadata = resolve_vbench(
        _args(VBENCH_CONFIG, "longlive2_standard_5pct", output, seed=0)
    )
    sparse = OmegaConf.load(output).model_kwargs.sparse_config

    assert sparse.enabled is True
    assert sparse.backend == "mindiesd"
    assert sparse.block_q == 128
    assert sparse.block_k == 128
    assert sparse.feature_map == "softmax"
    assert sparse.sparsity == 0.95
    assert metadata["sparsity_method"] == "sla_cag"
    assert metadata["sparsity_backend"] == "mindiesd"


def test_vbench_hsa_selects_hsa_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")
    monkeypatch.delenv("LONGLIVE_SPARSE_BACKEND", raising=False)
    output = tmp_path / "hsa.yaml"

    metadata = resolve_vbench(
        _args(VBENCH_CONFIG, "longlive2_standard_5pct", output, seed=0)
    )
    sparse = OmegaConf.load(output).model_kwargs.sparse_config

    assert sparse.enabled is True
    assert sparse.method == "hsa_cag"
    assert sparse.backend == "mindiesd"
    assert sparse.keep_frames == 6
    assert sparse.keep_sink == 1
    assert sparse.keep_near == 2
    assert "feature_map" not in sparse
    assert metadata["sparsity_method"] == "hsa_cag"


def test_vbench_rejects_bsa_for_hsa(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")
    monkeypatch.setenv("LONGLIVE_SPARSE_BACKEND", "mindiesd_bsa")

    with pytest.raises(ValueError, match="unsupported sparse inference backend"):
        resolve_vbench(
            _args(VBENCH_CONFIG, "longlive2_standard_5pct", tmp_path / "bad_hsa.yaml", seed=0)
        )


@pytest.mark.parametrize(
    ("mode", "expected_mode", "required_devices", "save_latents", "async_vae"),
    [
        ("dit_only", "dit_only", 4, True, False),
        ("sync_vae", "sync_vae", 4, False, False),
        ("async_vae", "async_vae", 5, False, True),
    ],
)
def test_benchmark_vae_modes(
    tmp_path, monkeypatch, mode, expected_mode, required_devices, save_latents, async_vae
):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")
    args = _args(MSPROF_CONFIG, "5s", tmp_path / f"{mode}.yaml")
    args.num_prompts = 4
    args.warmup_per_rank = 1
    args.vae_mode = mode

    metadata = resolve_benchmark(args)
    resolved = OmegaConf.load(args.output)

    assert metadata["latent_frames"] == 32
    assert metadata["pixel_frames"] == 125
    assert metadata["vae_mode"] == expected_mode
    assert metadata["required_devices"] == required_devices
    assert resolved.save_latents_only is save_latents
    assert resolved.inference.async_vae is async_vae


def test_vbench_can_select_mindiesd_bsa(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "sla_cag")
    monkeypatch.setenv("LONGLIVE_SLA_BACKEND", "mindiesd_bsa")
    output = tmp_path / "sla_bsa.yaml"

    metadata = resolve_vbench(
        _args(VBENCH_CONFIG, "longlive2_standard_5pct", output, seed=0)
    )

    assert OmegaConf.load(output).model_kwargs.sparse_config.backend == "mindiesd_bsa"
    assert OmegaConf.load(output).sparsity.options.backend == "mindiesd_bsa"
    assert OmegaConf.load(output).sparsity.method == "sla_cag"
    assert OmegaConf.load(output).sparsity.enabled is True
    assert metadata["sparsity_backend"] == "mindiesd_bsa"
    assert metadata["run_tag"].endswith("sla_cag-mindiesd_bsa")


def test_vbench_rejects_unknown_sla_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "sla_cag")
    monkeypatch.setenv("LONGLIVE_SLA_BACKEND", "unknown")

    with pytest.raises(ValueError, match="unsupported sparse inference backend"):
        resolve_vbench(
            _args(VBENCH_CONFIG, "longlive2_standard_5pct", tmp_path / "bad.yaml", seed=0)
        )


def test_vbench_rejects_sla_for_native_wan22(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "sla_cag")

    with pytest.raises(ValueError, match="LongLive2 causal inference only"):
        resolve_vbench(
            _args(VBENCH_CONFIG, "wan22_standard_5pct", tmp_path / "wan22.yaml", seed=0)
        )


def test_msprof_sla_uses_required_fused_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "sla_cag")
    monkeypatch.delenv("LONGLIVE_SLA_BACKEND", raising=False)
    output = tmp_path / "msprof.yaml"

    metadata = resolve_msprof(_args(MSPROF_CONFIG, "32s", output))
    sparse = OmegaConf.load(output).model_kwargs.sparse_config

    assert sparse.enabled is True
    assert sparse.backend == "mindiesd"
    assert sparse.block_q == 128
    assert sparse.block_k == 128
    assert sparse.linear_cache is True
    assert metadata["sparsity_method"] == "sla_cag"
    assert metadata["sparsity_backend"] == "mindiesd"
    assert metadata["msprof"]["ai_core"] is True


def test_benchmark_overrides_repetition_counts_without_profiler_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "sla_cag")
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
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "sla_cag")
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
    assert metadata["vae_mode"] == "dit_only"
    assert metadata["required_devices"] == 4
    assert "-dit_only-" in metadata["run_tag"]
