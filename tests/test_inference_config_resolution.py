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


def test_manifest_records_generator_checkpoint_override(tmp_path, monkeypatch):
    checkpoint = tmp_path / "generator.pt"
    checkpoint.touch()
    monkeypatch.setenv("LONGLIVE_GENERATOR_CKPT", str(checkpoint))

    benchmark = resolve_benchmark(
        _args(MSPROF_CONFIG, "5s", tmp_path / "benchmark.yaml")
    )
    vbench = resolve_vbench(
        _args(VBENCH_CONFIG, "longlive2_standard_5pct", tmp_path / "vbench.yaml", seed=0)
    )

    assert benchmark["generator_checkpoint"] == str(checkpoint)
    assert vbench["generator_checkpoint"] == str(checkpoint)


@pytest.mark.parametrize(
    ("sp_size", "mode", "required_devices"),
    [(1, "sync_vae", 1), (1, "async_vae", 2), (4, "sync_vae", 4), (4, "async_vae", 5)],
)
def test_benchmark_sp_override_controls_worker_and_vae_devices(
    tmp_path, monkeypatch, sp_size, mode, required_devices
):
    monkeypatch.setenv("LONGLIVE_SP_SIZE", str(sp_size))
    args = _args(MSPROF_CONFIG, "5s", tmp_path / f"sp{sp_size}-{mode}.yaml")
    args.vae_mode = mode

    metadata = resolve_benchmark(args)
    resolved = OmegaConf.load(args.output)

    assert metadata["sp_size"] == sp_size
    assert metadata["nproc_per_node"] == sp_size
    assert metadata["required_devices"] == required_devices
    assert resolved.sp_size == sp_size
    assert resolved.inference.vae_device == (
        f"npu:{sp_size}" if mode == "async_vae" else None
    )


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
    assert sparse.sparsity == 0.85
    assert sparse.sparsity_base == 0.95
    assert sparse.first_chunk_dense is True
    assert sparse.budget_reference == "full_resident_kv"
    assert sparse.full_kv_linear_compensation is True
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
    assert sparse.keep_near_history_frames == 4
    assert sparse.keep_dynamic_history_frames == 4
    assert sparse.protect_current_frames is True
    assert sparse.protect_longlive_sink_frames is True
    assert sparse.dense_current_blocks is False
    assert sparse.hsa_history_mode == "rolling"
    assert sparse.first_chunk_dense is True
    assert sparse.budget_reference == "full_resident_kv"
    assert "full_kv_linear_compensation" not in sparse
    assert "feature_map" not in sparse
    assert metadata["sparsity_method"] == "hsa_cag"


def test_vbench_hsa_history_mode_override_selects_full(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")
    monkeypatch.setenv("LONGLIVE_HSA_HISTORY_MODE", "full")
    output = tmp_path / "hsa_full.yaml"

    resolve_vbench(
        _args(VBENCH_CONFIG, "longlive2_standard_5pct", output, seed=0)
    )

    sparse = OmegaConf.load(output).model_kwargs.sparse_config
    assert sparse.method == "hsa_cag"
    assert sparse.hsa_history_mode == "full"


def test_vbench_rejects_hsa_history_mode_for_non_hsa(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "sla_cag")
    monkeypatch.setenv("LONGLIVE_HSA_HISTORY_MODE", "full")

    with pytest.raises(ValueError, match="only applies"):
        resolve_vbench(
            _args(VBENCH_CONFIG, "longlive2_standard_5pct", tmp_path / "bad.yaml", seed=0)
        )


def test_vbench_hybrid_selects_frame_filtered_sla_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_sla_cag")
    output = tmp_path / "hybrid.yaml"

    metadata = resolve_vbench(
        _args(VBENCH_CONFIG, "longlive2_standard_5pct", output, seed=0)
    )
    sparse = OmegaConf.load(output).model_kwargs.sparse_config

    assert sparse.method == "hsa_sla_cag"
    assert sparse.sparsity == 0.85
    assert sparse.sparsity_base == 0.95
    assert sparse.keep_near_history_frames == 4
    assert sparse.keep_dynamic_history_frames == 4
    assert sparse.max_global_sink_frames == 2
    assert sparse.max_shot_sink_frames == 2
    assert "hard_keep_sink_frames" not in sparse
    assert "hard_keep_recent_frames" not in sparse
    assert sparse.first_chunk_dense is True
    assert sparse.budget_reference == "full_resident_kv"
    assert sparse.full_kv_linear_compensation is True
    assert sparse.linear_cache is True
    assert "candidate_frames" not in sparse
    assert metadata["sparsity_method"] == "hsa_sla_cag"


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


@pytest.mark.parametrize("resolver", [resolve_benchmark, resolve_msprof])
@pytest.mark.parametrize(
    ("duration", "latent_frames"),
    [("5s", 32), ("32s", 192), ("64s", 384)],
)
@pytest.mark.parametrize(
    ("mode", "required_devices"),
    [("dit_only", 4), ("sync_vae", 4), ("async_vae", 5)],
)
@pytest.mark.parametrize(
    "method", ["dense", "hsa_cag", "sla_cag", "hsa_sla_cag"]
)
def test_all_performance_matrix_cases_resolve(
    tmp_path,
    monkeypatch,
    resolver,
    duration,
    latent_frames,
    mode,
    required_devices,
    method,
):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", method)
    output = tmp_path / (
        f"{resolver.__name__}-{method}-{duration}-{mode}.yaml"
    )
    args = _args(MSPROF_CONFIG, duration, output)
    args.vae_mode = mode

    metadata = resolver(args)
    resolved = OmegaConf.load(output)

    assert metadata["latent_frames"] == latent_frames
    assert metadata["required_devices"] == required_devices
    assert metadata["vae_mode"] == mode
    assert metadata["sparsity_method"] == method
    assert resolved.num_output_frames == latent_frames
    if method == "dense":
        assert "sparse_config" not in resolved.model_kwargs
    else:
        assert resolved.model_kwargs.sparse_config.method == method
