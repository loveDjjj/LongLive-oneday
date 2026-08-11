from types import SimpleNamespace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.evaluation.resolve_config import resolve_msprof, resolve_vbench


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

    with pytest.raises(ValueError, match="unsupported SLA inference backend"):
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
