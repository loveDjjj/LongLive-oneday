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


def test_vbench_hsa_uses_required_ascend_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")
    output = tmp_path / "hsa.yaml"

    metadata = resolve_vbench(
        _args(VBENCH_CONFIG, "longlive2_standard_5pct", output, seed=0)
    )
    sparse = OmegaConf.load(output).model_kwargs.sparse_config

    assert sparse.enabled is True
    assert sparse.backend == "ascend_triton"
    assert metadata["sparsity_method"] == "hsa_cag"


def test_vbench_rejects_hsa_for_native_wan22(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")

    with pytest.raises(ValueError, match="LongLive2 causal inference only"):
        resolve_vbench(
            _args(VBENCH_CONFIG, "wan22_standard_5pct", tmp_path / "wan22.yaml", seed=0)
        )


def test_msprof_hsa_uses_required_ascend_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_SPARSE_METHOD", "hsa_cag")
    output = tmp_path / "msprof.yaml"

    metadata = resolve_msprof(_args(MSPROF_CONFIG, "32s", output))
    sparse = OmegaConf.load(output).model_kwargs.sparse_config

    assert sparse.enabled is True
    assert sparse.backend == "ascend_triton"
    assert metadata["sparsity_method"] == "hsa_cag"
