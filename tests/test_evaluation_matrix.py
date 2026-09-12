import os
import subprocess
import sys
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluation" / "run_performance_matrix.sh"
VBENCH_SCRIPT = ROOT / "scripts" / "evaluation" / "run_vbench_matrix.sh"


@pytest.mark.parametrize(
    ("accelerator", "mutation", "can_resume"),
    [
        ("cuda", {}, True),
        ("npu", {}, True),
        ("cuda", {"accelerator": "npu"}, False),
        ("cuda", {"visible_devices": "4,5,6,7"}, False),
        ("cuda", {"sparsity_backend": "cuda_flex"}, False),
        ("npu", {"visible_devices": None}, False),
    ],
)
def test_matrix_resume_validates_manifest_before_skipping(tmp_path, accelerator, mutation, can_resume):
    # 运行真实矩阵与解析器，但把运行目录隔离到临时仓库；匹配结果无需加载任何模型。
    scripts = tmp_path / "scripts" / "evaluation"
    scripts.mkdir(parents=True)
    for name in ("run_performance_matrix.sh", "runtime.sh", "resolve_config.py", "summarize_suite.py", "summarize_vbench.py"):
        shutil.copy2(ROOT / "scripts" / "evaluation" / name, scripts / name)
    (tmp_path / "data").symlink_to(ROOT / "data", target_is_directory=True)
    environment = {
        **os.environ, "LLV2_DEVICE": accelerator, "DRY_RUN": "0", "RESUME_SUITE": "1",
        "SUITE_ID": "contract", "METHODS": "sla_cag", "DURATIONS": "5s", "MODES": "dit_only",
        "LONGLIVE_SP_SIZE": "4", "LONGLIVE_DP_SIZE": "1", "LONGLIVE_SPARSE_METHOD": "sla_cag",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3", "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
        "PERF_DEVICES": "0,1,2,3", "GENERATION_ENV": sys.prefix,
        "CONFIG_PATH": str(ROOT / "configs/inference/msprof.yaml"),
    }
    metadata = json.loads(subprocess.run(
        [sys.executable, str(scripts / "resolve_config.py"), "benchmark",
         "--config", environment["CONFIG_PATH"], "--preset", "5s", "--vae-mode", "dit_only"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=True,
    ).stdout)
    for key, value in mutation.items():
        if value is None:
            metadata.pop(key)
        else:
            metadata[key] = value
    run = tmp_path / "runs/performance/contract-sla_cag-5s-dit_only-sp4"
    run.mkdir(parents=True)
    manifest_text = json.dumps(metadata)
    (run / "manifest.json").write_text(manifest_text)
    (run / "summary.json").write_text(json.dumps({"generation_seconds_mean": 1.0}))
    result = subprocess.run(
        ["bash", str(scripts / "run_performance_matrix.sh")],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert (result.returncode == 0) is can_resume, result.stderr
    assert ("[resume] completed case skipped" in result.stdout) is can_resume
    if not can_resume:
        assert "new SUITE_ID" in result.stderr
    assert (run / "manifest.json").read_text() == manifest_text


def _dry_run(**overrides):
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "SUITE_ID": "contract",
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_complete_performance_matrix_expands_all_72_cases():
    output = _dry_run(PERF_DEVICES="0,1,2,3,4")
    suite_lines = [line for line in output.splitlines() if line.startswith("[suite]")]
    dry_run_lines = [
        line for line in output.splitlines() if line.startswith("[dry-run]")
    ]

    assert len(suite_lines) == 4 * 3 * 3 * 2
    assert len(dry_run_lines) == len(suite_lines)
    assert "method=dense duration=5s mode=dit_only" in suite_lines[0]
    assert any(
        "method=hsa_sla_cag duration=64s mode=async_vae sp=4" in line
        for line in suite_lines
    )
    assert any("devices=0,1,2,3,4" in line for line in dry_run_lines)
    assert all("run_dir=runs/performance/contract-" in line for line in dry_run_lines)


def test_cuda_matrix_defaults_fit_four_devices():
    output = _dry_run(LLV2_DEVICE="cuda", CUDA_VISIBLE_DEVICES="4,5,6,7")
    cases = [line for line in output.splitlines() if line.startswith("[dry-run]")]
    assert len(cases) == 4 * 3 * 2
    assert all("devices=4,5,6,7" in line and "accelerator=cuda" in line for line in cases)
    assert "mode=async_vae" not in output


def test_cuda_matrix_accounts_for_dp_replicas():
    output = _dry_run(
        LLV2_DEVICE="cuda", LONGLIVE_SP_SIZE="2", LONGLIVE_DP_SIZE="2",
        METHODS="dense", DURATIONS="5s", MODES="dit_only", PERF_DEVICES="0,1,2,3",
    )
    assert "devices=0,1,2,3" in output
    assert "-sp2-dp2" in output


def test_cuda_launchers_resolve_without_gpu_cann_or_evaluator(tmp_path):
    for launcher, preset in (("run_benchmark.sh", "5s"), ("run_vbench.sh", "longlive2_standard_5pct")):
        environment = os.environ.copy()
        environment.update({
            "LLV2_DEVICE": "cuda", "DRY_RUN": "1", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "GENERATION_ENV": sys.prefix, "CANN_ENV_SCRIPT": "/missing/cann.sh",
            "AISBENCH_ENV": "/missing/aisbench", "TMPDIR": str(tmp_path),
            "LONGLIVE_SPARSE_METHOD": "sla_cag",
        })
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/evaluation" / launcher), preset],
            cwd=ROOT, env=environment, capture_output=True, text=True, check=True,
        )
        assert "layout=SP4xDP1" in result.stdout
        assert "backend=portable" in result.stdout
    manifests = list(tmp_path.glob("*/manifest.json"))
    assert len(manifests) == 2
    for path in manifests:
        metadata = json.loads(path.read_text())
        assert metadata["accelerator"] == "cuda"
        assert metadata["required_devices"] == 4
        assert path.with_name("resolved.yaml").is_file()


def test_cuda_msprof_is_rejected_before_loading_cann():
    for launcher in ("run_msprof.sh", "run_vae_msprof.sh"):
        environment = {**os.environ, "LLV2_DEVICE": "cuda"}
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/evaluation" / launcher)],
            cwd=ROOT, env=environment, capture_output=True, text=True,
        )
        assert result.returncode == 2
        assert "run_benchmark.sh" in result.stderr


def test_dit_only_matrix_requires_only_four_devices():
    output = _dry_run(
        PERF_DEVICES="8,9,10,11",
        MODES="dit_only",
        LONGLIVE_SP_SIZE="4",
        DURATIONS="32s",
    )
    suite_lines = [line for line in output.splitlines() if line.startswith("[suite]")]

    assert len(suite_lines) == 4
    assert output.count("devices=8,9,10,11") == 4


def test_msprof_matrix_uses_profiled_dit_directory():
    output = _dry_run(
        TASK="msprof",
        PERF_DEVICES="0,1,2,3",
        METHODS="dense",
        MODES="dit_only",
        DURATIONS="5s",
        LONGLIVE_SP_SIZE="4",
    )
    assert "run_dir=runs/msprof/dit/contract-dense-5s-dit_only-sp4" in output


def test_matrix_rejects_invalid_resume_flag():
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "RESUME_SUITE": "sometimes",
            "PERF_DEVICES": "0,1,2,3,4",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "RESUME_SUITE must be 0 or 1" in result.stderr


def test_performance_matrix_selects_method_specific_checkpoints(tmp_path):
    checkpoints = {}
    for method in ("dense", "hsa_cag", "sla_cag", "hsa_sla_cag"):
        path = tmp_path / f"{method}.pt"
        path.touch()
        checkpoints[f"{method.upper()}_GENERATOR_CKPT"] = str(path)

    output = _dry_run(
        PERF_DEVICES="0,1,2,3",
        MODES="dit_only",
        DURATIONS="5s",
        LONGLIVE_SP_SIZE="4",
        **checkpoints,
    )

    for method in ("dense", "hsa_cag", "sla_cag", "hsa_sla_cag"):
        assert f"method={method} duration=5s mode=dit_only sp=4 checkpoint={tmp_path / f'{method}.pt'}" in output


def test_matrix_maps_sp1_and_sp4_async_to_two_and_five_devices():
    output = _dry_run(
        PERF_DEVICES="4,5,6,7,9",
        METHODS="dense",
        DURATIONS="5s",
        MODES="async_vae",
        LONGLIVE_SP_SIZE="1,4",
    )

    assert "devices=4,5 run_id=contract-dense-5s-async_vae-sp1" in output
    assert "devices=4,5,6,7,9 run_id=contract-dense-5s-async_vae-sp4" in output


def test_vbench_matrix_supports_shared_and_method_specific_checkpoints(tmp_path):
    shared = tmp_path / "shared.pt"
    shared.touch()
    hybrid = tmp_path / "hybrid.pt"
    hybrid.touch()
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "METHODS": "dense,hsa_sla_cag",
            "VBENCH_PRESETS": "longlive2_standard_20pct",
            "HSA_SLA_CAG_GENERATOR_CKPT": str(hybrid),
        }
    )

    output = subprocess.run(
        ["bash", str(VBENCH_SCRIPT), str(shared)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert f"checkpoint={shared}" in output
    assert f"checkpoint={hybrid}" in output
