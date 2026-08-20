import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluation" / "run_performance_matrix.sh"
VBENCH_SCRIPT = ROOT / "scripts" / "evaluation" / "run_vbench_matrix.sh"


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
