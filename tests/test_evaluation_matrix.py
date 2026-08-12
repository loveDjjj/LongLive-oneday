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


def test_complete_performance_matrix_expands_all_36_cases():
    output = _dry_run(PERF_DEVICES="0,1,2,3,4")
    suite_lines = [line for line in output.splitlines() if line.startswith("[suite]")]
    dry_run_lines = [
        line for line in output.splitlines() if line.startswith("[dry-run]")
    ]

    assert len(suite_lines) == 4 * 3 * 3
    assert len(dry_run_lines) == len(suite_lines)
    assert "method=dense duration=5s mode=dit_only" in suite_lines[0]
    assert any(
        "method=hsa_sla_cag duration=64s mode=async_vae" in line
        for line in suite_lines
    )
    assert any("devices=0,1,2,3,4" in line for line in dry_run_lines)


def test_dit_only_matrix_requires_only_four_devices():
    output = _dry_run(
        PERF_DEVICES="8,9,10,11",
        MODES="dit_only",
        DURATIONS="32s",
    )
    suite_lines = [line for line in output.splitlines() if line.startswith("[suite]")]

    assert len(suite_lines) == 4
    assert output.count("devices=8,9,10,11") == 4


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
        **checkpoints,
    )

    for method in ("dense", "hsa_cag", "sla_cag", "hsa_sla_cag"):
        assert f"method={method} duration=5s mode=dit_only checkpoint={tmp_path / f'{method}.pt'}" in output


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
