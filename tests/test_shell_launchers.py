from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_LAUNCHERS = (
    "scripts/evaluation/run_benchmark.sh",
    "scripts/evaluation/run_msprof.sh",
    "scripts/evaluation/run_vbench.sh",
)
ALL_SHELL_SCRIPTS = tuple(ROOT.glob("**/*.sh"))


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_single_node_evaluation_launchers_use_standalone_rendezvous():
    for relative_path in EVALUATION_LAUNCHERS:
        script = _read(relative_path)
        assert "--standalone" in script, relative_path
        assert 'export MASTER_PORT=' not in script, relative_path
        assert '--master_port=' not in script, relative_path


def test_training_launcher_allocates_multinode_port_dynamically():
    script = _read("scripts/training/run_sparse_cag.sh")
    assert 'sock.bind((host, 0))' in script
    assert 'RENDEZVOUS_FILE="${ARTIFACT_DIR}/rendezvous.env"' in script
    assert "torchrun_rendezvous_args+=(--standalone)" in script
    assert 'export MASTER_PORT="${MASTER_PORT:-' not in script


def test_shell_scripts_do_not_contain_fixed_master_ports():
    for path in ALL_SHELL_SCRIPTS:
        script = path.read_text(encoding="utf-8")
        assert 'MASTER_PORT="${MASTER_PORT:-' not in script, path.relative_to(ROOT)
        assert "MASTER_PORT=${MASTER_PORT:-" not in script, path.relative_to(ROOT)
