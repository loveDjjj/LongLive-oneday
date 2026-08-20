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


def test_hybrid_launchers_expose_main_and_linear_only_scopes():
    main = _read("scripts/training/run_hsa_sla_cag.sh")
    linear_only = _read("scripts/training/run_hsa_sla_cag_linear_only.sh")
    assert "SPARSE_METHOD=hsa_sla_cag" in main
    assert "GENERATOR_TRAIN_SCOPE" not in main
    assert 'LONGLIVE_SP_SIZE="${LONGLIVE_SP_SIZE:-${SP_SIZE:-8}}"' in main
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"' in main
    assert 'MAX_ITERS="${MAX_ITERS:-200}"' in main
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-20}"' in main
    assert 'MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-5}"' in main
    assert "SPARSE_METHOD=hsa_sla_cag" in linear_only
    assert "GENERATOR_TRAIN_SCOPE=linear_only" in linear_only
    assert 'LONGLIVE_SP_SIZE="${LONGLIVE_SP_SIZE:-${SP_SIZE:-8}}"' in linear_only
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"' in linear_only
    assert 'MAX_ITERS="${MAX_ITERS:-200}"' in linear_only
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-20}"' in linear_only
    assert 'MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-5}"' in linear_only


def test_sla_launcher_exposes_16_card_training_defaults():
    script = _read("scripts/training/run_sla_cag.sh")
    assert "SPARSE_METHOD=sla_cag" in script
    assert 'LONGLIVE_SP_SIZE="${LONGLIVE_SP_SIZE:-${SP_SIZE:-8}}"' in script
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"' in script
    assert 'MAX_ITERS="${MAX_ITERS:-200}"' in script
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-20}"' in script
    assert 'MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-5}"' in script


def test_sla_and_hybrid_share_raw_linear_checkpoint_validation():
    script = _read("scripts/training/run_sparse_cag.sh")
    assert '"${SPARSE_METHOD}" == "sla_cag"' in script
    assert '"${SPARSE_METHOD}" == "hsa_sla_cag"' in script
    assert '--expected-method "${SPARSE_METHOD}"' in script


def test_merge_script_exposes_scope_override_for_linear_only_export():
    script = (ROOT / "scripts/checkpoints/merge_lora.py").read_text(encoding="utf-8")
    assert '"--generator_train_scope"' in script
    assert '("lora", "linear_only", "lora_plus_linear")' in script


def test_shell_scripts_do_not_contain_fixed_master_ports():
    for path in ALL_SHELL_SCRIPTS:
        script = path.read_text(encoding="utf-8")
        assert 'MASTER_PORT="${MASTER_PORT:-' not in script, path.relative_to(ROOT)
        assert "MASTER_PORT=${MASTER_PORT:-" not in script, path.relative_to(ROOT)
