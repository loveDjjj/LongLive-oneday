import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "evaluation"


def _run(script: str, *arguments: str) -> str:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_benchmark_summary_writes_machine_readable_result(tmp_path):
    log = tmp_path / "torchrun.log"
    log.write_text(
        "\n".join(
            [
                "[benchmark] rank=0 generation_seconds=99 generation_fps=0 rtf=0",
                "[benchmark] rank=0 generation_seconds=10 generation_fps=2 rtf=1 "
                "save_seconds=3 peak_memory_gb=20 vae_peak_memory_gb=7 "
                "ar_loop_seconds=6 vae_decode_seconds=8 vae_enqueue_seconds=1 "
                "vae_drain_seconds=2 vae_overlap_seconds=6 vae_chunks=4 "
                "vae_queue_peak=2",
                "[benchmark] rank=0 generation_seconds=14 generation_fps=4 rtf=2 "
                "save_seconds=5 peak_memory_gb=22 vae_peak_memory_gb=8 "
                "ar_loop_seconds=8 vae_decode_seconds=10 vae_enqueue_seconds=1.4 "
                "vae_drain_seconds=4 vae_overlap_seconds=6 vae_chunks=4 "
                "vae_queue_peak=3",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"

    stdout = _run(
        "summarize_benchmark.py",
        str(log),
        "--warmup-per-rank",
        "1",
        "--json-output",
        str(output),
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["records"] == 2
    assert result["generation_seconds_mean"] == 12
    assert result["peak_memory_gb_max"] == 22
    assert result["ar_loop_seconds_mean"] == 7
    assert result["vae_decode_seconds_mean"] == 9
    assert result["vae_enqueue_seconds_mean"] == 1.2
    assert result["vae_drain_seconds_mean"] == 3
    assert result["vae_overlap_seconds_mean"] == 6
    assert result["vae_chunks_mean"] == 4
    assert result["vae_queue_peak_mean"] == 2.5
    assert "generation_seconds mean=12.000" in stdout
    assert "vae_overlap_seconds_mean=6.000" in stdout


def test_vbench_summary_preserves_official_aggregates(tmp_path):
    results = tmp_path / "results" / "vbench_eval"
    results.mkdir(parents=True)
    dimensions = (
        "subject_consistency background_consistency aesthetic_quality imaging_quality "
        "object_class multiple_objects color spatial_relationship scene temporal_style "
        "overall_consistency human_action temporal_flickering motion_smoothness "
        "dynamic_degree appearance_style"
    ).split()
    for index, dimension in enumerate(dimensions):
        (results / f"vbench_{dimension}.json").write_text(
            json.dumps({"accuracy": 80 + index / 10}), encoding="utf-8"
        )
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    with (summary_dir / "summary_1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "version", "metric", "mode", "vbench_eval"])
        writer.writerow(["vbench_quality", "-", "accuracy", "gen", "85.76"])
        writer.writerow(["vbench_semantic", "-", "accuracy", "gen", "68.71"])
        writer.writerow(["vbench_total", "-", "accuracy", "gen", "82.35"])
    output = tmp_path / "vbench_results.json"

    _run(
        "summarize_vbench.py",
        "--work-dir",
        str(tmp_path),
        "--output",
        str(output),
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(result["dimensions"]) == 16
    assert result["official_aggregates"] == {
        "vbench_quality": 85.76,
        "vbench_semantic": 68.71,
        "vbench_total": 82.35,
    }


def test_performance_suite_computes_dense_speedup_and_checkpoint_flag(tmp_path):
    runs_root = tmp_path / "runs"
    for method, latency, checkpoint in (
        ("dense", 48.0, "shared.pt"),
        ("sla_cag", 36.0, "shared.pt"),
        ("hsa_sla_cag", 32.0, "hybrid.pt"),
    ):
        run_dir = runs_root / f"suite-{method}-32s-dit_only"
        run_dir.mkdir(parents=True)
        manifest = {
            "task": "benchmark",
            "preset": "longlive2_32s",
            "vae_mode": "dit_only",
            "sparsity_method": method,
            "sparsity_backend": "dense" if method == "dense" else "mindiesd",
            "generator_checkpoint": checkpoint,
            "latent_frames": 192,
            "pixel_frames": 765,
        }
        summary = {
            "records": 3,
            "generation_seconds_mean": latency,
            "generation_seconds_p50": latency,
            "generation_seconds_p95": latency,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    output_dir = tmp_path / "suite_output"

    _run(
        "summarize_suite.py",
        "benchmark",
        "--suite-id",
        "suite",
        "--runs-root",
        str(runs_root),
        "--output-dir",
        str(output_dir),
    )

    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    rows = {row["method"]: row for row in payload["rows"]}
    assert rows["sla_cag"]["dense_speedup"] == 48 / 36
    assert rows["sla_cag"]["checkpoint_matched_to_dense"] is True
    assert rows["hsa_sla_cag"]["dense_speedup"] == 1.5
    assert rows["hsa_sla_cag"]["checkpoint_matched_to_dense"] is False
    assert (output_dir / "results.csv").is_file()
