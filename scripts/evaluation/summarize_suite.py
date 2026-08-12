#!/usr/bin/env python3
"""Aggregate a LongLive performance or VBench matrix into JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from summarize_vbench import AGGREGATES, DIMENSIONS


PERFORMANCE_FIELDS = (
    "records",
    "generation_seconds_mean",
    "generation_seconds_p50",
    "generation_seconds_p95",
    "generation_fps_mean",
    "rtf_mean",
    "save_seconds_mean",
    "peak_memory_gb_max",
    "vae_peak_memory_gb_max",
    "generation_videos_per_hour",
    "estimated_concurrent_wall_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("benchmark", "msprof", "vbench"))
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _common(run_dir: Path, manifest: dict) -> dict:
    return {
        "run_id": run_dir.name,
        "task": manifest["task"],
        "profiled": manifest["task"] == "msprof",
        "preset": manifest["preset"],
        "method": manifest["sparsity_method"],
        "backend": manifest["sparsity_backend"],
        "generator_checkpoint": manifest.get("generator_checkpoint"),
        "latent_frames": manifest.get("latent_frames"),
        "pixel_frames": manifest.get("pixel_frames"),
    }


def _performance_rows(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        manifest = _load(run_dir / "manifest.json")
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing structured result: {summary_path}")
        row = _common(run_dir, manifest)
        row["vae_mode"] = manifest["vae_mode"]
        summary = _load(summary_path)
        row.update({field: summary.get(field) for field in PERFORMANCE_FIELDS})
        rows.append(row)

    dense = {}
    for row in rows:
        if row["method"] != "dense":
            continue
        key = (row["preset"], row["vae_mode"])
        if key in dense:
            raise RuntimeError(f"duplicate dense reference for {key}")
        dense[key] = row
    for row in rows:
        reference = dense.get((row["preset"], row["vae_mode"]))
        row["dense_speedup"] = (
            reference["generation_seconds_mean"] / row["generation_seconds_mean"]
            if reference is not None
            else None
        )
        row["checkpoint_matched_to_dense"] = (
            row["generator_checkpoint"] == reference["generator_checkpoint"]
            if reference is not None
            else None
        )
    return rows


def _vbench_rows(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        manifest = _load(run_dir / "manifest.json")
        result_path = run_dir / "vbench_results.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"missing VBench result: {result_path}")
        result = _load(result_path)
        row = _common(run_dir, manifest)
        row.update(result["dimensions"])
        row.update(
            {
                name: result.get("official_aggregates", {}).get(name)
                for name in AGGREGATES
            }
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("suite contains no completed runs")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root or Path("runs") / args.task
    output_dir = args.output_dir or Path("runs/suites") / args.suite_id / args.task
    run_dirs = sorted(
        path
        for path in runs_root.glob(f"{args.suite_id}-*")
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    rows = _vbench_rows(run_dirs) if args.task == "vbench" else _performance_rows(run_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "suite_id": args.suite_id,
        "task": args.task,
        "profiled": args.task == "msprof",
        "rows": rows,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "results.csv", rows)
    print(f"suite_rows={len(rows)} json={output_dir / 'results.json'} csv={output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
