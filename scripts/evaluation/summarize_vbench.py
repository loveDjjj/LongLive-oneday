#!/usr/bin/env python3
"""Collect one AISBench VBench session into a stable JSON result."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "appearance_style",
)
AGGREGATES = ("vbench_quality", "vbench_semantic", "vbench_total")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-summary-to", type=Path)
    return parser.parse_args()


def _accuracy(value) -> float:
    if isinstance(value, dict):
        if "accuracy" in value and isinstance(value["accuracy"], (int, float)):
            return float(value["accuracy"])
        matches = []
        for nested in value.values():
            try:
                matches.append(_accuracy(nested))
            except ValueError:
                pass
        if len(matches) == 1:
            return matches[0]
    raise ValueError("result JSON does not contain one numeric accuracy value")


def _official_summary(work_dir: Path, copy_to: Path | None) -> tuple[list[dict], dict]:
    summary_files = sorted(work_dir.glob("**/summary/summary_*.csv"))
    rows: list[dict] = []
    aggregates = {}
    if summary_files:
        source = summary_files[-1]
        with source.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            dataset = row.get("dataset", "")
            if dataset not in AGGREGATES:
                continue
            for key, value in row.items():
                if key in {"dataset", "version", "metric", "mode"} or not value:
                    continue
                try:
                    aggregates[dataset] = float(value)
                    break
                except ValueError:
                    continue

    if copy_to is not None:
        copy_to.mkdir(parents=True, exist_ok=True)
        for suffix in ("csv", "md", "txt"):
            candidates = sorted(work_dir.glob(f"**/summary/summary_*.{suffix}"))
            if candidates:
                shutil.copy2(candidates[-1], copy_to / f"aisbench_summary.{suffix}")
    return rows, aggregates


def collect(work_dir: Path, copy_to: Path | None = None) -> dict:
    result_files = sorted(work_dir.glob("**/results/vbench_eval/vbench_*.json"))
    dimensions = {}
    sources = {}
    for path in result_files:
        name = path.stem.removeprefix("vbench_")
        if name not in DIMENSIONS:
            continue
        if name in dimensions:
            raise RuntimeError(f"duplicate VBench result for {name}: {path}")
        dimensions[name] = _accuracy(json.loads(path.read_text(encoding="utf-8")))
        sources[name] = str(path)
    missing = sorted(set(DIMENSIONS) - dimensions.keys())
    if missing:
        raise RuntimeError(f"missing VBench dimensions: {', '.join(missing)}")

    summary_rows, aggregates = _official_summary(work_dir, copy_to)
    return {
        "schema_version": 1,
        "dimensions": {name: dimensions[name] for name in DIMENSIONS},
        "official_aggregates": aggregates,
        "official_summary_rows": summary_rows,
        "source_files": sources,
    }


def main() -> None:
    args = parse_args()
    result = collect(args.work_dir, args.copy_summary_to)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    aggregates = result["official_aggregates"]
    aggregate_text = " ".join(f"{key}={value:.2f}" for key, value in aggregates.items())
    print(f"vbench_dimensions={len(result['dimensions'])} {aggregate_text}".rstrip())


if __name__ == "__main__":
    main()
