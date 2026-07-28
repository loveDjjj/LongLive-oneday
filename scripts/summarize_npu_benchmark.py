#!/usr/bin/env python3
"""Summarize LongLive [benchmark] records from torchrun logs."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path


NUMERIC_FIELDS = {
    "rank": int,
    "prompt_index": int,
    "generation_seconds": float,
    "save_seconds": float,
    "pixel_frames": int,
    "video_seconds": float,
    "generation_fps": float,
    "rtf": float,
    "peak_memory_gb": float,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+", help="torchrun log files")
    parser.add_argument(
        "--warmup-per-rank",
        type=int,
        default=1,
        help="number of initial records discarded independently for each rank",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def read_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            marker = line.find("[benchmark]")
            if marker < 0:
                continue
            fields = {}
            for token in line[marker + len("[benchmark]"):].split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                converter = NUMERIC_FIELDS.get(key)
                if converter is None or value == "n/a":
                    continue
                fields[key] = converter(value)
            required = {"rank", "generation_seconds", "generation_fps", "rtf"}
            if required <= fields.keys():
                records.append(fields)
    return records


def main() -> None:
    args = parse_args()
    if args.warmup_per_rank < 0:
        raise ValueError("--warmup-per-rank must be non-negative")

    grouped = defaultdict(list)
    for record in read_records(args.logs):
        grouped[record["rank"]].append(record)
    measured = [
        record
        for rank_records in grouped.values()
        for record in rank_records[args.warmup_per_rank:]
    ]
    if not measured:
        raise RuntimeError("no measured benchmark records found after warmup removal")

    latencies = [record["generation_seconds"] for record in measured]
    fps_values = [record["generation_fps"] for record in measured]
    rtf_values = [record["rtf"] for record in measured]
    save_values = [record.get("save_seconds", 0.0) for record in measured]
    peak_values = [record["peak_memory_gb"] for record in measured if "peak_memory_gb" in record]
    per_rank_seconds = defaultdict(float)
    for record in measured:
        per_rank_seconds[record["rank"]] += record["generation_seconds"]
    concurrent_wall_seconds = max(per_rank_seconds.values())

    print(f"records={len(measured)} ranks={','.join(map(str, sorted(per_rank_seconds)))}")
    print(
        f"generation_seconds mean={statistics.fmean(latencies):.3f} "
        f"p50={statistics.median(latencies):.3f} p95={percentile(latencies, 0.95):.3f}"
    )
    print(
        f"generation_fps mean={statistics.fmean(fps_values):.3f} "
        f"rtf_mean={statistics.fmean(rtf_values):.3f} "
        f"save_seconds_mean={statistics.fmean(save_values):.3f}"
    )
    if peak_values:
        print(f"peak_memory_gb_max={max(peak_values):.2f}")
    print(
        f"generation_videos_per_hour={len(measured) / concurrent_wall_seconds * 3600:.2f} "
        f"estimated_concurrent_wall_seconds={concurrent_wall_seconds:.3f}"
    )


if __name__ == "__main__":
    main()
