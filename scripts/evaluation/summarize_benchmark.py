#!/usr/bin/env python3
"""Summarize LongLive [benchmark] records from torchrun logs."""

from __future__ import annotations

import argparse
import json
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
    "vae_peak_memory_gb": float,
    "ar_loop_seconds": float,
    "vae_decode_seconds": float,
    "vae_enqueue_seconds": float,
    "vae_drain_seconds": float,
    "vae_overlap_seconds": float,
    "vae_chunks": int,
    "vae_queue_peak": int,
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
    parser.add_argument(
        "--json-output",
        type=Path,
        help="also write the machine-readable summary to this path",
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


def summarize_records(records: list[dict], warmup_per_rank: int) -> dict:
    if warmup_per_rank < 0:
        raise ValueError("warmup_per_rank must be non-negative")

    grouped = defaultdict(list)
    for record in records:
        grouped[record["rank"]].append(record)
    measured = [
        record
        for rank_records in grouped.values()
        for record in rank_records[warmup_per_rank:]
    ]
    if not measured:
        raise RuntimeError("no measured benchmark records found after warmup removal")

    latencies = [record["generation_seconds"] for record in measured]
    fps_values = [record["generation_fps"] for record in measured]
    rtf_values = [record["rtf"] for record in measured]
    save_values = [record.get("save_seconds", 0.0) for record in measured]
    peak_values = [
        record["peak_memory_gb"] for record in measured if "peak_memory_gb" in record
    ]
    vae_peak_values = [
        record["vae_peak_memory_gb"]
        for record in measured
        if "vae_peak_memory_gb" in record
    ]
    optional_means = {}
    for field in (
        "ar_loop_seconds",
        "vae_decode_seconds",
        "vae_enqueue_seconds",
        "vae_drain_seconds",
        "vae_overlap_seconds",
        "vae_chunks",
        "vae_queue_peak",
    ):
        values = [record[field] for record in measured if field in record]
        optional_means[f"{field}_mean"] = (
            statistics.fmean(values) if values else None
        )
    per_rank_seconds = defaultdict(float)
    for record in measured:
        per_rank_seconds[record["rank"]] += record["generation_seconds"]
    concurrent_wall_seconds = max(per_rank_seconds.values())
    return {
        "records": len(measured),
        "ranks": sorted(per_rank_seconds),
        "generation_seconds_mean": statistics.fmean(latencies),
        "generation_seconds_p50": statistics.median(latencies),
        "generation_seconds_p95": percentile(latencies, 0.95),
        "generation_fps_mean": statistics.fmean(fps_values),
        "rtf_mean": statistics.fmean(rtf_values),
        "save_seconds_mean": statistics.fmean(save_values),
        "peak_memory_gb_max": max(peak_values) if peak_values else None,
        "vae_peak_memory_gb_max": max(vae_peak_values) if vae_peak_values else None,
        "generation_videos_per_hour": (
            len(measured) / concurrent_wall_seconds * 3600
        ),
        "estimated_concurrent_wall_seconds": concurrent_wall_seconds,
        **optional_means,
    }


def format_summary(summary: dict) -> str:
    lines = [
        f"records={summary['records']} ranks={','.join(map(str, summary['ranks']))}",
        (
            f"generation_seconds mean={summary['generation_seconds_mean']:.3f} "
            f"p50={summary['generation_seconds_p50']:.3f} "
            f"p95={summary['generation_seconds_p95']:.3f}"
        ),
        (
            f"generation_fps mean={summary['generation_fps_mean']:.3f} "
            f"rtf_mean={summary['rtf_mean']:.3f} "
            f"save_seconds_mean={summary['save_seconds_mean']:.3f}"
        ),
    ]
    if summary["peak_memory_gb_max"] is not None:
        lines.append(f"peak_memory_gb_max={summary['peak_memory_gb_max']:.2f}")
    if summary["vae_peak_memory_gb_max"] is not None:
        lines.append(f"vae_peak_memory_gb_max={summary['vae_peak_memory_gb_max']:.2f}")
    if summary["ar_loop_seconds_mean"] is not None:
        lines.append(
            f"ar_loop_seconds_mean={summary['ar_loop_seconds_mean']:.3f} "
            f"vae_decode_seconds_mean={summary['vae_decode_seconds_mean']:.3f} "
            f"vae_enqueue_seconds_mean={summary['vae_enqueue_seconds_mean']:.3f} "
            f"vae_drain_seconds_mean={summary['vae_drain_seconds_mean']:.3f} "
            f"vae_overlap_seconds_mean={summary['vae_overlap_seconds_mean']:.3f} "
            f"vae_chunks_mean={summary['vae_chunks_mean']:.1f} "
            f"vae_queue_peak_mean={summary['vae_queue_peak_mean']:.1f}"
        )
    lines.append(
        f"generation_videos_per_hour={summary['generation_videos_per_hour']:.2f} "
        "estimated_concurrent_wall_seconds="
        f"{summary['estimated_concurrent_wall_seconds']:.3f}"
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = summarize_records(read_records(args.logs), args.warmup_per_rank)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(format_summary(summary))


if __name__ == "__main__":
    main()
