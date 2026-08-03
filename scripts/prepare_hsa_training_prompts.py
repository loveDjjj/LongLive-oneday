#!/usr/bin/env python3
"""Prepare the filtered VidProM prompt corpus for sparse post-training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded: set[str] = set()
    exclusion_counts = {}
    for path in args.exclude:
        if not path.exists():
            continue
        prompts = [line.strip() for line in path.open(encoding="utf-8") if line.strip()]
        exclusion_counts[str(path)] = len(prompts)
        excluded.update(normalize(prompt) for prompt in prompts)

    source_count = 0
    duplicate_count = 0
    excluded_count = 0
    seen: set[str] = set()
    output_lines = []
    with args.source.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            prompt = raw.strip()
            if not prompt:
                continue
            source_count += 1
            key = normalize(prompt)
            if key in excluded:
                excluded_count += 1
                continue
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            output_lines.append(prompt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    manifest = {
        "source": str(args.source),
        "source_sha256": file_sha256(args.source),
        "source_prompts": source_count,
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "output_prompts": len(output_lines),
        "removed_duplicates": duplicate_count,
        "removed_evaluation_overlaps": excluded_count,
        "exclusion_files": exclusion_counts,
        "provenance": {
            "dataset": "gdhe17/Self-Forcing vidprom_filtered_extended.txt",
            "upstream": "WenhaoWang/VidProM VidProS",
            "license_note": "Treat as CC BY-NC 4.0 due to VidProM provenance.",
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
