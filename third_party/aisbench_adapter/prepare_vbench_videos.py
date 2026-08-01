#!/usr/bin/env python3
"""Convert LongLive rank-index video names to the VBench naming convention."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DEFAULTS = {
    "standard": (
        REPO_ROOT / "runs/vbench/manual/videos/raw",
        REPO_ROOT / "data/benchmarks/vbench_standard/full/prompts.txt",
        REPO_ROOT / "runs/vbench/manual/videos/prepared",
    ),
}
VIDEO_PATTERN = re.compile(r"^rank\d+-(\d+)-(\d+)_.*\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LongLive outputs for AISBench/VBench evaluation."
    )
    parser.add_argument("--benchmark", choices=BENCHMARK_DEFAULTS, default="standard")
    parser.add_argument("--src-dir", type=Path, help="LongLive output directory")
    parser.add_argument("--prompts-file", type=Path, help="Prompt file used for generation")
    parser.add_argument("--dst-dir", type=Path, help="VBench-compatible output directory")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="VBench sample index assigned to this generation run",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow partial generation output instead of requiring every prompt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files for the selected sample index",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_src, default_prompts, default_dst = BENCHMARK_DEFAULTS[args.benchmark]
    src_dir = (args.src_dir or default_src).resolve()
    prompts_file = (args.prompts_file or default_prompts).resolve()
    dst_dir = (args.dst_dir or default_dst).resolve()

    if args.sample_index < 0:
        raise ValueError("--sample-index must be non-negative")
    if not src_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {src_dir}")
    if not prompts_file.is_file():
        raise FileNotFoundError(f"prompt file does not exist: {prompts_file}")

    prompts = [line.strip() for line in prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"prompt file is empty: {prompts_file}")

    videos_by_prompt: dict[int, Path] = {}
    for video_path in sorted(src_dir.glob("rank*-*-*_*.mp4")):
        match = VIDEO_PATTERN.match(video_path.name)
        if match is None:
            continue
        prompt_index = int(match.group(1))
        batch_sample_index = int(match.group(2))
        if batch_sample_index != 0:
            continue
        if prompt_index >= len(prompts):
            raise IndexError(
                f"{video_path.name} uses prompt index {prompt_index}, "
                f"but {prompts_file} contains only {len(prompts)} prompts"
            )
        if prompt_index in videos_by_prompt:
            raise RuntimeError(
                f"multiple videos found for prompt index {prompt_index}: "
                f"{videos_by_prompt[prompt_index].name}, {video_path.name}"
            )
        videos_by_prompt[prompt_index] = video_path

    missing = sorted(set(range(len(prompts))) - videos_by_prompt.keys())
    if missing and not args.allow_missing:
        preview = ", ".join(map(str, missing[:10]))
        raise RuntimeError(
            f"missing {len(missing)} of {len(prompts)} prompts (first indices: {preview}); "
            "use --allow-missing only for a partial smoke test"
        )

    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for prompt_index, src_path in sorted(videos_by_prompt.items()):
        prompt = prompts[prompt_index]
        if "/" in prompt or "\0" in prompt:
            raise ValueError(f"prompt cannot be used as a file name: {prompt!r}")
        dst_path = dst_dir / f"{prompt}-{args.sample_index}.mp4"
        if dst_path.exists() and not args.overwrite:
            raise FileExistsError(f"destination already exists: {dst_path}")
        shutil.copy2(src_path, dst_path)
        copied += 1

    print(
        f"prepared {copied}/{len(prompts)} videos for sample index "
        f"{args.sample_index} in {dst_dir}"
    )


if __name__ == "__main__":
    main()
