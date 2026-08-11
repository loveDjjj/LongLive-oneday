#!/usr/bin/env python3
"""Write reproducibility metadata for a training or inference run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: str | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for module_name in ("torch", "torch_npu", "triton"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--kind", choices=("training", "inference"), required=True)
    parser.add_argument("--config")
    parser.add_argument("--data")
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--sp-size", type=int)
    parser.add_argument("--dp-size", type=int)
    parser.add_argument("--effective-batch", type=int)
    args = parser.parse_args()

    git_commit = _command("git", "rev-parse", "HEAD")
    git_status = _command("git", "status", "--porcelain")
    manifest = {
        "schema_version": 1,
        "kind": args.kind,
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "git": {"commit": git_commit, "dirty": bool(git_status)},
        "distributed": {
            "world_size": args.world_size,
            "sequence_parallel_size": args.sp_size,
            "data_parallel_size": args.dp_size,
            "effective_batch_size": args.effective_batch,
        },
        "inputs": {
            "config": args.config,
            "config_sha256": _sha256(args.config),
            "data": args.data,
            "data_sha256": _sha256(args.data),
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "MASTER_ADDR",
                "MASTER_PORT",
                "SLA_BACKEND",
                "PYTORCH_NPU_ALLOC_CONF",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
