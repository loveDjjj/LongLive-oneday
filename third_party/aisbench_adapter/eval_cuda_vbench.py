#!/usr/bin/env python3
"""内部适配器：调用官方 CUDA VBench，复用项目的 16 维结果格式。"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluation.summarize_vbench import DIMENSIONS


def load_official_scoring(source_dir: Path):
    """直接使用官方归一化与聚合函数，禁止在此维护另一套权重。"""
    script = source_dir / "scripts" / "cal_final_score.py"
    if not script.is_file() or not script.with_name("constant.py").is_file():
        raise FileNotFoundError(
            "CUDA VBench aggregation requires the official VBench source checkout; "
            "set VBENCH_SOURCE_DIR to its root containing scripts/cal_final_score.py"
        )
    spec = importlib.util.spec_from_file_location("longlive_official_vbench_scoring", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_dimension(work_dir: Path, dimension: str, result) -> float:
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise ValueError(f"unexpected official VBench result for {dimension}")
    score = float(result[0])
    if not math.isfinite(score):
        raise ValueError(f"non-finite VBench score for {dimension}")
    target = work_dir / "results" / "vbench_eval" / f"vbench_{dimension}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "accuracy": score * 100.0,
        "raw_score": score,
        "evaluator": "official_vbench_cuda",
    }, indent=2) + "\n", encoding="utf-8")
    return score


def write_official_summary(work_dir: Path, scores: dict, scoring) -> None:
    if set(scores) != set(DIMENSIONS):
        raise ValueError("official aggregation requires all 16 VBench dimensions")
    normalized = scoring.get_nomalized_score({name.replace("_", " "): value for name, value in scores.items()})
    quality = scoring.get_quality_score(normalized)
    semantic = scoring.get_semantic_score(normalized)
    total = scoring.get_final_score(quality, semantic)
    summary_dir = work_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "summary_official.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "version", "metric", "mode", "vbench_eval"])
        for name, value in (("quality", quality), ("semantic", semantic), ("total", total)):
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite official VBench {name} score")
            writer.writerow([f"vbench_{name}", "official", "accuracy", "eval", value * 100.0])


def main() -> None:
    import torch
    import vbench

    source_dir = Path(os.environ.get("VBENCH_SOURCE_DIR", Path(vbench.__file__).resolve().parents[1]))
    scoring = load_official_scoring(source_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA VBench requires a CUDA PyTorch environment and visible GPU")
    if sys.argv[1:] == ["--check-environment"]:
        print(f"CUDA VBench environment ready; official scoring source={source_dir}")
        return
    videos = Path(os.environ["LONGLIVE_VBENCH_DATA_PATH"])
    full_info = Path(os.environ["LONGLIVE_VBENCH_FULL_INFO"])
    work_dir = Path(os.environ["AISBENCH_WORK_DIR"])
    # 官方 Standard 约定每条提示词 5 个 seed；禁止遗漏后仍输出正式质量分数。
    for entry in json.loads(full_info.read_text(encoding="utf-8")):
        for index in range(5):
            expected = videos / f"{entry['prompt_en']}-{index}.mp4"
            if not expected.is_file():
                raise FileNotFoundError(f"missing VBench Standard sample: {expected}")
    evaluator = vbench.VBench(torch.device("cuda:0"), str(full_info), str(work_dir / "official"))
    scores = {}
    # 逐维释放模型与显存，避免同时驻留 16 组评测模型。原始逐视频结果完整保留。
    for dimension in DIMENSIONS:
        evaluator.evaluate(
            videos_path=str(videos), name=dimension, dimension_list=[dimension],
            local=True, read_frame=False, mode="vbench_standard",
        )
        raw_path = work_dir / "official" / f"{dimension}_eval_results.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        scores[dimension] = write_dimension(work_dir, dimension, raw[dimension])
        torch.cuda.empty_cache()
    write_official_summary(work_dir, scores, scoring)
    source_revision = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    (work_dir / "evaluator.json").write_text(json.dumps({
        "accelerator": "cuda", "evaluator": "official_vbench",
        "module": str(Path(vbench.__file__).resolve()),
        "scoring_source": str(source_dir / "scripts" / "cal_final_score.py"),
        "source_git_revision": source_revision.stdout.strip() if source_revision.returncode == 0 else None,
        "device": "cuda:0", "dimension_scale": "raw_score * 100",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
