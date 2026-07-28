"""AISBench VBench 1.0 config for LongLive-generated videos.

Run this file through the ``ais_bench`` command. Environment variables can
override all machine-specific paths without editing the AISBench repository.
"""

import os
from pathlib import Path

from ais_bench.benchmark.datasets import VBenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.summarizers import VBenchSummarizer
from ais_bench.benchmark.tasks import VBenchEvalTask


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = os.environ.get(
    "LONGLIVE_VBENCH_DATA_PATH",
    str(REPO_ROOT / "videos/benchmarks/vbench_mini_5s_vbench"),
)
FULL_JSON_PATH = os.environ.get(
    "LONGLIVE_VBENCH_FULL_INFO",
    str(REPO_ROOT / "data/benchmarks/vbench_mini/VBench_full_info.json"),
)
VBENCH_CACHE_DIR = os.environ.get(
    "VBENCH_CACHE_DIR",
    str(Path.home() / ".cache/vbench"),
)

VBENCH_DEFAULT_DIMENSIONS = [
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
]

models = [
    dict(attr="local", type="VBenchEvalPlaceholder", abbr="vbench_eval"),
]

vbench_eval_cfg = dict(
    load_ckpt_from_local=True,
    full_json_dir=FULL_JSON_PATH,
    device="npu",
)

datasets = [
    dict(
        abbr=f"vbench_{dimension}",
        type=VBenchDataset,
        path=DATA_PATH,
        eval_cfg=dict(**vbench_eval_cfg, dimension_list=[dimension]),
    )
    for dimension in VBENCH_DEFAULT_DIMENSIONS
]

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, task=dict(type=VBenchEvalTask)),
)

summarizer = dict(attr="accuracy", type=VBenchSummarizer)
