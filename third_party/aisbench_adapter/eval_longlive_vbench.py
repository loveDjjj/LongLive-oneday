"""AISBench VBench 1.0 config for LongLive-generated videos.

The launcher renders the placeholder paths below before passing this config to
``ais_bench``. Keeping this file free of runtime calls is required by
MMEngine's lazy config parser.
"""

from ais_bench.benchmark.datasets import VBenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.summarizers import VBenchSummarizer
from ais_bench.benchmark.tasks import VBenchEvalTask


DATA_PATH = "__LONGLIVE_VBENCH_DATA_PATH__"
FULL_JSON_PATH = "__LONGLIVE_VBENCH_FULL_INFO__"
VBENCH_CACHE_DIR = "__VBENCH_CACHE_DIR__"

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
