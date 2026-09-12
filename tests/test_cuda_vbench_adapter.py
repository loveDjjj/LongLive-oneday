import json
from types import SimpleNamespace
import sys

import pytest

from scripts.evaluation.summarize_vbench import DIMENSIONS, collect
from third_party.aisbench_adapter.eval_cuda_vbench import (
    load_official_scoring,
    main,
    write_dimension,
)


def test_official_cuda_api_and_existing_summary_contract(tmp_path, monkeypatch):
    source = tmp_path / "upstream"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "constant.py").write_text("# Stub official source marker\n")
    (scripts / "cal_final_score.py").write_text(
        "def get_nomalized_score(scores):\n"
        "    assert len(scores) == 16 and 'subject consistency' in scores\n"
        "    assert scores['subject consistency'] == 0.5\n"
        "    return {'official_normalized': True}\n"
        "def get_quality_score(scores):\n"
        "    assert scores == {'official_normalized': True}\n"
        "    return 0.8\n"
        "def get_semantic_score(scores):\n"
        "    return 0.6\n"
        "def get_final_score(quality, semantic):\n"
        "    assert quality == 0.8 and semantic == 0.6\n"
        "    return 0.73\n"
    )
    videos = tmp_path / "videos"
    videos.mkdir()
    for index in range(5):
        (videos / f"example-{index}.mp4").touch()
    full_info = tmp_path / "full_info.json"
    full_info.write_text(json.dumps([{"prompt_en": "example", "dimension": list(DIMENSIONS)}]))
    work = tmp_path / "work"
    calls = []

    class FakeVBench:
        def __init__(self, device, full_info_dir, output_path):
            assert device == "device:cuda:0"
            assert full_info_dir == str(full_info)
            self.output = work / "official"
            self.output.mkdir(parents=True)

        def evaluate(self, *, videos_path, name, dimension_list, local, read_frame, mode):
            assert videos_path == str(videos)
            assert dimension_list == [name]
            assert local is True and read_frame is False and mode == "vbench_standard"
            calls.append(name)
            (self.output / f"{name}_eval_results.json").write_text(json.dumps({name: [0.5, []]}))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        device=lambda value: f"device:{value}",
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
    ))
    monkeypatch.setitem(sys.modules, "vbench", SimpleNamespace(
        VBench=FakeVBench, __file__=str(source / "vbench" / "__init__.py"),
    ))
    monkeypatch.setattr(sys, "argv", ["eval_cuda_vbench.py"])
    for key, value in {
        "VBENCH_SOURCE_DIR": source,
        "LONGLIVE_VBENCH_DATA_PATH": videos,
        "LONGLIVE_VBENCH_FULL_INFO": full_info,
        "AISBENCH_WORK_DIR": work,
    }.items():
        monkeypatch.setenv(key, str(value))
    main()
    result = collect(work)
    assert calls == list(DIMENSIONS)
    assert result["dimensions"] == {dimension: 50.0 for dimension in DIMENSIONS}
    assert result["official_aggregates"] == {
        "vbench_quality": 80.0, "vbench_semantic": 60.0, "vbench_total": 73.0,
    }
    assert (work / "official" / "subject_consistency_eval_results.json").is_file()


def test_cuda_adapter_requires_official_scoring_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="VBENCH_SOURCE_DIR"):
        load_official_scoring(tmp_path)


@pytest.mark.parametrize("result", [[float("nan"), []], {"accuracy": 1}])
def test_cuda_adapter_rejects_invalid_official_scores(tmp_path, result):
    with pytest.raises(ValueError):
        write_dimension(tmp_path, "subject_consistency", result)
