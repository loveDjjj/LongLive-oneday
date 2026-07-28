# LongLive AISBench adapter

This directory is a lightweight integration layer for an existing AISBench
installation. It does not vendor AISBench or VBench source code and does not
download evaluator checkpoints.

- `prepare_vbench_videos.py` converts LongLive rank/index output names to
  VBench `{prompt}-{sample_index}.mp4` names.
- `eval_longlive_vbench.py` mirrors the public AISBench VBench 1.0 config while
  accepting paths from environment variables.
- `run_vbench_16npu.sh` exposes 16 Ascend NPUs and starts 16 AISBench workers.

Required environment variables can override the defaults:

```bash
export LONGLIVE_VBENCH_DATA_PATH=/path/to/prepared/videos
export LONGLIVE_VBENCH_FULL_INFO=/path/to/VBench_full_info.json
export VBENCH_CACHE_DIR=/path/to/existing/vbench/cache
export AISBENCH_MAX_WORKERS=16
```

See `docs/NPU_BF16_RUN_GUIDE.md` for the complete Chinese workflow.

The launcher prepends `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` and verifies
that `decord` can load before starting AISBench. This avoids accidentally
loading an older `/usr/lib64/libstdc++.so.6` on Ascend servers.
