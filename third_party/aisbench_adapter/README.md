# LongLive AISBench adapter

This directory is a lightweight integration layer for an existing AISBench
installation. It does not vendor AISBench or VBench source code and does not
download evaluator checkpoints.

- `prepare_vbench_videos.py` converts LongLive rank/index output names to
  VBench `{prompt}-{sample_index}.mp4` names.
- `eval_longlive_vbench.py` is a runtime-rendered template mirroring the public
  AISBench VBench 1.0 config. It intentionally contains no `os.environ` calls
  because MMEngine parses configs in lazy mode.
- `run_vbench_16npu.sh` exposes 16 Ascend NPUs and starts 16 AISBench workers.

Required environment variables can override the defaults:

```bash
export LONGLIVE_VBENCH_DATA_PATH=/path/to/prepared/videos
export LONGLIVE_VBENCH_FULL_INFO=/path/to/VBench_full_info.json
export VBENCH_CACHE_DIR=/mnt/weight/vbench_models/
export AISBENCH_MAX_WORKERS=1
```

See `docs/NPU_BF16_RUN_GUIDE.md` for the complete Chinese workflow.

The launcher sources `/usr/local/Ascend/ascend-toolkit/set_env.sh`, prepends `$CONDA_PREFIX/lib` to
`LD_LIBRARY_PATH`, and verifies both `torch_npu` and `decord` before starting
AISBench. This avoids missing `libhccl.so` or loading an old system C++ runtime.
