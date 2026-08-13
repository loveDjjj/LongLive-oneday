# LongLive AISBench 适配层

本目录是对现有 AISBench 安装的轻量集成层，不包含 AISBench 或 VBench 源码，也不会下载 evaluator checkpoint。

- `prepare_vbench_videos.py`：将 LongLive 的 rank/index 输出文件名转换为 VBench 要求的 `{prompt}-{sample_index}.mp4`。
- `eval_longlive_vbench.py`：运行时渲染的配置模板，与公开 AISBench VBench 1.0 配置保持一致。MMEngine 使用 lazy mode 解析配置，因此该文件不调用 `os.environ`。
- `run_vbench_eval.sh`：使用当前可见的昇腾设备和指定 worker 数启动 AISBench evaluator。

可通过环境变量覆盖默认路径：

```bash
export LONGLIVE_VBENCH_DATA_PATH=/path/to/prepared/videos
export LONGLIVE_VBENCH_FULL_INFO=/path/to/VBench_full_info.json
export VBENCH_CACHE_DIR=/mnt/a800_share/r50063443/vbench_models
export AISBENCH_MAX_WORKERS=16
```

完整生成与评测流程见[推理与评测指南](../../docs/inference_and_evaluation.md)。

启动器会加载 `/mnt/a800_share/r50063443/conda_envs/cann-8.5/Ascend/cann-8.5.0/set_env.sh`，将 `$CONDA_PREFIX/lib` 加到 `LD_LIBRARY_PATH` 前部，并在启动 AISBench 前检查 `torch_npu` 与 `decord`。这些检查用于避免缺少 `libhccl.so` 或错误加载旧版系统 C++ runtime。
