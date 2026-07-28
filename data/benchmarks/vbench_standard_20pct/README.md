# VBench Standard 20% K-Means subset

This dataset is a fixed, representative subset of VBench 1.0 Standard for
faster LongLive quality regression tests.

- `prompts.txt`: 186 unique prompts used for generation.
- `VBench_full_info.json`: 187 official metadata records covering all 16
  VBench metrics. One prompt occurs in two metadata records.

The subset was generated from AISBench/datasets commit `26f93b6` using its
official VBench metadata features and K-Means pipeline:

```bash
python mini_datasets/select_metadata_by_kmeans.py \
  --input mini_datasets/vbench_1.0_mini/vbench_metadata \
  --work-dir <output> \
  --compression-ratio 0.2 \
  -a
```

The ratio is applied independently to the 11 source prompt suites before they
are merged. The resulting 186 unique prompts are 19.7% of the 944-prompt full
suite. This subset is suitable for regression and comparison experiments, but
its scores must not be reported as full VBench Standard results.

Source: https://github.com/AISBench/datasets/tree/main/mini_datasets/vbench_1.0_mini
