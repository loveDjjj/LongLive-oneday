# 训练提示词数据

大型提示词文件不提交到 Git。运行：

```bash
bash scripts/data/prepare_training_data.sh
```

脚本默认离线：先复用通过 SHA256 和数量校验的 `prompts_train.txt`，否则查找本地 `source_prompts.txt` 或 `vidprom_filtered_extended.txt`。也可以指定：

```bash
SOURCE_FILE=/path/to/vidprom_filtered_extended.txt \
bash scripts/data/prepare_training_data.sh
```

只有显式设置 `ALLOW_DOWNLOAD=1` 时才会访问 Hugging Face。源文件预期包含 248221 条提示词，SHA256 为：

```text
7896742f468bc8aef9e4547424d1ce0a951acdb2a82233790155401a99bf5aa5
```

准备过程会去重，并移除与完整 VBench Standard 和 Augmented 提示词规范化后完全相同的样本。输出应包含 248217 条提示词，SHA256 为：

```text
c5ca345c5cb83db295dee0dda0f06530032e5ea2fe0e83c6fe686a4111b02623
```

VidProM 衍生数据应按 CC BY-NC 4.0 的适用范围使用。文件结构、Dataset 契约、训练样本计算和运行产物见 [SLA+CAG 训练指南](../../docs/sla_cag_training.md)。
