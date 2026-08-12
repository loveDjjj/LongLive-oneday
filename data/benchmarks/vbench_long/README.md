# VBench-Long 协议文件

VBench-Long 复用 `../vbench_standard/full/` 的 944 条提示词和元数据，不是另一种 Standard 抽样比例。

本目录只保存协议特有的评测参数，包括各维度 clip 长度、慢速/快速片段内和跨片段聚合，以及主体/背景分数校准。官方 VBench-Long evaluator 尚未完成昇腾适配，因此当前 `scripts/evaluation/run_vbench.sh` 不提供对应 preset。
