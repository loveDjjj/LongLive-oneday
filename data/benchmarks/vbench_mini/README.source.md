---
license: apache-2.0
task_categories:
  - text-to-video
size_categories: n<1K
---

# VBench-1.0-mini
## 简介
VBench-1.0-mini 是基于[VBench-1.0的数据集](https://github.com/Vchitect/VBench/tree/master/prompts/prompts_per_dimension)进行约1/10规模的采样得到的一个小规模数据集，它在测试得分上与原始数据集大致相同。

## 数据集使用方式
数据集目录结构如下：
```shell
VBench-1.0-mini/
├── process_data # 采样过程数据
│   ├── vbench_metadata_compressed_0.10 # 采样后的元数据（包含kmeans采样和随机采样），同时包含kmeans算法的聚类效果可视化
│   └── vbench_metadata_figures_0.10 # 采样效果图
├── prompts_per_dimension_kmeans # 基于 https://github.com/Vchitect/VBench/tree/master/prompts/prompts_per_dimension 使用kmeans采样的VBench 1.0数据集
│   ├── appearance_style.txt
│   ├── color.txt
│   ├── human_action.txt
│   ├── multiple_objects.txt
│   ├── object_class.txt
│   ├── overall_consistency.txt
│   ├── scene.txt
│   ├── spatial_relationship.txt
│   ├── subject_consistency.txt
│   ├── temporal_flickering.txt
│   └── temporal_style.txt
├── README.md
└── VBench_kmeans_info.json # 采样后的 https://github.com/Vchitect/VBench/blob/master/vbench/VBench_full_info.json
```

### 推理过程使用
直接使用`prompts_per_dimension_kmeans`中的提示词推理。

### 评估过程使用
执行评估前在安装了VBench的环境（linux环境）执行如下命令找到vbench所在路径：
```bash
pip show vbench | grep Location
```
得到Location，进入`{Location}/vbench`路径中，此路径下有`VBench_full_info.json`文件，用`VBench_kmeans_info.json`替换即可（做好备份）：
```bash
# 备份原始数据集的json
mv VBench_full_info.json VBench_full_info.json.bak_origin
# 用VBench 1.0 mini的json替换
mv {your_path}/VBench_kmeans_info.json VBench_full_info.json
```

## 采样原理介绍
采样过程kmeans算法实现。对应每个子集，将某个典型模型在此子集上某个得分指标作为特性（存在大量这样的特征）对子集的每个case进行聚类，具体聚类流程如下：

### 1. K-Means 聚类
- 将数据点根据特征相似度聚类成指定数量的簇
- 每个簇代表一个数据特征组
- 簇中心是该组数据的特征平均值

### 2. 代表性样本抽取
- 首先从每个簇中选择距离中心最近的数据点（保证代表性）
- 剩余样本根据簇大小按比例随机分配
- 确保保留数据的整体分布特征

### 3. 最优簇数计算
- 默认使用 `n_clusters = min(sqrt(n_samples), n_samples)`
- 同时考虑唯一数据组合数，避免无效聚类
- 可选自动优化模式，根据平均分数相似度寻找最佳簇数

## Kmeans采样效果展示
我们提取了OpenSora2，Sora，WAN2.1，WAN2.2，Vidu，Kling-1.6，HuanyuanVideo,Veo 3 这8个模型在VBench-1.0 11个子集上的16个评分指标，作为Kmeans聚类的特征。（注：这8个模型的推理生成的视频均来自[VBench Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)）

下图展示了通过Kmeans聚类采样得到的VBench-1.0-mini与原始数据集、随机采样的数据集在不同模型不同评分指标上的得分分布对比，可以看到VBench-1.0-mini 采样数据集与VBench-1.0 原始数据集非常接近且整体接近程度远好于随机采样。
- **Full Dataset**: VBench-1.0 原始数据集
- **Kmeans Representative**: VBench-1.0-mini 采样数据集
- **Random**: 随机采样的数据集

![](process_data/vbench_metadata_figures_0.10/metadata_appearance_style_means_comparison.png)
![](process_data/vbench_metadata_figures_0.10/metadata_color_means_comparison.png)
![](process_data/vbench_metadata_figures_0.10/metadata_human_action_means_comparison.png)
![](process_data/vbench_metadata_figures_0.10/metadata_multiple_objects_means_comparison.png)
![](process_data/vbench_metadata_figures_0.10/metadata_object_class_means_comparison.png)
![](process_data/vbench_metadata_figures_0.10/metadata_overall_consistency_means_comparison.png)
![](process_data/vbench_metadata_figures_0.10/metadata_scene_means_comparison.png)
![](process_data/vbench_metadata_figures_0.10/metadata_spatial_relationship_means_comparison.png)
![](process_data/vbench_metadata_figures_0.10/metadata_subject_consistency_means_comparison.png)
![](process_data/vbench_metadata_figures_0.10/metadata_temporal_flickering_means_comparison.png)
![](process_data/vbench_metadata_figures_0.10/metadata_temporal_style_means_comparison.png)

## 复现采样过程
8个模型的得分特征数据：[📄vbench_metadata](https://github.com/AISBench/datasets/tree/main/mini_datasets/vbench_1.0_mini/vbench_metadata)
参考[VBench-1.0-mini](https://github.com/AISBench/datasets/tree/main/mini_datasets/vbench_1.0_mini/)复现采样过程。

## LICENSE
[apache-2.0](https://github.com/AISBench/datasets/blob/main/LICENSE)