# Fabric Image Retrieval

这个项目用 DINOv2 提取布料图片特征，然后做自动聚类和 PCA 可视化。

## 目录准备

图片放在：

```text
Data/images
```

YOLO 标注放在：

```text
Data/labels
```

模型权重放在项目根目录：

```text
dinov2_vitb14_pretrain.pth
```

DINOv2 源码放在：

```text
third_party/dinov2
```

## 运行

先提取每张图片的 image embedding：

```bash
python auto.py
```

输出到：

```text
Data/image_feats
```

再自动聚类：

```bash
python auto_cluster.py
```

输出：

```text
cluster_result_auto.txt
Data/cluster_sheets_auto
```

最后画 PCA 可视化：

```bash
python pca_visualization.py
```

输出：

```text
Data/visualizations/pca_clusters.png
```

## 脚本说明

- `auto.py`：随机取每张图 4 个有效 patch，提取特征并直接聚合成 image embedding。
- `auto_cluster.py`：使用自动聚类算法，不需要手动指定聚类数。
- `pca_visualization.py`：把 image embedding 降到二维并保存图片。
- `softlink.py`：把原始图片目录软链接到 `Data/images`。

## 注意

如果图片是软链接，先确认原始数据盘已经挂载，否则 OpenCV 会读图失败。
