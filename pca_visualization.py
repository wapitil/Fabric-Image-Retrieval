from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA


def load_image_features(img_feat_dir):
    feats = []
    names = []

    for f in sorted(img_feat_dir.glob("*.npy")):
        feats.append(np.load(str(f)))
        names.append(f.stem)

    if len(feats) == 0:
        print("没有找到image embedding文件")
        return

    feats = np.stack(feats, axis=0)
    return feats, names


def load_cluster_labels(cluster_file, names):
    label_map = {}
    if cluster_file.exists():
        with cluster_file.open("r") as f:
            for line in f:
                name, label = line.strip().split()
                label_map[name] = int(label)

    labels = [label_map.get(name, 0) for name in names]
    return labels


def save_pca_visualization(feats, names, labels, output_file):
    """对image embedding做PCA降维并保存散点图"""
    feats_2d = PCA(n_components=2, random_state=0).fit_transform(feats)

    canvas_h, canvas_w = 1200, 1600
    margin = 80
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # 取出所有数据的 第一个PCA坐标 和 第二个PCA坐标
    x_vals = feats_2d[:, 0]
    y_vals = feats_2d[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # 分配颜色
    unique_labels = sorted(set(int(x) for x in labels))
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (230, 190, 255),
    ]
    color_map = {
        label: palette[i % len(palette)] for i, label in enumerate(unique_labels)
    }

    cv2.rectangle(
        canvas, (margin, margin), (canvas_w - margin, canvas_h - margin), (0, 0, 0), 2
    )
    cv2.putText(
        canvas,
        "PCA of Image Embeddings",
        (margin, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
    )

    # 把每张图画成一个点
    for x_val, y_val, name, label in zip(x_vals, y_vals, names, labels):
        px = int(margin + (x_val - x_min) / (x_max - x_min) * (canvas_w - 2 * margin))
        py = int(
            canvas_h
            - margin
            - (y_val - y_min) / (y_max - y_min) * (canvas_h - 2 * margin)
        )
        color = color_map[int(label)]
        cv2.circle(canvas, (px, py), 8, color, -1)
        cv2.putText(
            canvas,
            str(label),
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    # 画图例
    legend_y = 80
    for label in unique_labels:
        color = color_map[label]
        cv2.circle(canvas, (canvas_w - 220, legend_y), 8, color, -1)
        cv2.putText(
            canvas,
            f"cluster {label}",
            (canvas_w - 200, legend_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        legend_y += 30

    cv2.imwrite(str(output_file), canvas)
    print(f"PCA可视化已保存到 {output_file}")


if __name__ == "__main__":
    data_dir = Path("Data")
    img_feat_dir = data_dir / "image_feats"
    vis_dir = data_dir / "visualizations"
    cluster_file = Path("cluster_result_auto.txt")

    vis_dir.mkdir(parents=True, exist_ok=True)

    result = load_image_features(img_feat_dir)
    if result is not None:
        feats, names = result
        labels = load_cluster_labels(cluster_file, names)
        save_pca_visualization(feats, names, labels, vis_dir / "pca_clusters.png")
