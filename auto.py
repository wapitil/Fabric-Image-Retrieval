# pyright: reportAttributeAccessIssue=false
import os
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import numpy as np
import torch


def mask_defects(image_path, label_path):
    """将具有缺陷的部分使用 mask 屏蔽，YOLO 标注格式 class_id center_x center_y width height"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图片读取失败: {image_path}")

    h, w = img.shape[:2]
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x, y, bw, bh = map(float, parts)
            cx, cy = int(x * w), int(y * h)
            bw, bh = int(bw * w), int(bh * h)
            x1 = max(cx - bw // 2, 0)
            y1 = max(cy - bh // 2, 0)
            x2 = min(cx + bw // 2, w - 1)
            y2 = min(cy + bh // 2, h - 1)
            img[y1:y2, x1:x2] = 0
    return img


def is_masked(patch, threshold=0.05):
    """判断patch内黑色像素比例，超过阈值则认为有mask"""
    black = np.all(patch == 0, axis=2)
    ratio = np.sum(black) / (patch.shape[0] * patch.shape[1])
    return ratio > threshold


def extract_feature(
    img,
    image_path,
    model,
    patch_dir,
    feat_dir,
    device,
    patch_size=768,
    stride=512,
    max_keep=4,
    batch_size=8,
):
    """切分图像patch，遇到含mask的块跳过，提取特征"""
    h, w = img.shape[:2]
    base = image_path.stem
    idx = 0
    valid_patches = []
    count = 0

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y : y + patch_size, x : x + patch_size]
            count += 1
            if is_masked(patch):
                continue
            valid_patches.append((idx, patch))
            idx += 1

    valid_patches = valid_patches[:max_keep]
    print(f"{base}: 切了{count}块，保留{len(valid_patches)}块")

    patch_tensors = []
    patch_indices = []
    for idx, patch in valid_patches:
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_rgb = cv2.resize(patch_rgb, (448, 448))
        img_tensor = torch.from_numpy(patch_rgb).float().permute(2, 0, 1) / 255.0
        patch_tensors.append(img_tensor) # 把patch 都转成tensor
        patch_indices.append(idx)

    if not patch_tensors:
        return

    for start in range(0, len(patch_tensors), batch_size):
        batch_tensors = patch_tensors[start : start + batch_size]
        batch_indices = patch_indices[start : start + batch_size]
        batch = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True) # 把tensor 拼成一个batch

        with torch.inference_mode():
            feats = model(batch).detach().cpu().numpy()

        for idx, feat in zip(batch_indices, feats):
            feat_path = Path(feat_dir) / f"{base}_{idx}.npy"
            np.save(str(feat_path), feat)


def aggregate_patch_features(feat_dir, out_dir):
    """聚合patch特征为image级embedding"""
    feat_dir = Path(feat_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    img2patch = {}
    for f in feat_dir.glob("*.npy"):
        img_name = "_".join(f.stem.split("_")[:-1])
        img2patch.setdefault(img_name, []).append(f)

    for img_name, patch_files in img2patch.items():
        feats = [np.load(str(pf)) for pf in patch_files]
        feats = np.stack(feats, axis=0)
        img_feat = feats.mean(axis=0)
        np.save(str(out_dir / f"{img_name}.npy"), img_feat)
        print(f"聚合 {img_name}: {len(patch_files)} patches → 1 image embedding")


def save_pca_visualization(feats, names, labels, output_file="pca_visualization.png"):
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

    # if x_max - x_min < 1e-8:
    #     x_max += 1.0
    # if y_max - y_min < 1e-8:
    #     y_max += 1.0

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
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}

    cv2.rectangle(canvas, (margin, margin), (canvas_w - margin, canvas_h - margin), (0, 0, 0), 2)
    cv2.putText(canvas, "PCA of Image Embeddings", (margin, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    # 把每张图画成一个点
    for idx, (x_val, y_val, name, label) in enumerate(zip(x_vals, y_vals, names, labels)):
        px = int(margin + (x_val - x_min) / (x_max - x_min) * (canvas_w - 2 * margin))
        py = int(canvas_h - margin - (y_val - y_min) / (y_max - y_min) * (canvas_h - 2 * margin))
        color = color_map[int(label)]
        cv2.circle(canvas, (px, py), 8, color, -1)
        cv2.putText(canvas, str(label), (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 画图例
    legend_y = 80
    for label in unique_labels:
        color = color_map[label]
        cv2.circle(canvas, (canvas_w - 220, legend_y), 8, color, -1)
        cv2.putText(canvas, f"cluster {label}", (canvas_w - 200, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        legend_y += 30

    cv2.imwrite(str(output_file), canvas)
    print(f"PCA可视化已保存到 {output_file}")


def save_cluster_sheets(names, labels, img_dir, out_dir, thumb_size=220, columns=4):
    """根据聚类结果输出每个cluster的图片缩略图sheet"""
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    cluster_to_names = {}
    for name, label in zip(names, labels):
        cluster_to_names.setdefault(int(label), []).append(name)

    exts = [".jpg", ".png", ".jpeg", ".bmp"]

    for label, cluster_names in sorted(cluster_to_names.items()):
        total = len(cluster_names)
        rows = (total + columns - 1) // columns
        title_h = 60
        label_h = 30
        cell_h = thumb_size + label_h
        sheet_h = title_h + rows * cell_h + 20
        sheet_w = columns * thumb_size
        sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

        cv2.putText(
            sheet,
            f"cluster {label} ({total} images)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )

        for idx, name in enumerate(cluster_names):
            row = idx // columns
            col = idx % columns
            x0 = col * thumb_size
            y0 = title_h + row * cell_h

            img = None
            for ext in exts:
                img_path = img_dir / f"{name}{ext}"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    break

            if img is None:
                thumb = np.full((thumb_size, thumb_size, 3), 235, dtype=np.uint8)
                cv2.putText(thumb, "missing", (40, thumb_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                h, w = img.shape[:2]
                scale = min(thumb_size / max(w, 1), thumb_size / max(h, 1))
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                resized = cv2.resize(img, (new_w, new_h))
                thumb = np.full((thumb_size, thumb_size, 3), 255, dtype=np.uint8)
                offset_x = (thumb_size - new_w) // 2
                offset_y = (thumb_size - new_h) // 2
                thumb[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

            sheet[y0 : y0 + thumb_size, x0 : x0 + thumb_size] = thumb
            cv2.rectangle(sheet, (x0, y0), (x0 + thumb_size - 1, y0 + thumb_size - 1), (180, 180, 180), 1)
            cv2.putText(
                sheet,
                name[:24],
                (x0 + 8, y0 + thumb_size + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
            )

        out_path = out_dir / f"cluster_{label:02d}.jpg"
        cv2.imwrite(str(out_path), sheet)
        print(f"cluster {label} 缩略图sheet已保存到 {out_path}")


def cluster_images(img_feat_dir, n_clusters=10, output_file="cluster_result.txt"):
    """对图片embedding进行KMeans聚类"""
    img_feat_dir = Path(img_feat_dir)
    feats = []
    names = []

    for f in sorted(img_feat_dir.glob("*.npy")):
        feats.append(np.load(str(f)))
        names.append(f.stem)

    if len(feats) == 0:
        print("没有找到image embedding文件")
        return

    feats = np.stack(feats, axis=0)
    print(f"\n开始聚类 {len(feats)} 张图片，聚类数 {n_clusters}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(feats)

    with open(output_file, "w") as f:
        for name, label in zip(names, kmeans.labels_):
            f.write(f"{name}\t{label}\n")
            print(f"{name}: cluster {label}")
    print(f"\n聚类结果已保存到 {output_file}")
    return feats, names, kmeans.labels_


if __name__ == "__main__":
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=False)
    state_dict = torch.load("dinov2_vitb14_pretrain.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"使用设备: {device}")

    # 设置目录
    data_dir = Path("Data")
    img_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    patch_dir = data_dir / "patches"
    feat_dir = data_dir / "feats"
    img_feat_dir = data_dir / "image_feats"
    vis_dir = data_dir / "visualizations"
    cluster_sheet_dir = data_dir / "cluster_sheets"

    patch_dir.mkdir(exist_ok=True)
    feat_dir.mkdir(exist_ok=True)
    img_feat_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    cluster_sheet_dir.mkdir(exist_ok=True)

    # # 步骤1：mask缺陷 + 提取patch特征
    # print("=== 步骤1: mask缺陷 + 提取patch特征 ===")
    # for img_path in img_dir.iterdir():
    #     if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg", ".bmp"]:
    #         continue
    #     label_path = label_dir / f"{img_path.stem}.txt"
    #     if not label_path.exists():
    #         img = cv2.imread(str(img_path))
    #         mask_img = img
    #     else:
    #         mask_img = mask_defects(str(img_path), str(label_path))
    #     extract_feature(mask_img, img_path, model, patch_dir, feat_dir, device)

    # 步骤2：聚合patch特征为image级embedding
    print("\n=== 步骤2: 聚合patch特征为image级embedding ===")
    aggregate_patch_features(feat_dir, img_feat_dir)

    # 步骤3：聚类
    print("\n=== 步骤3: 对图片进行KMeans聚类 ===")
    result = cluster_images(img_feat_dir, n_clusters=10)

    if result is not None:
        feats, names, labels = result

        print("\n=== 步骤4: PCA可视化 ===")
        save_pca_visualization(feats, names, labels, vis_dir / "pca_clusters.png")

        print("\n=== 步骤5: 输出每个cluster的缩略图sheet ===")
        save_cluster_sheets(names, labels, img_dir, cluster_sheet_dir)
