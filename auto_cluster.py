from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import normalize


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
    feats = normalize(feats)
    return feats, names


def cluster_images(img_feat_dir):
    """对图片embedding进行自动聚类"""
    result = load_image_features(Path(img_feat_dir))
    if result is None:
        return

    feats, names = result
    print(f"\n开始自动聚类 {len(feats)} 张图片")

    cluster = AffinityPropagation(random_state=0).fit(feats)
    labels = cluster.labels_

    print(f"自动得到 {len(set(labels))} 个cluster")
    return names, labels


def save_cluster_sheets(names, labels, img_dir, out_dir, thumb_size=220, columns=4):
    """根据聚类结果输出每个cluster的图片缩略图sheet"""
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

            img_path = next(
                img_dir / f"{name}{ext}"
                for ext in exts
                if (img_dir / f"{name}{ext}").exists()
            )
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"图片读取失败: {img_path}")

            h, w = img.shape[:2]
            scale = min(thumb_size / w, thumb_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            thumb = np.full((thumb_size, thumb_size, 3), 255, dtype=np.uint8)
            offset_x = (thumb_size - new_w) // 2
            offset_y = (thumb_size - new_h) // 2
            thumb[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

            sheet[y0 : y0 + thumb_size, x0 : x0 + thumb_size] = thumb
            cv2.rectangle(
                sheet,
                (x0, y0),
                (x0 + thumb_size - 1, y0 + thumb_size - 1),
                (180, 180, 180),
                1,
            )
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


if __name__ == "__main__":
    data_dir = Path("Data")
    img_feat_dir = data_dir / "image_feats"
    img_dir = data_dir / "images"
    cluster_sheet_dir = Path("output/cluster_sheets_auto")

    result = cluster_images(img_feat_dir)
    if result is not None:
        names, labels = result
        save_cluster_sheets(names, labels, img_dir, cluster_sheet_dir)
