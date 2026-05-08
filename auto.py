# pyright: reportAttributeAccessIssue=false
import random
from pathlib import Path

import cv2
import numpy as np
import torch


def mask_defects(image_path, label_path):
    """将具有缺陷的部分使用 mask 屏蔽，YOLO 标注格式 class_id center_x center_y width height"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"图片读取失败: {image_path}")

    h, w = img.shape[:2]
    with Path(label_path).open("r") as f:
        for line in f:
            parts = line.strip().split()
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
    ratio = black.mean()
    return ratio > threshold


def get_batch_features(batch_patches, model, device):
    model_size = 448
    patch_tensors = []
    for patch in batch_patches:
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_rgb = cv2.resize(patch_rgb, (model_size, model_size))
        img_tensor = torch.from_numpy(patch_rgb).float().permute(2, 0, 1) / 255.0
        patch_tensors.append(img_tensor)  # 把patch 都转成tensor

    batch = torch.stack(patch_tensors, dim=0).to(
        device, non_blocking=True
    )  # 把tensor 拼成一个batch

    with torch.inference_mode():
        feats = model(batch).cpu().numpy()

    return feats


def extract_feature(img, image_path, model, img_feat_dir, device):
    """切分图像patch，遇到含mask的块跳过，提取特征"""
    patch_size = 768
    stride = 512
    max_keep = 4
    h, w = img.shape[:2]
    base = image_path.stem
    positions = [
        (y, x)
        for y in range(0, h - patch_size + 1, stride)
        for x in range(0, w - patch_size + 1, stride)
    ]
    random.shuffle(positions)

    patches = []
    for idx, (y, x) in enumerate(positions):
        patch = img[y : y + patch_size, x : x + patch_size]
        if is_masked(patch):
            continue

        patches.append(patch)

        if len(patches) == max_keep:
            break

    if patches:
        # 得到 4 个 patch feature
        feats = get_batch_features(patches, model, device)
        img_feat = feats.mean(axis=0)
        np.save(str(img_feat_dir / f"{base}.npy"), img_feat)

    # print(f"{base}: 随机保留{len(patches)}块")


def load_dinov2_model():
    model_name = "dinov2_vitb14"
    weights_path = Path("dinov2_vitb14_pretrain.pth")
    dinov2_dir = Path("third_party") / "dinov2"

    model = torch.hub.load(
        str(dinov2_dir), model_name, source="local", pretrained=False
    )
    state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2_model()
    model.eval()
    model.to(device)
    print(f"使用设备: {device}")

    # 设置目录
    data_dir = Path("Data")
    img_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    img_feat_dir = data_dir / "image_feats"

    img_feat_dir.mkdir(parents=True, exist_ok=True)

    # 步骤1：mask缺陷 + 提取图片特征
    print("=== 步骤1: mask缺陷 + 提取图片特征 ===")
    for img_path in img_dir.iterdir():
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"图片读取失败: {img_path}")
            mask_img = img
        else:
            mask_img = mask_defects(img_path, label_path)
        extract_feature(mask_img, img_path, model, img_feat_dir, device)
