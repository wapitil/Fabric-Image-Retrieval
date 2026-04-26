import math
import os
import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from dinov2_classifier_common import (
    build_feature_matrix_for_manifest,
    ensure_device,
    load_or_build_gallery_features,
    load_training_feature_cache,
    save_json,
    save_training_feature_cache,
    scan_class_folder_dataset,
)


# =========================
# 这里直接改参数
# =========================

dataset_root_dir = r""
gallery_dir = r"G:\images\fabric\images"
gallery_recursive = False
output_dir = r"G:\images\ZSY\dinov2_retrieval_result\classifier_pipeline"
feature_cache_dir = r"G:\images\ZSY\dinov2_retrieval_result\feature_cache"
local_pretrained_path = r"G:\images\ZSY\dinov2_vitb14_pretrain.pth"

random_seed = 20260426
val_ratio = 0.15
min_train_samples_per_class = 3
epochs = 120
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 128
label_smoothing = 0.02


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def split_train_val(manifest_df: pd.DataFrame, val_ratio_value: float):
    train_indices = []
    val_indices = []
    for _, class_df in manifest_df.groupby("class_name"):
        indices = class_df.index.tolist()
        random.shuffle(indices)
        if len(indices) < min_train_samples_per_class:
            train_indices.extend(indices)
            continue
        val_count = max(1, int(math.floor(len(indices) * val_ratio_value)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    train_df = manifest_df.loc[sorted(train_indices)].reset_index(drop=True)
    val_df = manifest_df.loc[sorted(val_indices)].reset_index(drop=True)
    return train_df, val_df


def build_centroids(feature_tensor: torch.Tensor, label_tensor: torch.Tensor, num_classes: int):
    centroids = []
    for class_index in range(num_classes):
        class_features = feature_tensor[label_tensor == class_index]
        centroid = class_features.mean(dim=0)
        centroid = F.normalize(centroid, dim=0)
        centroids.append(centroid)
    return torch.stack(centroids, dim=0)


def evaluate(model, features, labels):
    if len(features) == 0:
        return {"loss": None, "accuracy": None}
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        loss = F.cross_entropy(logits, labels).item()
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return {"loss": round(loss, 6), "accuracy": round(acc, 6)}


def run_training(config=None):
    config = config or {}
    resolved_dataset_root_dir = config.get("dataset_root_dir", dataset_root_dir)
    resolved_gallery_dir = config.get("gallery_dir", gallery_dir)
    resolved_gallery_recursive = config.get("gallery_recursive", gallery_recursive)
    resolved_output_dir = config.get("output_dir", output_dir)
    resolved_feature_cache_dir = config.get("feature_cache_dir", feature_cache_dir)
    resolved_local_pretrained_path = config.get("local_pretrained_path", local_pretrained_path)
    resolved_epochs = int(config.get("epochs", epochs))
    resolved_learning_rate = float(config.get("learning_rate", learning_rate))
    resolved_weight_decay = float(config.get("weight_decay", weight_decay))
    resolved_batch_size = int(config.get("batch_size", batch_size))
    resolved_label_smoothing = float(config.get("label_smoothing", label_smoothing))
    resolved_val_ratio = float(config.get("val_ratio", val_ratio))

    random.seed(random_seed)
    torch.manual_seed(random_seed)

    if not resolved_dataset_root_dir:
        raise RuntimeError("dataset_root_dir 为空，请先指定按类别分文件夹的训练数据集目录。")

    manifest_df = scan_class_folder_dataset(resolved_dataset_root_dir)

    class_counts = manifest_df["class_name"].value_counts()
    if (class_counts < 2).any():
        small_classes = class_counts[class_counts < 2].to_dict()
        raise RuntimeError(f"以下类别样本太少，至少需要 2 张: {small_classes}")

    device = ensure_device()
    gallery_bundle = load_or_build_gallery_features(
        gallery_dir=resolved_gallery_dir,
        gallery_recursive=resolved_gallery_recursive,
        output_dir=resolved_output_dir,
        feature_cache_dir=resolved_feature_cache_dir,
        local_pretrained_path=resolved_local_pretrained_path,
        device=device,
    )
    training_feature_cache = load_training_feature_cache(resolved_output_dir)
    feature_tensor = build_feature_matrix_for_manifest(
        manifest_df=manifest_df,
        gallery_index_map=gallery_bundle["gallery_index_map"],
        gallery_feature_matrix=gallery_bundle["gallery_feature_matrix"],
        model=gallery_bundle["model"],
        device=device,
        training_feature_cache=training_feature_cache,
    )
    save_training_feature_cache(resolved_output_dir, training_feature_cache)

    class_names = sorted(manifest_df["class_name"].unique().tolist())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    label_tensor = torch.tensor([class_to_index[name] for name in manifest_df["class_name"].tolist()], dtype=torch.long)

    train_df, val_df = split_train_val(manifest_df, resolved_val_ratio)
    train_indices = torch.tensor(train_df.index.tolist(), dtype=torch.long)
    val_indices = torch.tensor(val_df.index.tolist(), dtype=torch.long) if not val_df.empty else torch.tensor([], dtype=torch.long)

    train_features = feature_tensor[train_indices].to(device)
    train_labels = label_tensor[train_indices].to(device)
    val_features = feature_tensor[val_indices].to(device) if len(val_indices) > 0 else torch.empty((0, feature_tensor.shape[1]), device=device)
    val_labels = label_tensor[val_indices].to(device) if len(val_indices) > 0 else torch.empty((0,), dtype=torch.long, device=device)

    model = LinearHead(in_dim=feature_tensor.shape[1], num_classes=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=resolved_learning_rate, weight_decay=resolved_weight_decay)

    best_state = None
    best_val_acc = -1.0
    history_rows = []
    num_batches = max(1, math.ceil(len(train_features) / resolved_batch_size))

    for epoch in range(1, resolved_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_features), device=device)
        epoch_loss = 0.0

        for batch_index in range(num_batches):
            batch_perm = permutation[batch_index * resolved_batch_size:(batch_index + 1) * resolved_batch_size]
            batch_features = train_features[batch_perm]
            batch_labels = train_labels[batch_perm]

            logits = model(batch_features)
            loss = F.cross_entropy(logits, batch_labels, label_smoothing=resolved_label_smoothing)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_perm)

        train_metrics = evaluate(model, train_features, train_labels)
        val_metrics = evaluate(model, val_features, val_labels)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

        current_val_acc = val_metrics["accuracy"] if val_metrics["accuracy"] is not None else train_metrics["accuracy"]
        if current_val_acc is not None and current_val_acc >= best_val_acc:
            best_val_acc = current_val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)

    centroid_tensor = build_centroids(feature_tensor, label_tensor, len(class_names))

    manifest_path = os.path.join(resolved_output_dir, "folder_dataset_manifest.csv")
    artifact_path = os.path.join(resolved_output_dir, "dinov2_linear_classifier.pt")
    history_path = os.path.join(resolved_output_dir, "training_history.csv")
    summary_path = os.path.join(resolved_output_dir, "training_summary.json")
    os.makedirs(resolved_output_dir, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    torch.save(
        {
            "state_dict": best_state,
            "class_names": class_names,
            "class_to_index": class_to_index,
            "input_dim": int(feature_tensor.shape[1]),
            "train_count": int(len(train_df)),
            "val_count": int(len(val_df)),
            "centroids": centroid_tensor.cpu(),
            "config": {
                "epochs": resolved_epochs,
                "learning_rate": resolved_learning_rate,
                "weight_decay": resolved_weight_decay,
                "batch_size": resolved_batch_size,
                "label_smoothing": resolved_label_smoothing,
                "val_ratio": resolved_val_ratio,
            },
        },
        artifact_path,
    )

    pd.DataFrame(history_rows).to_csv(history_path, index=False, encoding="utf-8-sig")
    save_json(
        summary_path,
        {
            "dataset_root_dir": os.path.abspath(resolved_dataset_root_dir),
            "manifest_path": manifest_path,
            "artifact_path": artifact_path,
            "history_path": history_path,
            "class_count": len(class_names),
            "sample_count": int(len(manifest_df)),
            "train_count": int(len(train_df)),
            "val_count": int(len(val_df)),
            "best_val_accuracy": best_val_acc,
            "class_distribution": manifest_df["class_name"].value_counts().sort_index().to_dict(),
        },
    )

    result = {
        "manifest_path": manifest_path,
        "artifact_path": artifact_path,
        "history_path": history_path,
        "summary_path": summary_path,
        "class_count": len(class_names),
        "sample_count": int(len(manifest_df)),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "best_val_accuracy": best_val_acc,
    }
    print("分类器训练完成")
    print("artifact:", artifact_path)
    print("history:", history_path)
    print("summary:", summary_path)
    return result


def main():
    run_training()


if __name__ == "__main__":
    main()
