import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from dinov2_classifier_common import (
    ensure_device,
    load_or_build_gallery_features,
    normalize_path,
    read_csv_if_exists,
    save_json,
)


# =========================
# 这里直接改参数
# =========================

classifier_artifact_path = r"G:\images\ZSY\dinov2_retrieval_result\classifier_pipeline\dinov2_linear_classifier.pt"
dataset_manifest_csv_path = r"G:\images\ZSY\dinov2_retrieval_result\classifier_pipeline\folder_dataset_manifest.csv"
gallery_dir = r"G:\images\fabric\images"
gallery_recursive = False
output_dir = r"G:\images\ZSY\dinov2_retrieval_result\classifier_pipeline"
feature_cache_dir = r"G:\images\ZSY\dinov2_retrieval_result\feature_cache"
local_pretrained_path = r"G:\images\ZSY\dinov2_vitb14_pretrain.pth"

high_confidence_probability_threshold = 0.92
review_probability_threshold = 0.65
unknown_probability_threshold = 0.45
min_top1_top2_margin = 0.18


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def classify_tier(top1_prob: float, margin: float):
    if top1_prob < unknown_probability_threshold:
        return "unknown"
    if top1_prob >= high_confidence_probability_threshold and margin >= min_top1_top2_margin:
        return "high_confidence_candidate"
    if top1_prob >= review_probability_threshold:
        return "review"
    return "unknown"


def run_prediction(config=None):
    config = config or {}
    resolved_classifier_artifact_path = config.get("classifier_artifact_path", classifier_artifact_path)
    resolved_dataset_manifest_csv_path = config.get("dataset_manifest_csv_path", dataset_manifest_csv_path)
    resolved_gallery_dir = config.get("gallery_dir", gallery_dir)
    resolved_gallery_recursive = config.get("gallery_recursive", gallery_recursive)
    resolved_output_dir = config.get("output_dir", output_dir)
    resolved_feature_cache_dir = config.get("feature_cache_dir", feature_cache_dir)
    resolved_local_pretrained_path = config.get("local_pretrained_path", local_pretrained_path)
    resolved_high_confidence_probability_threshold = config.get("high_confidence_probability_threshold", high_confidence_probability_threshold)
    resolved_review_probability_threshold = config.get("review_probability_threshold", review_probability_threshold)
    resolved_unknown_probability_threshold = config.get("unknown_probability_threshold", unknown_probability_threshold)
    resolved_min_top1_top2_margin = config.get("min_top1_top2_margin", min_top1_top2_margin)

    def local_classify_tier(top1_prob: float, margin: float):
        if top1_prob < resolved_unknown_probability_threshold:
            return "unknown"
        if top1_prob >= resolved_high_confidence_probability_threshold and margin >= resolved_min_top1_top2_margin:
            return "high_confidence_candidate"
        if top1_prob >= resolved_review_probability_threshold:
            return "review"
        return "unknown"

    if not os.path.isfile(resolved_classifier_artifact_path):
        raise FileNotFoundError(f"分类器权重不存在: {resolved_classifier_artifact_path}")

    artifact = torch.load(resolved_classifier_artifact_path, map_location="cpu")
    class_names = artifact["class_names"]
    centroids = artifact["centroids"].float()

    device = ensure_device()
    gallery_bundle = load_or_build_gallery_features(
        gallery_dir=resolved_gallery_dir,
        gallery_recursive=resolved_gallery_recursive,
        output_dir=resolved_output_dir,
        feature_cache_dir=resolved_feature_cache_dir,
        local_pretrained_path=resolved_local_pretrained_path,
        device=device,
    )
    gallery_paths = gallery_bundle["gallery_image_list"]
    gallery_features = gallery_bundle["gallery_feature_matrix"].float()

    model = LinearHead(in_dim=artifact["input_dim"], num_classes=len(class_names))
    model.load_state_dict(artifact["state_dict"])
    model.to(device)
    model.eval()

    verified_df = read_csv_if_exists(resolved_dataset_manifest_csv_path)
    labeled_paths = {normalize_path(path) for path in verified_df["image_path"].tolist()} if not verified_df.empty else set()

    with torch.inference_mode():
        logits = model(gallery_features.to(device))
        probs = torch.softmax(logits, dim=1).cpu()
        centroid_scores = torch.matmul(gallery_features.cpu(), centroids.T)

    top2_probs, top2_indices = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
    prediction_rows = []
    for index, image_path in enumerate(gallery_paths):
        top1_index = int(top2_indices[index, 0].item())
        top1_prob = float(top2_probs[index, 0].item())
        if top2_probs.shape[1] > 1:
            top2_index = int(top2_indices[index, 1].item())
            top2_prob = float(top2_probs[index, 1].item())
        else:
            top2_index = top1_index
            top2_prob = 0.0
        margin = top1_prob - top2_prob
        tier = local_classify_tier(top1_prob=top1_prob, margin=margin)
        if normalize_path(image_path) in labeled_paths:
            tier = "labeled_skip"

        prediction_rows.append(
            {
                "image_path": image_path,
                "predicted_class": class_names[top1_index],
                "top1_probability": round(top1_prob, 6),
                "second_class": class_names[top2_index],
                "top2_probability": round(top2_prob, 6),
                "probability_margin": round(margin, 6),
                "centroid_similarity_top1": round(float(centroid_scores[index, top1_index].item()), 6),
                "tier": tier,
            }
        )

    prediction_df = pd.DataFrame(prediction_rows).sort_values(
        by=["tier", "top1_probability", "probability_margin"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    os.makedirs(resolved_output_dir, exist_ok=True)
    all_path = os.path.join(resolved_output_dir, "prediction_all.csv")
    high_confidence_path = os.path.join(resolved_output_dir, "prediction_high_confidence_candidate.csv")
    review_path = os.path.join(resolved_output_dir, "prediction_review_queue.csv")
    unknown_path = os.path.join(resolved_output_dir, "prediction_unknown.csv")
    summary_path = os.path.join(resolved_output_dir, "prediction_summary.json")

    prediction_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    prediction_df[prediction_df["tier"] == "high_confidence_candidate"].to_csv(high_confidence_path, index=False, encoding="utf-8-sig")
    prediction_df[prediction_df["tier"] == "review"].to_csv(review_path, index=False, encoding="utf-8-sig")
    prediction_df[prediction_df["tier"] == "unknown"].to_csv(unknown_path, index=False, encoding="utf-8-sig")

    save_json(
        summary_path,
        {
            "gallery_count": int(len(gallery_paths)),
            "labeled_skip_count": int((prediction_df["tier"] == "labeled_skip").sum()),
            "high_confidence_candidate_count": int((prediction_df["tier"] == "high_confidence_candidate").sum()),
            "review_count": int((prediction_df["tier"] == "review").sum()),
            "unknown_count": int((prediction_df["tier"] == "unknown").sum()),
            "thresholds": {
                "high_confidence_probability_threshold": resolved_high_confidence_probability_threshold,
                "review_probability_threshold": resolved_review_probability_threshold,
                "unknown_probability_threshold": resolved_unknown_probability_threshold,
                "min_top1_top2_margin": resolved_min_top1_top2_margin,
            },
        },
    )

    print("全图库预测完成")
    print("all:", all_path)
    print("high_confidence:", high_confidence_path)
    print("review:", review_path)
    print("unknown:", unknown_path)
    print("summary:", summary_path)
    return {
        "all_path": all_path,
        "high_confidence_path": high_confidence_path,
        "review_path": review_path,
        "unknown_path": unknown_path,
        "summary_path": summary_path,
        "high_confidence_candidate_count": int((prediction_df["tier"] == "high_confidence_candidate").sum()),
        "review_count": int((prediction_df["tier"] == "review").sum()),
        "unknown_count": int((prediction_df["tier"] == "unknown").sum()),
        "gallery_count": int(len(gallery_paths)),
    }


def main():
    run_prediction()


if __name__ == "__main__":
    main()
