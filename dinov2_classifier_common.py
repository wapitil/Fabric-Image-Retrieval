import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

import dinov2_patch_retrieval_fast as retrieval_core


def normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def dedupe_keep_order(paths: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for path in paths:
        normalized = normalize_path(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(os.path.abspath(path))
    return deduped


def get_file_signature(path: str) -> str:
    normalized = normalize_path(path)
    if not os.path.isfile(path):
        return f"{normalized}|missing"
    stat = os.stat(path)
    return f"{normalized}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}"


def scan_class_folder_dataset(dataset_root_dir: str) -> pd.DataFrame:
    dataset_root_dir = os.path.abspath(dataset_root_dir)
    if not os.path.isdir(dataset_root_dir):
        raise FileNotFoundError(f"训练数据集目录不存在: {dataset_root_dir}")

    rows = []
    for class_name in sorted(os.listdir(dataset_root_dir)):
        class_dir = os.path.join(dataset_root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        image_paths = retrieval_core.get_image_list(class_dir, recursive=False)
        for image_path in image_paths:
            rows.append(
                {
                    "class_name": class_name,
                    "image_path": os.path.abspath(image_path),
                    "label_source": "dataset_folder",
                    "verification_level": "trusted",
                }
            )

    manifest_df = pd.DataFrame(rows)
    if manifest_df.empty:
        raise RuntimeError("训练数据集目录里没有找到任何分类图片。")
    return manifest_df.sort_values(by=["class_name", "image_path"]).reset_index(drop=True)


def ensure_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_retrieval_core(
    gallery_dir: str,
    gallery_recursive: bool,
    output_dir: str,
    feature_cache_dir: str,
    local_pretrained_path: str,
):
    retrieval_core.gallery_dir = gallery_dir
    retrieval_core.gallery_recursive = gallery_recursive
    retrieval_core.output_dir = output_dir
    retrieval_core.feature_cache_dir = feature_cache_dir
    retrieval_core.local_pretrained_path = local_pretrained_path


def load_or_build_gallery_features(
    gallery_dir: str,
    gallery_recursive: bool,
    output_dir: str,
    feature_cache_dir: str,
    local_pretrained_path: str,
    device: torch.device,
) -> Dict[str, object]:
    configure_retrieval_core(
        gallery_dir=gallery_dir,
        gallery_recursive=gallery_recursive,
        output_dir=output_dir,
        feature_cache_dir=feature_cache_dir,
        local_pretrained_path=local_pretrained_path,
    )

    gallery_image_list = retrieval_core.get_image_list(gallery_dir, recursive=gallery_recursive)
    if not gallery_image_list:
        raise RuntimeError(f"图库为空: {gallery_dir}")

    retrieval_core.reference_image_path = gallery_image_list[0]
    model = retrieval_core.get_or_load_model(device)
    feature_bundle = retrieval_core.build_or_reuse_gallery_features(
        gallery_image_list=gallery_image_list,
        model=model,
        device=device,
        progress_desc="分类图库建库",
    )
    gallery_feature_matrix = feature_bundle["image_feature_matrix"].float()
    gallery_patch_feature_tensor = feature_bundle["patch_feature_tensor"]
    gallery_patch_feature_tensor = gallery_patch_feature_tensor.float() if gallery_patch_feature_tensor is not None else None

    gallery_index_map = {normalize_path(path): index for index, path in enumerate(gallery_image_list)}
    return {
        "gallery_image_list": gallery_image_list,
        "gallery_index_map": gallery_index_map,
        "gallery_feature_matrix": gallery_feature_matrix,
        "gallery_patch_feature_tensor": gallery_patch_feature_tensor,
        "model": model,
    }


def get_feature_for_image_path(
    image_path: str,
    gallery_index_map: Dict[str, int],
    gallery_feature_matrix: torch.Tensor,
    model,
    device: torch.device,
    training_feature_cache: Optional[Dict[str, Dict[str, object]]] = None,
) -> torch.Tensor:
    image_signature = get_file_signature(image_path)
    if training_feature_cache is not None:
        cached_entry = training_feature_cache.get(normalize_path(image_path))
        if cached_entry and cached_entry.get("signature") == image_signature:
            cached_feature = cached_entry.get("feature")
            if isinstance(cached_feature, torch.Tensor):
                return cached_feature.float()

    image_index = gallery_index_map.get(normalize_path(image_path))
    if image_index is not None:
        feature = gallery_feature_matrix[image_index].float().cpu()
    else:
        feature_bundle = retrieval_core.extract_single_feature(image_path, model, device)
        feature = feature_bundle["image_feature"].float().cpu()

    if training_feature_cache is not None:
        training_feature_cache[normalize_path(image_path)] = {
            "signature": image_signature,
            "feature": feature.clone(),
        }
    return feature


def build_feature_matrix_for_manifest(
    manifest_df: pd.DataFrame,
    gallery_index_map: Dict[str, int],
    gallery_feature_matrix: torch.Tensor,
    model,
    device: torch.device,
    training_feature_cache: Optional[Dict[str, Dict[str, object]]] = None,
) -> torch.Tensor:
    feature_rows = []
    for row in tqdm(manifest_df.to_dict("records"), total=len(manifest_df), desc="提取训练特征"):
        feature_rows.append(
            get_feature_for_image_path(
                image_path=row["image_path"],
                gallery_index_map=gallery_index_map,
                gallery_feature_matrix=gallery_feature_matrix,
                model=model,
                device=device,
                training_feature_cache=training_feature_cache,
            )
        )
    return torch.stack(feature_rows, dim=0)


def get_training_feature_cache_path(output_dir: str) -> str:
    return os.path.join(os.path.abspath(output_dir), "training_feature_cache.pt")


def load_training_feature_cache(output_dir: str) -> Dict[str, Dict[str, object]]:
    cache_path = get_training_feature_cache_path(output_dir)
    if not os.path.isfile(cache_path):
        return {}
    payload = torch.load(cache_path, map_location="cpu")
    raw_entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
    cache: Dict[str, Dict[str, object]] = {}
    for key, value in raw_entries.items():
        if not isinstance(value, dict):
            continue
        feature = value.get("feature")
        if not isinstance(feature, torch.Tensor):
            continue
        cache[str(key)] = {
            "signature": str(value.get("signature", "")),
            "feature": feature.float().cpu(),
        }
    return cache


def save_training_feature_cache(output_dir: str, cache: Dict[str, Dict[str, object]]):
    cache_path = get_training_feature_cache_path(output_dir)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    sanitized_entries = {}
    for key, value in cache.items():
        feature = value.get("feature")
        if not isinstance(feature, torch.Tensor):
            continue
        sanitized_entries[str(key)] = {
            "signature": str(value.get("signature", "")),
            "feature": feature.float().cpu(),
        }
    torch.save({"entries": sanitized_entries}, cache_path)


def cache_training_feature_for_path(
    output_dir: str,
    target_image_path: str,
    feature: torch.Tensor,
):
    if feature is None or not os.path.isfile(target_image_path):
        return
    cache = load_training_feature_cache(output_dir)
    cache[normalize_path(target_image_path)] = {
        "signature": get_file_signature(target_image_path),
        "feature": feature.float().cpu(),
    }
    save_training_feature_cache(output_dir, cache)


def save_json(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str, default_value: Dict) -> Dict:
    if not os.path.isfile(path):
        return default_value
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def sigmoid_probability_gap(top1_prob: torch.Tensor, top2_prob: torch.Tensor) -> torch.Tensor:
    return top1_prob - top2_prob
