import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# =========================
# 这里直接改参数
# =========================

reference_image_path = r""
gallery_dir = r"G:\images\dataset0422\plain"
gallery_recursive = False
output_dir = r"G:\images\ZSY\dinov2_retrieval_result"
use_gallery_feature_cache = True
feature_cache_dir = r"G:\images\ZSY\dinov2_retrieval_result\feature_cache"
dinov2_repo_or_dir = "facebookresearch/dinov2"
dinov2_source = "github"
dinov2_model_name = "dinov2_vitb14"
local_pretrained_path = r"G:\images\ZSY\dinov2_vitb14_pretrain.pth"
patch_count_per_image = 6
patch_scale = 0.6
retrieval_topk = 10
gallery_batch_size = 16
input_size = 518
use_amp_inference = True
global_similarity_weight = 0.4
local_similarity_weight = 0.6
feedback_alpha = 1.0
feedback_beta = 0.55
random_seed = 20260425
image_suffixes = (".bmp", ".jpg", ".jpeg", ".png")

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

_MODEL_CACHE = {}
_GALLERY_RUNTIME_CACHE = {}


def get_image_list(folder, recursive=False):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"图片目录不存在: {folder}")

    image_list = []
    if recursive:
        for root, _, file_names in os.walk(folder):
            for file_name in file_names:
                if file_name.lower().endswith(image_suffixes):
                    image_list.append(os.path.join(root, file_name))
    else:
        for file_name in os.listdir(folder):
            if file_name.lower().endswith(image_suffixes):
                image_list.append(os.path.join(folder, file_name))
    image_list.sort()
    return image_list


def validate_image_file(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    if not image_path.lower().endswith(image_suffixes):
        raise ValueError(f"不是支持的图片格式: {image_path}")


def get_file_info(image_path):
    stat = os.stat(image_path)
    return {
        "path": os.path.abspath(image_path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def get_image_signature(image_path):
    stat = os.stat(image_path)
    return {
        "path": os.path.abspath(image_path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def get_gallery_signature(gallery_image_list):
    return {
        "gallery_dir": os.path.abspath(gallery_dir),
        "gallery_recursive": gallery_recursive,
        "dinov2_repo_or_dir": dinov2_repo_or_dir,
        "dinov2_source": dinov2_source,
        "dinov2_model_name": dinov2_model_name,
        "local_pretrained_path": os.path.abspath(local_pretrained_path) if local_pretrained_path else "",
        "patch_count_per_image": patch_count_per_image,
        "patch_scale": patch_scale,
        "input_size": input_size,
        "gallery_batch_size": gallery_batch_size,
        "image_suffixes": image_suffixes,
        "images": [get_file_info(path) for path in gallery_image_list],
    }


def get_gallery_cache_path(gallery_signature):
    signature_text = json.dumps(gallery_signature, ensure_ascii=False, sort_keys=True)
    signature_hash = hashlib.md5(signature_text.encode("utf-8")).hexdigest()
    return os.path.join(feature_cache_dir, f"gallery_features_fast_{signature_hash}.pt")


def get_gallery_signature_hash(gallery_signature):
    signature_text = json.dumps(gallery_signature, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(signature_text.encode("utf-8")).hexdigest()


def get_gallery_cache_namespace():
    return {
        "gallery_dir": os.path.abspath(gallery_dir),
        "gallery_recursive": gallery_recursive,
        "dinov2_repo_or_dir": dinov2_repo_or_dir,
        "dinov2_source": dinov2_source,
        "dinov2_model_name": dinov2_model_name,
        "local_pretrained_path": os.path.abspath(local_pretrained_path) if local_pretrained_path else "",
        "patch_count_per_image": patch_count_per_image,
        "patch_scale": patch_scale,
        "input_size": input_size,
        "image_suffixes": image_suffixes,
    }


def get_gallery_item_cache_path():
    namespace = get_gallery_cache_namespace()
    namespace_text = json.dumps(namespace, ensure_ascii=False, sort_keys=True)
    namespace_hash = hashlib.md5(namespace_text.encode("utf-8")).hexdigest()
    return os.path.join(feature_cache_dir, f"gallery_feature_items_{namespace_hash}.pt")


def load_gallery_feature_cache(gallery_signature):
    if not use_gallery_feature_cache:
        return None

    cache_path = get_gallery_cache_path(gallery_signature)
    if not os.path.isfile(cache_path):
        return None

    cache = torch.load(cache_path, map_location="cpu")
    if cache.get("signature") != gallery_signature:
        return None

    print(f"读取图库特征缓存: {cache_path}")
    if "image_feature_matrix" in cache and "patch_feature_tensor" in cache:
        return {
            "image_feature_matrix": cache["image_feature_matrix"],
            "patch_feature_tensor": cache["patch_feature_tensor"],
        }

    legacy_feature_matrix = cache.get("feature_matrix")
    if legacy_feature_matrix is None:
        return None
    return {
        "image_feature_matrix": legacy_feature_matrix,
        "patch_feature_tensor": None,
    }


def save_gallery_feature_cache(gallery_signature, image_feature_matrix, patch_feature_tensor):
    if not use_gallery_feature_cache:
        return

    os.makedirs(feature_cache_dir, exist_ok=True)
    cache_path = get_gallery_cache_path(gallery_signature)
    torch.save(
        {
            "signature": gallery_signature,
            "image_feature_matrix": image_feature_matrix.cpu(),
            "patch_feature_tensor": patch_feature_tensor.cpu(),
        },
        cache_path,
    )
    print(f"图库特征已缓存: {cache_path}")


def load_gallery_item_feature_cache():
    if not use_gallery_feature_cache:
        return {}

    cache_path = get_gallery_item_cache_path()
    if not os.path.isfile(cache_path):
        return {}

    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("namespace") != get_gallery_cache_namespace():
        return {}

    raw_entries = payload.get("entries", {})
    if not isinstance(raw_entries, dict):
        return {}

    entries = {}
    for image_path, entry in raw_entries.items():
        if not isinstance(entry, dict):
            continue
        image_feature = entry.get("image_feature")
        patch_features = entry.get("patch_features")
        if not isinstance(image_feature, torch.Tensor) or not isinstance(patch_features, torch.Tensor):
            continue
        entries[str(image_path)] = {
            "signature": entry.get("signature"),
            "image_feature": image_feature.float().cpu(),
            "patch_features": patch_features.float().cpu(),
        }
    if entries:
        print(f"读取图库增量特征缓存: {cache_path} | entries={len(entries)}")
    return entries


def save_gallery_item_feature_cache(entries):
    if not use_gallery_feature_cache:
        return

    os.makedirs(feature_cache_dir, exist_ok=True)
    cache_path = get_gallery_item_cache_path()
    sanitized_entries = {}
    for image_path, entry in entries.items():
        image_feature = entry.get("image_feature")
        patch_features = entry.get("patch_features")
        if not isinstance(image_feature, torch.Tensor) or not isinstance(patch_features, torch.Tensor):
            continue
        sanitized_entries[str(image_path)] = {
            "signature": entry.get("signature"),
            "image_feature": image_feature.float().cpu(),
            "patch_features": patch_features.float().cpu(),
        }

    torch.save(
        {
            "namespace": get_gallery_cache_namespace(),
            "entries": sanitized_entries,
        },
        cache_path,
    )
    print(f"图库增量特征缓存已更新: {cache_path} | entries={len(sanitized_entries)}")


def build_or_reuse_gallery_features(gallery_image_list, model, device, progress_desc):
    gallery_signature = get_gallery_signature(gallery_image_list)
    cached_feature_bundle = load_gallery_feature_cache(gallery_signature)
    if cached_feature_bundle is not None and cached_feature_bundle.get("patch_feature_tensor") is not None:
        return {
            "gallery_signature": gallery_signature,
            "image_feature_matrix": cached_feature_bundle["image_feature_matrix"].float(),
            "patch_feature_tensor": cached_feature_bundle["patch_feature_tensor"].float(),
            "cache_mode": "exact",
        }

    item_cache_entries = load_gallery_item_feature_cache()
    ordered_image_features = [None] * len(gallery_image_list)
    ordered_patch_features = [None] * len(gallery_image_list)
    missing_indices = []
    reused_count = 0

    for index, image_path in enumerate(gallery_image_list):
        normalized_path = os.path.abspath(image_path)
        image_signature = get_image_signature(image_path)
        cached_entry = item_cache_entries.get(normalized_path)
        if cached_entry and cached_entry.get("signature") == image_signature:
            ordered_image_features[index] = cached_entry["image_feature"].float().cpu()
            ordered_patch_features[index] = cached_entry["patch_features"].float().cpu()
            reused_count += 1
        else:
            missing_indices.append(index)

    if missing_indices:
        print(
            f"图库增量缓存命中 {reused_count}/{len(gallery_image_list)}，"
            f"开始补提 {len(missing_indices)} 张图片特征..."
        )
        total_batches = (len(missing_indices) + gallery_batch_size - 1) // gallery_batch_size
        for batch_indices in tqdm(
            chunked(missing_indices, gallery_batch_size),
            total=total_batches,
            desc=progress_desc,
        ):
            batch_image_list = [gallery_image_list[index] for index in batch_indices]
            batch_feature_bundle = extract_feature_batch(batch_image_list, model, device)
            batch_image_features = batch_feature_bundle["image_features"].float().cpu()
            batch_patch_features = batch_feature_bundle["patch_features"].float().cpu()
            for offset, image_index in enumerate(batch_indices):
                image_path = os.path.abspath(gallery_image_list[image_index])
                image_feature = batch_image_features[offset]
                patch_features = batch_patch_features[offset]
                ordered_image_features[image_index] = image_feature
                ordered_patch_features[image_index] = patch_features
                item_cache_entries[image_path] = {
                    "signature": get_image_signature(image_path),
                    "image_feature": image_feature,
                    "patch_features": patch_features,
                }
    else:
        print(f"图库增量缓存全命中，直接复用 {reused_count} 张图片特征。")

    gallery_feature_matrix = torch.stack(ordered_image_features, dim=0).float()
    gallery_patch_feature_tensor = torch.stack(ordered_patch_features, dim=0).float()
    current_gallery_paths = {os.path.abspath(path) for path in gallery_image_list}
    item_cache_entries = {
        image_path: entry
        for image_path, entry in item_cache_entries.items()
        if image_path in current_gallery_paths
    }
    save_gallery_item_feature_cache(item_cache_entries)
    save_gallery_feature_cache(gallery_signature, gallery_feature_matrix, gallery_patch_feature_tensor)
    return {
        "gallery_signature": gallery_signature,
        "image_feature_matrix": gallery_feature_matrix,
        "patch_feature_tensor": gallery_patch_feature_tensor,
        "cache_mode": "incremental",
    }


def get_patch_list(image):
    image = image.convert("RGB")
    width, height = image.size
    short_side = min(width, height)
    patch_size = int(short_side * patch_scale)

    positions = [
        (0.5, 0.5),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 0.0),
    ]
    positions = positions[:patch_count_per_image]

    patch_list = []
    for x_ratio, y_ratio in positions:
        left = int((width - patch_size) * x_ratio)
        top = int((height - patch_size) * y_ratio)
        patch = image.crop((left, top, left + patch_size, top + patch_size))
        patch_list.append(patch)

    return patch_list


def preprocess_patch(patch):
    patch = patch.resize((input_size, input_size), Image.BILINEAR)
    patch_array = np.asarray(patch, dtype=np.float32) / 255.0
    patch_tensor = torch.from_numpy(patch_array).permute(2, 0, 1)
    patch_tensor = (patch_tensor - IMAGE_MEAN) / IMAGE_STD
    return patch_tensor


def build_patch_batch(image_path_list):
    batch_patches = []
    for image_path in image_path_list:
        image = Image.open(image_path).convert("RGB")
        patch_list = get_patch_list(image)
        batch_patches.extend(preprocess_patch(patch) for patch in patch_list)
    return torch.stack(batch_patches, dim=0)


def extract_feature_batch(image_path_list, model, device):
    patch_batch = build_patch_batch(image_path_list).to(device, non_blocking=True)
    use_amp = use_amp_inference and device.type == "cuda"

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            patch_features = model(patch_batch)
        patch_features = torch.nn.functional.normalize(patch_features.float(), dim=1)

    patch_features = patch_features.view(len(image_path_list), patch_count_per_image, -1)
    image_features = patch_features.mean(dim=1)
    image_features = torch.nn.functional.normalize(image_features, dim=1)
    return {
        "image_features": image_features.cpu(),
        "patch_features": patch_features.cpu(),
    }


def extract_single_feature(image_path, model, device):
    feature_batch = extract_feature_batch([image_path], model, device)
    return {
        "image_feature": feature_batch["image_features"][0],
        "patch_features": feature_batch["patch_features"][0],
    }


def load_local_checkpoint_if_needed(model, checkpoint_path):
    if not checkpoint_path:
        print("未设置本地权重路径，使用 torch.hub 默认权重。")
        return model

    if not os.path.isfile(checkpoint_path):
        print(f"未找到本地权重文件: {checkpoint_path}")
        print("将继续使用 torch.hub 默认权重。")
        return model

    print(f"检测到本地权重，开始加载: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "teacher" in checkpoint:
            checkpoint = checkpoint["teacher"]

    if not isinstance(checkpoint, dict):
        raise RuntimeError("本地权重文件格式不正确，无法解析为 state_dict。")

    cleaned_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key
        for prefix in ("module.", "backbone.", "teacher.", "student."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned_state_dict[new_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"本地权重加载完成 | missing_keys={len(missing_keys)} | unexpected_keys={len(unexpected_keys)}")
    return model


def should_use_local_pretrained(checkpoint_path):
    return bool(checkpoint_path) and os.path.isfile(checkpoint_path)


def chunked(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def get_valid_gallery_indices(gallery_paths, reference_path, hidden_paths=None):
    reference_abs_path = os.path.abspath(reference_path)
    hidden_abs_paths = {os.path.abspath(path) for path in (hidden_paths or set())}
    valid_indices = []
    for index, gallery_path in enumerate(gallery_paths):
        gallery_abs_path = os.path.abspath(gallery_path)
        if gallery_abs_path == reference_abs_path:
            continue
        if gallery_abs_path in hidden_abs_paths:
            continue
        valid_indices.append(index)
    return valid_indices


def get_model_cache_key():
    return (
        dinov2_repo_or_dir,
        dinov2_source,
        dinov2_model_name,
        os.path.abspath(local_pretrained_path) if local_pretrained_path else "",
    )


def get_or_load_model(device):
    cache_key = get_model_cache_key()
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    use_local_pretrained = should_use_local_pretrained(local_pretrained_path)
    model = torch.hub.load(
        dinov2_repo_or_dir,
        dinov2_model_name,
        source=dinov2_source,
        pretrained=not use_local_pretrained,
    )
    if use_local_pretrained:
        print("已检测到本地权重，跳过在线预训练权重下载。")
    model = load_local_checkpoint_if_needed(model, local_pretrained_path)
    model.eval()
    model.to(device)
    _MODEL_CACHE[cache_key] = model
    return model


def get_feedback_feature_map(gallery_paths, gallery_feature_matrix, feedback_labels):
    feedback_labels = feedback_labels or {}
    feature_map = {
        "accepted": [],
        "rejected": [],
    }
    if not feedback_labels:
        return feature_map

    path_to_index = {os.path.abspath(path): index for index, path in enumerate(gallery_paths)}
    for image_path, label in feedback_labels.items():
        normalized_label = str(label).strip().lower()
        if normalized_label not in ("accepted", "rejected"):
            continue
        image_index = path_to_index.get(os.path.abspath(image_path))
        if image_index is None:
            continue
        feature_map[normalized_label].append(gallery_feature_matrix[image_index])
    return feature_map


def build_feedback_query_feature(query_feature, feedback_feature_map):
    expanded_query = query_feature.clone().float()

    accepted_features = feedback_feature_map["accepted"]
    if accepted_features:
        accepted_center = torch.stack(accepted_features, dim=0).mean(dim=0)
        accepted_center = torch.nn.functional.normalize(accepted_center, dim=0)
        expanded_query = expanded_query + feedback_alpha * accepted_center

    rejected_features = feedback_feature_map["rejected"]
    if rejected_features:
        rejected_center = torch.stack(rejected_features, dim=0).mean(dim=0)
        rejected_center = torch.nn.functional.normalize(rejected_center, dim=0)
        expanded_query = expanded_query - feedback_beta * rejected_center

    expanded_query = torch.nn.functional.normalize(expanded_query, dim=0)
    return expanded_query


def compute_local_patch_similarity(query_patch_features, gallery_patch_feature_tensor):
    patch_similarity = torch.einsum("gpd,qd->gpq", gallery_patch_feature_tensor, query_patch_features)
    query_to_gallery = patch_similarity.max(dim=1).values.mean(dim=1)
    gallery_to_query = patch_similarity.max(dim=2).values.mean(dim=1)
    return 0.5 * (query_to_gallery + gallery_to_query)


def blend_similarity(global_similarity, local_similarity):
    return global_similarity_weight * global_similarity + local_similarity_weight * local_similarity


def apply_feedback_rerank(query_feature, gallery_feature_matrix, base_local_similarity, feedback_labels, gallery_paths):
    feedback_feature_map = get_feedback_feature_map(gallery_paths, gallery_feature_matrix, feedback_labels)
    feedback_query_feature = build_feedback_query_feature(query_feature, feedback_feature_map)
    rerank_global_similarity = torch.matmul(gallery_feature_matrix, feedback_query_feature)
    rerank_similarity = blend_similarity(rerank_global_similarity, base_local_similarity)
    return rerank_similarity, feedback_query_feature


def build_result_dataframe(
    reference_path,
    gallery_paths,
    base_similarity,
    rerank_similarity,
    ranked_indices,
    feedback_labels,
    base_global_similarity=None,
    base_local_similarity=None,
):
    feedback_labels = feedback_labels or {}
    result_rows = []
    for rank, gallery_index in enumerate(ranked_indices, start=1):
        gallery_path = gallery_paths[gallery_index]
        gallery_abs = os.path.abspath(gallery_path)
        result_rows.append(
            {
                "rank": rank,
                "reference_image_path": reference_path,
                "gallery_image_path": gallery_path,
                "global_score": round(float(base_global_similarity[gallery_index].item()), 6) if base_global_similarity is not None else None,
                "local_score": round(float(base_local_similarity[gallery_index].item()), 6) if base_local_similarity is not None else None,
                "similarity_score": round(float(base_similarity[gallery_index].item()), 6),
                "rerank_score": round(float(rerank_similarity[gallery_index].item()), 6),
                "feedback_label": feedback_labels.get(gallery_abs, "unlabeled"),
            }
        )
    return pd.DataFrame(result_rows)


def ensure_gallery_resources(device):
    validate_image_file(reference_image_path)
    gallery_image_list = get_image_list(gallery_dir, recursive=gallery_recursive)
    if len(gallery_image_list) == 0:
        raise RuntimeError("图库中没有可检索图片。")

    gallery_signature = get_gallery_signature(gallery_image_list)
    signature_hash = get_gallery_signature_hash(gallery_signature)
    runtime_cache = _GALLERY_RUNTIME_CACHE.get(signature_hash)
    if runtime_cache is not None:
        return runtime_cache

    model = get_or_load_model(device)
    feature_bundle = build_or_reuse_gallery_features(
        gallery_image_list=gallery_image_list,
        model=model,
        device=device,
        progress_desc="图库建库",
    )
    gallery_feature_matrix = feature_bundle["image_feature_matrix"]
    gallery_patch_feature_tensor = feature_bundle["patch_feature_tensor"]

    runtime_cache = {
        "gallery_image_list": gallery_image_list,
        "gallery_feature_matrix": gallery_feature_matrix.float(),
        "gallery_patch_feature_tensor": gallery_patch_feature_tensor.float(),
        "gallery_signature": gallery_signature,
    }
    _GALLERY_RUNTIME_CACHE[signature_hash] = runtime_cache
    return runtime_cache


def retrieve_with_feedback(
    reference_path=None,
    feedback_labels=None,
    hidden_paths=None,
    topk=None,
    write_outputs=True,
):
    torch.manual_seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    reference_path = reference_path or reference_image_path
    validate_image_file(reference_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    gallery_resources = ensure_gallery_resources(device)
    gallery_image_list = gallery_resources["gallery_image_list"]
    gallery_feature_matrix = gallery_resources["gallery_feature_matrix"]
    gallery_patch_feature_tensor = gallery_resources["gallery_patch_feature_tensor"]

    print(f"参考图: {reference_path}")
    print(f"图库目录: {gallery_dir}")
    print(f"图库图片数量: {len(gallery_image_list)}")
    print(f"建库批大小: {gallery_batch_size}")
    print(f"输入尺寸: {input_size}")
    print(f"启用 AMP: {use_amp_inference and device.type == 'cuda'}")
    print(f"全局相似度权重: {global_similarity_weight}")
    print(f"局部 patch 相似度权重: {local_similarity_weight}")

    model = get_or_load_model(device)
    query_feature_bundle = extract_single_feature(reference_path, model, device)
    query_feature = query_feature_bundle["image_feature"].float()
    query_patch_features = query_feature_bundle["patch_features"].float()
    base_global_similarity = torch.matmul(gallery_feature_matrix, query_feature)
    base_local_similarity = compute_local_patch_similarity(query_patch_features, gallery_patch_feature_tensor)
    base_similarity = blend_similarity(base_global_similarity, base_local_similarity)
    rerank_similarity, feedback_query_feature = apply_feedback_rerank(
        query_feature=query_feature,
        gallery_feature_matrix=gallery_feature_matrix,
        base_local_similarity=base_local_similarity,
        feedback_labels=feedback_labels,
        gallery_paths=gallery_image_list,
    )

    valid_gallery_indices = get_valid_gallery_indices(
        gallery_paths=gallery_image_list,
        reference_path=reference_path,
        hidden_paths=hidden_paths,
    )
    if len(valid_gallery_indices) == 0:
        raise RuntimeError("排除参考图和已隐藏图片后，图库中没有可展示图片。")

    ranked_valid_indices = sorted(
        valid_gallery_indices,
        key=lambda index: float(rerank_similarity[index].item()),
        reverse=True,
    )
    topk = topk or retrieval_topk
    top_indices = ranked_valid_indices[: min(topk, len(ranked_valid_indices))]
    top_gallery_paths = [gallery_image_list[i] for i in top_indices]
    top_gallery_scores = [round(float(rerank_similarity[i].item()), 6) for i in top_indices]

    result_df = build_result_dataframe(
        reference_path=reference_path,
        gallery_paths=gallery_image_list,
        base_similarity=base_similarity,
        rerank_similarity=rerank_similarity,
        ranked_indices=ranked_valid_indices,
        feedback_labels=feedback_labels,
        base_global_similarity=base_global_similarity,
        base_local_similarity=base_local_similarity,
    )

    if write_outputs:
        result_path = os.path.join(output_dir, "retrieval_result.csv")
        result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
        with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("DINOv2 Query-by-Example 图像检索实验（带反馈加速版）\n")
            f.write("=" * 40 + "\n")
            f.write(f"参考图路径: {reference_path}\n")
            f.write(f"图库目录: {gallery_dir}\n")
            f.write(f"是否递归扫描图库: {gallery_recursive}\n")
            f.write(f"每图大 patch 数量: {patch_count_per_image}\n")
            f.write(f"patch 比例: {patch_scale}\n")
            f.write(f"输入尺寸: {input_size}\n")
            f.write(f"全局相似度权重: {global_similarity_weight}\n")
            f.write(f"局部 patch 相似度权重: {local_similarity_weight}\n")
            f.write(f"建库批大小: {gallery_batch_size}\n")
            f.write(f"反馈正样本权重: {feedback_alpha}\n")
            f.write(f"反馈负样本权重: {feedback_beta}\n")
            f.write(f"反馈查询向量范数: {float(torch.linalg.norm(feedback_query_feature).item()):.6f}\n")
            f.write(f"是否启用图库特征缓存: {use_gallery_feature_cache}\n")
            f.write(f"图库图片数量: {len(gallery_image_list)}\n")
            f.write(f"输出 TopK: {topk}\n")
            f.write("\nTopK 检索结果:\n")
            for rank, gallery_path, score in zip(range(1, len(top_gallery_paths) + 1), top_gallery_paths, top_gallery_scores):
                f.write(f"Top{rank} | rerank_score {score:.6f} | path {gallery_path}\n")

    return {
        "result_df": result_df,
        "top_gallery_paths": top_gallery_paths,
        "top_gallery_scores": top_gallery_scores,
        "gallery_image_list": gallery_image_list,
    }


def main():
    retrieval_result = retrieve_with_feedback(
        reference_path=reference_image_path,
        feedback_labels=None,
        hidden_paths=None,
        topk=retrieval_topk,
        write_outputs=True,
    )
    print("\n实验完成")
    print("检索结果文件:", os.path.join(output_dir, "retrieval_result.csv"))
    print("Top1:", retrieval_result["top_gallery_paths"][0] if retrieval_result["top_gallery_paths"] else "无")
    print("结果目录:", output_dir)


if __name__ == "__main__":
    main()
