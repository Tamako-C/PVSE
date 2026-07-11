from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from pvse.core.prototypes import l2_normalize
from pvse.patch.correspondence import NOISY_PATCH_FEATURES, support_patch_feature_rows

GLOBAL_RELIABILITY_FEATURES: tuple[str, ...] = (
    "cos_to_own_class_centroid_LOO",
    "distance_to_own_class_centroid_LOO",
    "nearest_other_class_centroid_similarity",
    "gap_own_vs_nearest_other",
    "support_norm",
    "class_compactness_LOO",
    "own_class_rank_within_support",
    "nearest_neighbor_same_class_similarity",
    "nearest_neighbor_other_class_similarity",
    "within_class_outlier_percentile",
    "prototype_shift_if_removed_norm",
    "class_compactness_gain_if_removed",
    "nearest_other_gap_gain_if_removed",
)

RELIABILITY_FEATURES: tuple[str, ...] = GLOBAL_RELIABILITY_FEATURES + NOISY_PATCH_FEATURES


def _class_compactness(support_norm: np.ndarray, indices: np.ndarray) -> float:
    if len(indices) <= 1:
        return 0.0
    centroid = l2_normalize(support_norm[indices].mean(axis=0, keepdims=True))[0]
    return float((support_norm[indices] @ centroid).mean())


def global_support_feature_rows(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    *,
    corrupt_mask: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    way: int = 5,
) -> list[dict[str, Any]]:
    """Build the 13 query-independent support-level global features."""
    raw = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels, dtype=np.int64)
    if raw.ndim != 2 or labels.ndim != 1 or len(raw) != len(labels):
        raise ValueError("support_features/support_labels shape mismatch")
    if any(not np.any(labels == c) for c in range(int(way))):
        raise ValueError("every episodic class must contain support examples")
    support = l2_normalize(raw, axis=1)
    corrupt = np.zeros(len(labels), dtype=bool) if corrupt_mask is None else np.asarray(corrupt_mask, dtype=bool)
    if len(corrupt) != len(labels):
        raise ValueError("corrupt_mask must match support labels")


    centroids = np.zeros((int(way), support.shape[1]), dtype=np.float32)
    compact_before = np.zeros(int(way), dtype=np.float32)
    class_gap_before = np.zeros(int(way), dtype=np.float32)
    for c in range(int(way)):
        idx = np.flatnonzero(labels == c)
        centroids[c] = l2_normalize(support[idx].mean(axis=0, keepdims=True))[0]
    for c in range(int(way)):
        idx = np.flatnonzero(labels == c)
        compact_before[c] = _class_compactness(support, idx)
        other = np.ones(int(way), dtype=bool)
        other[c] = False
        class_gap_before[c] = float((centroids[c] @ centroids[c]) - (centroids[other] @ centroids[c]).max())

    prefix = dict(metadata or {})
    rows: list[dict[str, Any]] = []
    for i, observed_label in enumerate(labels.astype(int)):
        class_idx = np.flatnonzero(labels == observed_label)
        loo_idx = class_idx[class_idx != i]
        own_centroid = (
            l2_normalize(support[loo_idx].mean(axis=0, keepdims=True))[0]
            if len(loo_idx)
            else centroids[observed_label]
        )
        own_similarity = float(support[i] @ own_centroid)
        other_mask = np.ones(int(way), dtype=bool)
        other_mask[observed_label] = False
        nearest_other = float((centroids[other_mask] @ support[i]).max())
        same_nn = float((support[loo_idx] @ support[i]).max()) if len(loo_idx) else own_similarity
        other_idx = np.flatnonzero(labels != observed_label)
        other_nn = float((support[other_idx] @ support[i]).max()) if len(other_idx) else 0.0

        own_scores: list[float] = []
        for j in class_idx:
            j_loo = class_idx[class_idx != j]
            j_cent = (
                l2_normalize(support[j_loo].mean(axis=0, keepdims=True))[0]
                if len(j_loo)
                else centroids[observed_label]
            )
            own_scores.append(float(support[j] @ j_cent))
        rank = int(1 + np.sum(np.asarray(own_scores) < own_similarity))
        percentile = float((rank - 1) / max(len(class_idx) - 1, 1))

        compact_after = _class_compactness(support, loo_idx) if len(loo_idx) else float(compact_before[observed_label])
        loo_centroids = centroids.copy()
        if len(loo_idx):
            loo_centroids[observed_label] = l2_normalize(support[loo_idx].mean(axis=0, keepdims=True))[0]
        other = np.ones(int(way), dtype=bool)
        other[observed_label] = False
        class_gap_after = float(
            (loo_centroids[observed_label] @ loo_centroids[observed_label])
            - (loo_centroids[other] @ loo_centroids[observed_label]).max()
        )
        row: dict[str, Any] = {
            **prefix,
            "support_index": int(i),
            "observed_support_label_for_analysis": int(observed_label),
            "corrupted_for_analysis_only": int(corrupt[i]),
            "cos_to_own_class_centroid_LOO": own_similarity,
            "distance_to_own_class_centroid_LOO": float(1.0 - own_similarity),
            "nearest_other_class_centroid_similarity": nearest_other,
            "gap_own_vs_nearest_other": float(own_similarity - nearest_other),
            "support_norm": float(np.linalg.norm(raw[i])),
            "class_compactness_LOO": float(compact_after),
            "own_class_rank_within_support": rank,
            "nearest_neighbor_same_class_similarity": same_nn,
            "nearest_neighbor_other_class_similarity": other_nn,
            "within_class_outlier_percentile": percentile,
            "prototype_shift_if_removed_norm": float(
                np.linalg.norm(centroids[observed_label] - loo_centroids[observed_label])
            ),
            "class_compactness_gain_if_removed": float(compact_after - compact_before[observed_label]),
            "nearest_other_gap_gain_if_removed": float(class_gap_after - class_gap_before[observed_label]),
        }
        rows.append(row)
    return rows


def reliability_feature_rows(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    support_maps: torch.Tensor,
    *,
    corrupt_mask: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    way: int = 5,
) -> list[dict[str, Any]]:
    global_rows = global_support_feature_rows(
        support_features,
        support_labels,
        corrupt_mask=corrupt_mask,
        metadata=metadata,
        way=int(way),
    )
    patch_rows = support_patch_feature_rows(
        support_maps,
        support_labels,
        corrupt_mask=corrupt_mask,
        metadata=metadata,
        way=int(way),
    )
    patch_by_index = {int(row["support_index"]): row for row in patch_rows}
    out: list[dict[str, Any]] = []
    for row in global_rows:
        merged = dict(row)
        patch = patch_by_index[int(row["support_index"])]
        for key in NOISY_PATCH_FEATURES:
            merged[key] = patch[key]
        out.append(merged)
    return out


def reliability_matrix(
    rows: Sequence[dict[str, Any]],
    features: Sequence[str] = RELIABILITY_FEATURES,
) -> np.ndarray:
    missing = sorted({name for name in features if any(name not in row for row in rows)})
    if missing:
        raise KeyError(f"missing reliability features: {missing}")
    x = np.asarray([[row[name] for name in features] for row in rows], dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
