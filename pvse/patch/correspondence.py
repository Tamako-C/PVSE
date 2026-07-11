from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

NOISY_PATCH_FEATURES: tuple[str, ...] = (
    "own_patch_match_mean",
    "own_patch_match_max",
    "own_patch_match_std",
    "own_patch_top5_mean",
    "own_patch_top10_mean",
    "own_patch_coverage_t05",
    "own_patch_coverage_t06",
    "own_patch_coverage_t07",
    "own_patch_entropy",
    "own_support_agreement_count",
    "own_support_agreement_mean",
    "own_support_agreement_std",
    "own_support_agreement_max",
    "nearest_other_patch_match",
    "nearest_other_patch_class",
    "own_minus_nearest_other_patch_margin",
    "other_patch_rank_of_observed_class",
    "best_matching_class_is_not_observed",
    "second_best_other_patch_match",
    "other_patch_entropy",
    "own_q2s",
    "own_s2q",
    "own_bidir",
    "nearest_other_q2s",
    "nearest_other_s2q",
    "nearest_other_bidir",
    "own_minus_other_bidir",
)

CLEAN_PATCH_FEATURES: tuple[str, ...] = (
    "patch_score_proposal_q2s",
    "patch_score_a0_q2s",
    "patch_score_proposal_s2q",
    "patch_score_a0_s2q",
    "patch_score_proposal_bidir",
    "patch_score_a0_bidir",
    "patch_margin_over_a0_bidir",
    "patch_margin_over_a0_q2s",
    "patch_margin_over_a0_s2q",
    "proposal_patch_rank_bidir",
    "proposal_patch_rank_q2s",
    "proposal_patch_rank_s2q",
    "proposal_equals_best_patch_class",
    "proposal_equals_best_nonA0_patch_class",
    "patch_margin_over_best_other",
    "proposal_patch_top5_mean",
    "proposal_patch_top10_mean",
    "a0_patch_top5_mean",
    "a0_patch_top10_mean",
    "proposal_minus_a0_top5",
    "proposal_minus_a0_top10",
    "proposal_patch_coverage_t05",
    "proposal_patch_coverage_t06",
    "proposal_patch_coverage_t07",
    "proposal_patch_entropy",
    "a0_patch_entropy",
    "proposal_support_agreement_count",
    "proposal_support_agreement_mean",
    "proposal_support_agreement_std",
    "proposal_support_agreement_max",
    "a0_support_agreement_mean",
    "a0_support_agreement_std",
    "a0_support_agreement_max",
)


def patch_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """Convert ``[C,H,W]`` or ``[N,C,H,W]`` feature maps to normalized patches."""
    if feature_map.ndim == 4:
        patches = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
    elif feature_map.ndim == 3:
        patches = feature_map.permute(1, 2, 0).reshape(-1, feature_map.shape[0])
    else:
        raise ValueError(f"feature_map must have 3 or 4 dimensions, got {tuple(feature_map.shape)}")
    return F.normalize(patches.float(), p=2, dim=-1)


def _entropy_from_counts(counts: torch.Tensor) -> float:
    total = counts.sum().item()
    if total <= 0:
        return 0.0
    p = counts.float() / float(total)
    p = p[p > 0]
    return float(-(p * torch.log(p)).sum().item())


def empty_patch_stats(topks: Sequence[int] = (5, 10), thresholds: Sequence[float] = (0.5, 0.6, 0.7)) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "score": 0.0,
        "bidirectional_score": 0.0,
        "max_patch_match": 0.0,
        "std_patch_match": 0.0,
        "support_hit_entropy": 0.0,
        "support_hit_max_frac": 0.0,
        "support_instance_score_mean": 0.0,
        "support_instance_score_max": 0.0,
        "support_instance_score_std": 0.0,
        "support_agreement_count": 0,
    }
    for k in topks:
        out[f"top{int(k)}_patch_match"] = 0.0
    for threshold in thresholds:
        out[f"coverage_{str(float(threshold)).replace('.', '')}"] = 0.0
    return out


def class_patch_stats(
    query_map: torch.Tensor,
    support_maps: torch.Tensor,
    *,
    topks: Sequence[int] = (5, 10),
    thresholds: Sequence[float] = (0.5, 0.6, 0.7),
) -> dict[str, float | int]:
    """Compute the exact bidirectional patch statistics used by PVSE.

    ``score`` is query-to-support. ``bidirectional_score`` averages
    query-to-support and support-to-query maxima. Class-level support agreement
    is computed from each support instance independently.
    """
    if support_maps.ndim != 4:
        raise ValueError("support_maps must have shape [N,C,H,W]")
    if int(support_maps.shape[0]) <= 0:
        return empty_patch_stats(topks, thresholds)
    q = patch_matrix(query_map)
    n_support = int(support_maps.shape[0])
    patches_per_support = int(support_maps.shape[2] * support_maps.shape[3])
    s = patch_matrix(support_maps)
    sim = q @ s.T
    max_q, arg_q = sim.max(dim=1)
    max_s = sim.max(dim=0).values
    out: dict[str, float | int] = {
        "score": float(max_q.mean().item()),
        "bidirectional_score": float(0.5 * (max_q.mean().item() + max_s.mean().item())),
        "max_patch_match": float(max_q.max().item()),
        "std_patch_match": float(max_q.std(unbiased=False).item()),
    }
    for k in topks:
        kk = min(int(k), int(max_q.numel()))
        out[f"top{int(k)}_patch_match"] = float(torch.topk(max_q, kk).values.mean().item())
    for threshold in thresholds:
        key = str(float(threshold)).replace(".", "")
        out[f"coverage_{key}"] = float((max_q >= float(threshold)).float().mean().item())

    support_hit = torch.div(arg_q, patches_per_support, rounding_mode="floor").clamp(0, n_support - 1)
    counts = torch.bincount(support_hit, minlength=n_support)
    out["support_hit_entropy"] = _entropy_from_counts(counts)
    out["support_hit_max_frac"] = float(counts.max().item() / max(1, counts.sum().item()))

    per_support_scores: list[float] = []
    for support_idx in range(n_support):
        sj = patch_matrix(support_maps[support_idx])
        per_support_scores.append(float((q @ sj.T).max(dim=1).values.mean().item()))
    scores = np.asarray(per_support_scores, dtype=np.float64)
    out["support_instance_score_mean"] = float(scores.mean())
    out["support_instance_score_max"] = float(scores.max())
    out["support_instance_score_std"] = float(scores.std())
    out["support_agreement_count"] = int((scores >= (scores.max() - 0.05)).sum())
    return out


def rank_desc(values: np.ndarray, class_index: int) -> int:
    order = np.argsort(-np.asarray(values, dtype=np.float64), kind="mergesort")
    return int(np.where(order == int(class_index))[0][0] + 1)


def class_stats_for_query(
    query_map: torch.Tensor,
    support_maps: torch.Tensor,
    support_labels: np.ndarray,
    *,
    way: int = 5,
) -> list[dict[str, float | int]]:
    labels = np.asarray(support_labels, dtype=np.int64)
    return [
        class_patch_stats(
            query_map,
            support_maps[labels == c],
            topks=(5, 10, 20),
            thresholds=(0.5, 0.6, 0.7),
        )
        for c in range(int(way))
    ]


def clean_patch_feature_dict(
    class_stats: Sequence[dict[str, float | int]],
    *,
    a0_prediction: int,
    proposal: int,
) -> dict[str, float | int]:
    q2s = np.asarray([float(st["score"]) for st in class_stats], dtype=np.float64)
    bidir = np.asarray([float(st["bidirectional_score"]) for st in class_stats], dtype=np.float64)
    s2q = 2.0 * bidir - q2s
    pc = int(proposal)
    a0 = int(a0_prediction)
    prop = class_stats[pc]
    a0_stats = class_stats[a0]
    other_bidir = bidir.copy()
    other_bidir[pc] = -np.inf
    best_non_a0 = int(np.where(np.arange(len(bidir)) == a0, -np.inf, bidir).argmax())
    return {
        "patch_score_proposal_q2s": float(q2s[pc]),
        "patch_score_a0_q2s": float(q2s[a0]),
        "patch_score_proposal_s2q": float(s2q[pc]),
        "patch_score_a0_s2q": float(s2q[a0]),
        "patch_score_proposal_bidir": float(bidir[pc]),
        "patch_score_a0_bidir": float(bidir[a0]),
        "patch_margin_over_a0_bidir": float(bidir[pc] - bidir[a0]),
        "patch_margin_over_a0_q2s": float(q2s[pc] - q2s[a0]),
        "patch_margin_over_a0_s2q": float(s2q[pc] - s2q[a0]),
        "proposal_patch_rank_bidir": rank_desc(bidir, pc),
        "proposal_patch_rank_q2s": rank_desc(q2s, pc),
        "proposal_patch_rank_s2q": rank_desc(s2q, pc),
        "proposal_equals_best_patch_class": int(pc == int(bidir.argmax())),
        "proposal_equals_best_nonA0_patch_class": int(pc == best_non_a0),
        "patch_margin_over_best_other": float(bidir[pc] - float(np.max(other_bidir))),
        "proposal_patch_top5_mean": float(prop["top5_patch_match"]),
        "proposal_patch_top10_mean": float(prop["top10_patch_match"]),
        "a0_patch_top5_mean": float(a0_stats["top5_patch_match"]),
        "a0_patch_top10_mean": float(a0_stats["top10_patch_match"]),
        "proposal_minus_a0_top5": float(prop["top5_patch_match"]) - float(a0_stats["top5_patch_match"]),
        "proposal_minus_a0_top10": float(prop["top10_patch_match"]) - float(a0_stats["top10_patch_match"]),
        "proposal_patch_coverage_t05": float(prop["coverage_05"]),
        "proposal_patch_coverage_t06": float(prop["coverage_06"]),
        "proposal_patch_coverage_t07": float(prop["coverage_07"]),
        "proposal_patch_entropy": float(prop["support_hit_entropy"]),
        "a0_patch_entropy": float(a0_stats["support_hit_entropy"]),
        "proposal_support_agreement_count": int(prop["support_agreement_count"]),
        "proposal_support_agreement_mean": float(prop["support_instance_score_mean"]),
        "proposal_support_agreement_std": float(prop["support_instance_score_std"]),
        "proposal_support_agreement_max": float(prop["support_instance_score_max"]),
        "a0_support_agreement_mean": float(a0_stats["support_instance_score_mean"]),
        "a0_support_agreement_std": float(a0_stats["support_instance_score_std"]),
        "a0_support_agreement_max": float(a0_stats["support_instance_score_max"]),
    }


def support_patch_feature_rows(
    support_maps: torch.Tensor,
    support_labels: np.ndarray,
    *,
    corrupt_mask: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    way: int = 5,
) -> list[dict[str, Any]]:
    """Return one exact 27-feature row per support, excluding self matching."""
    labels = np.asarray(support_labels, dtype=np.int64)
    if support_maps.ndim != 4 or len(support_maps) != len(labels):
        raise ValueError("support_maps and support_labels must describe the same supports")
    corrupt = np.zeros(len(labels), dtype=bool) if corrupt_mask is None else np.asarray(corrupt_mask, dtype=bool)
    if len(corrupt) != len(labels):
        raise ValueError("corrupt_mask must match support_labels")
    prefix = dict(metadata or {})
    rows: list[dict[str, Any]] = []
    for i, observed_label in enumerate(labels.astype(int)):
        own_idx = np.flatnonzero(labels == observed_label)
        own_idx = own_idx[own_idx != i]
        own = class_patch_stats(
            support_maps[i],
            support_maps[own_idx],
            topks=(5, 10),
            thresholds=(0.5, 0.6, 0.7),
        )
        class_scores: dict[int, dict[str, float | int]] = {}
        for c in range(int(way)):
            idx = np.flatnonzero(labels == c)
            if c == observed_label:
                idx = idx[idx != i]
            class_scores[c] = class_patch_stats(
                support_maps[i],
                support_maps[idx],
                topks=(5, 10),
                thresholds=(0.5, 0.6, 0.7),
            )
        bidir = np.asarray([float(class_scores[c]["bidirectional_score"]) for c in range(int(way))], dtype=np.float64)
        q2s = np.asarray([float(class_scores[c]["score"]) for c in range(int(way))], dtype=np.float64)
        s2q = 2.0 * bidir - q2s
        other_classes = [c for c in range(int(way)) if c != observed_label]
        # Python's stable sort preserves ascending class id under exact ties.
        other_sorted = sorted(other_classes, key=lambda c: bidir[c], reverse=True)
        nearest = int(other_sorted[0])
        second = int(other_sorted[1]) if len(other_sorted) > 1 else nearest
        own_rank = int(1 + np.sum(bidir > bidir[observed_label]))
        exp = np.exp(bidir - bidir.max())
        p = exp / np.maximum(exp.sum(), 1e-12)
        p_nonzero = p[p > 0]
        row: dict[str, Any] = {
            **prefix,
            "support_index": int(i),
            "observed_support_label_for_analysis": int(observed_label),
            "corrupted_for_analysis_only": int(corrupt[i]),
            "own_patch_match_mean": float(own["score"]),
            "own_patch_match_max": float(own["max_patch_match"]),
            "own_patch_match_std": float(own["std_patch_match"]),
            "own_patch_top5_mean": float(own["top5_patch_match"]),
            "own_patch_top10_mean": float(own["top10_patch_match"]),
            "own_patch_coverage_t05": float(own["coverage_05"]),
            "own_patch_coverage_t06": float(own["coverage_06"]),
            "own_patch_coverage_t07": float(own["coverage_07"]),
            "own_patch_entropy": float(own["support_hit_entropy"]),
            "own_support_agreement_count": int(own["support_agreement_count"]),
            "own_support_agreement_mean": float(own["support_instance_score_mean"]),
            "own_support_agreement_std": float(own["support_instance_score_std"]),
            "own_support_agreement_max": float(own["support_instance_score_max"]),
            "nearest_other_patch_match": float(bidir[nearest]),
            "nearest_other_patch_class": nearest,
            "own_minus_nearest_other_patch_margin": float(bidir[observed_label] - bidir[nearest]),
            "other_patch_rank_of_observed_class": own_rank,
            "best_matching_class_is_not_observed": int(int(np.argmax(bidir)) != observed_label),
            "second_best_other_patch_match": float(bidir[second]),
            "other_patch_entropy": float(-(p_nonzero * np.log(p_nonzero)).sum()),
            "own_q2s": float(q2s[observed_label]),
            "own_s2q": float(s2q[observed_label]),
            "own_bidir": float(bidir[observed_label]),
            "nearest_other_q2s": float(q2s[nearest]),
            "nearest_other_s2q": float(s2q[nearest]),
            "nearest_other_bidir": float(bidir[nearest]),
            "own_minus_other_bidir": float(bidir[observed_label] - bidir[nearest]),
        }
        rows.append(row)
    return rows
