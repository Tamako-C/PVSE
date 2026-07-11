from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import torch

from pvse.core.prototypes import DEFAULT_LOGIT_SCALE, entropy_np, prototype_probabilities
from pvse.patch.correspondence import CLEAN_PATCH_FEATURES, class_stats_for_query, clean_patch_feature_dict

PROPOSAL_FEATURES: tuple[str, ...] = (
    "original_proposal_score",
    "original_proposal_utility",
    "original_apply",
    "A0_prob",
    "A0_margin",
    "A0_entropy",
    "proposal_base_prob",
    "proposal_A0_posterior_ratio",
)

CLEAN_VERIFIER_FEATURES: tuple[str, ...] = CLEAN_PATCH_FEATURES + PROPOSAL_FEATURES


def runner_up_classes(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape [queries, classes]")
    a0 = probs.argmax(axis=1)
    tmp = probs.copy()
    tmp[np.arange(len(a0)), a0] = -np.inf
    return tmp.argmax(axis=1).astype(np.int64)


def clean_feature_rows(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    support_maps: torch.Tensor,
    query_maps: torch.Tensor,
    *,
    true_labels: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    way: int = 5,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
    proposal_score_mode: Literal["probability", "probability_difference"] = "probability",
) -> list[dict[str, Any]]:
    """Build the 41-dimensional runner-up verifier rows.

    ``proposal_score_mode="probability"`` is the miniImageNet calibration/test
    serialization and therefore the paper default. The external-transfer
    protocol serializes the proposal field as ``p(proposal)-p(A0)``; callers
    executing that protocol request ``"probability_difference"`` explicitly.
    """
    queries = np.asarray(query_features, dtype=np.float32)
    if queries.ndim != 2:
        raise ValueError("query_features must have shape [Q,D]")
    if query_maps.ndim != 4 or int(query_maps.shape[0]) != len(queries):
        raise ValueError("query_maps must have shape [Q,C,H,W]")
    labels = np.asarray(support_labels, dtype=np.int64)
    probs = prototype_probabilities(
        support_features,
        labels,
        queries,
        way=int(way),
        logit_scale=float(logit_scale),
    )
    a0 = probs.argmax(axis=1).astype(np.int64)
    proposal = runner_up_classes(probs)
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    entropies = entropy_np(probs, axis=1)
    truth = None if true_labels is None else np.asarray(true_labels, dtype=np.int64)
    if truth is not None and len(truth) != len(queries):
        raise ValueError("true_labels must match query_features")
    if proposal_score_mode not in {"probability", "probability_difference"}:
        raise ValueError(f"unsupported proposal_score_mode: {proposal_score_mode}")
    prefix = dict(metadata or {})
    rows: list[dict[str, Any]] = []
    for query_index in range(len(queries)):
        a0_class = int(a0[query_index])
        proposal_class = int(proposal[query_index])
        stats = class_stats_for_query(
            query_maps[query_index],
            support_maps,
            labels,
            way=int(way),
        )
        proposal_probability = float(probs[query_index, proposal_class])
        proposal_difference = float(proposal_probability - probs[query_index, a0_class])
        serialized_proposal_score = (
            proposal_probability if proposal_score_mode == "probability" else proposal_difference
        )
        row: dict[str, Any] = {
            **prefix,
            "query_index": int(query_index),
            "proposal_source": "runnerup_a0",
            "a0_pred": a0_class,
            "proposal_class": proposal_class,
            "original_proposal_score": serialized_proposal_score,
            "original_proposal_utility": serialized_proposal_score,
            "original_apply": 1,
            "A0_prob": float(probs[query_index, a0_class]),
            "A0_margin": float(margins[query_index]),
            "A0_entropy": float(entropies[query_index]),
            "proposal_base_prob": float(probs[query_index, proposal_class]),
            "proposal_A0_posterior_ratio": float(
                probs[query_index, proposal_class] / max(1e-12, float(probs[query_index, a0_class]))
            ),
        }
        row.update(
            clean_patch_feature_dict(
                stats,
                a0_prediction=a0_class,
                proposal=proposal_class,
            )
        )
        if truth is not None:
            y = int(truth[query_index])
            row.update(
                {
                    "analysis_only_true_label": y,
                    "analysis_only_a0_correct": int(a0_class == y),
                    "help_label": int(a0_class != y and proposal_class == y),
                    "hurt_label": int(a0_class == y and proposal_class != y),
                }
            )
        rows.append(row)
    return rows


def feature_matrix(rows: Sequence[dict[str, Any]], features: Sequence[str] = CLEAN_VERIFIER_FEATURES) -> np.ndarray:
    missing = sorted({name for name in features if any(name not in row for row in rows)})
    if missing:
        raise KeyError(f"missing clean verifier features: {missing}")
    matrix = np.asarray([[row[name] for name in features] for row in rows], dtype=np.float32)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
