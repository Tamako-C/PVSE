from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from pvse.core.prototypes import DEFAULT_LOGIT_SCALE, l2_normalize, softmax_np
from pvse.editing.delete_lattice import DeleteActionBank, build_action_bank


@dataclass(frozen=True)
class QueryLatticeResult:
    query_index: int
    true_label: int
    a0_prediction: int
    a0_correct: bool
    a0_probability: float
    a0_margin: float
    oracle_correct: bool
    scene_type: str
    correct_action_count: int
    correct_delete_action_count: int
    total_action_count: int
    total_delete_action_count: int
    correct_action_density: float
    correct_delete_density: float
    correct_actions_k1: int
    correct_actions_k2: int
    correct_actions_k3: int
    minimum_correct_delete_size: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeLatticeResult:
    query_results: tuple[QueryLatticeResult, ...]
    action_count: int
    delete_action_count: int

    def rows(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.query_results]


def action_prototypes(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    action_bank: DeleteActionBank,
    *,
    way: int = 5,
) -> np.ndarray:
    """Compute normalized prototypes for every hard-delete action.

    Returns an array with shape ``[actions, way, dimension]``. Supports are
    normalized before averaging, and each action-specific class prototype is
    normalized afterward, matching the prototype classifier used throughout
    the submitted experiments.
    """
    support = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels, dtype=np.int64)
    if support.ndim != 2 or labels.ndim != 1 or len(support) != len(labels):
        raise ValueError("support features/labels have incompatible shapes")
    if action_bank.keep_masks.shape[1] != len(labels):
        raise ValueError("action bank does not match the support count")
    missing = [c for c in range(int(way)) if not np.any(labels == c)]
    if missing:
        raise ValueError(f"classes without support examples: {missing}")

    normalized = l2_normalize(support, axis=1)
    masks = np.asarray(action_bank.keep_masks, dtype=np.float32)
    per_class: list[np.ndarray] = []
    for c in range(int(way)):
        class_membership = (labels == c).astype(np.float32)
        weights = masks * class_membership[None, :]
        counts = weights.sum(axis=1, keepdims=True)
        if np.any(counts <= 0):
            raise ValueError(
                "an action removes every support from a class; reduce kmax or provide more shots"
            )
        means = (weights @ normalized) / counts
        per_class.append(l2_normalize(means, axis=1))
    return np.stack(per_class, axis=1).astype(np.float32, copy=False)


def evaluate_episode_lattice(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    true_labels: np.ndarray,
    *,
    action_bank: DeleteActionBank | None = None,
    kmax: int = 3,
    way: int = 5,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> EpisodeLatticeResult:
    """Evaluate the complete delete-``<=kmax`` lattice for all episode queries."""
    labels = np.asarray(support_labels, dtype=np.int64)
    queries = np.asarray(query_features, dtype=np.float32)
    truth = np.asarray(true_labels, dtype=np.int64)
    if queries.ndim == 1:
        queries = queries[None, :]
    if queries.ndim != 2 or truth.ndim != 1 or len(queries) != len(truth):
        raise ValueError("query features and labels have incompatible shapes")
    bank = action_bank or build_action_bank(len(labels), int(kmax))
    protos = action_prototypes(support_features, labels, bank, way=int(way))
    q = l2_normalize(queries, axis=1)
    # [queries, actions, classes]
    logits = (
        float(logit_scale) * np.einsum("qd,awd->qaw", q, protos, optimize=True)
    ).astype(np.float32)
    predictions = np.argmax(logits, axis=2).astype(np.int64)
    correct = predictions == truth[:, None]

    a0_logits = logits[:, 0, :]
    a0_probabilities = softmax_np(a0_logits, axis=1).astype(np.float32)
    a0_predictions = predictions[:, 0]
    sorted_probs = np.sort(a0_probabilities, axis=1)
    a0_margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    delete_mask = np.asarray(bank.k_values > 0, dtype=bool)
    delete_action_count = int(delete_mask.sum())
    rows: list[QueryLatticeResult] = []
    for qi in range(len(queries)):
        correct_delete = correct[qi] & delete_mask
        a0_correct = bool(correct[qi, 0])
        has_correct_delete = bool(correct_delete.any())
        scene = "safe" if a0_correct else ("rescue" if has_correct_delete else "dead")
        k_counts = {
            k: int((correct[qi] & (bank.k_values == k)).sum()) for k in (1, 2, 3)
        }
        valid_k = bank.k_values[correct_delete]
        rows.append(
            QueryLatticeResult(
                query_index=int(qi),
                true_label=int(truth[qi]),
                a0_prediction=int(a0_predictions[qi]),
                a0_correct=a0_correct,
                a0_probability=float(a0_probabilities[qi, a0_predictions[qi]]),
                a0_margin=float(a0_margins[qi]),
                oracle_correct=bool(a0_correct or has_correct_delete),
                scene_type=scene,
                correct_action_count=int(correct[qi].sum()),
                correct_delete_action_count=int(correct_delete.sum()),
                total_action_count=int(len(bank)),
                total_delete_action_count=delete_action_count,
                # Rescue density is defined over the complete 2,626-action bank,
                # including A0. For rescue rows A0 is incorrect, so the numerator
                # is the number of correct delete actions and the denominator is
                # still 2,626 rather than 2,625. Keep the delete-only density as
                # a separate delete-only density field.
                correct_action_density=float(correct[qi].sum() / max(1, len(bank))),
                correct_delete_density=float(correct_delete.sum() / max(1, delete_action_count)),
                correct_actions_k1=k_counts[1],
                correct_actions_k2=k_counts[2],
                correct_actions_k3=k_counts[3],
                minimum_correct_delete_size=(int(valid_k.min()) if len(valid_k) else None),
            )
        )
    return EpisodeLatticeResult(tuple(rows), len(bank), delete_action_count)


def evaluate_query_lattice(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_feature: np.ndarray,
    true_label: int,
    *,
    action_bank: DeleteActionBank | None = None,
    way: int = 5,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> QueryLatticeResult:
    """Compatibility wrapper for a single query."""
    result = evaluate_episode_lattice(
        support_features,
        support_labels,
        np.asarray(query_feature, dtype=np.float32)[None, :],
        np.asarray([true_label], dtype=np.int64),
        action_bank=action_bank,
        way=int(way),
        logit_scale=float(logit_scale),
    )
    return result.query_results[0]
