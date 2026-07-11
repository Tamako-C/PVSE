from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pvse.core.prototypes import DEFAULT_LOGIT_SCALE, prototype_predict

QueryScoreMode = Literal["margin", "maxprob"]


def _validated_probabilities(prob_corrupt: np.ndarray) -> np.ndarray:
    p = np.asarray(prob_corrupt, dtype=np.float32)
    if p.ndim != 1:
        raise ValueError(f"prob_corrupt must be one-dimensional, got {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("prob_corrupt must contain only finite values")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("prob_corrupt must lie in [0, 1]")
    return p


def descending_probability_order(prob_corrupt: np.ndarray) -> np.ndarray:
    """Return the paper-code ordering: descending probability, stable by index."""
    p = _validated_probabilities(prob_corrupt)
    return np.argsort(-p, kind="mergesort").astype(np.int64)


def fixed_topk_keep(prob_corrupt: np.ndarray, k: int) -> np.ndarray:
    """Delete exactly the top-``k`` supports ranked by corruption probability."""
    p = _validated_probabilities(prob_corrupt)
    if int(k) < 0 or int(k) > len(p):
        raise ValueError(f"k must be in [0, {len(p)}]")
    keep = np.ones(len(p), dtype=np.float32)
    if int(k):
        keep[descending_probability_order(p)[: int(k)]] = 0.0
    return keep


def threshold_cap_keep(
    prob_corrupt: np.ndarray,
    *,
    threshold: float,
    max_delete: int = 3,
) -> np.ndarray:
    """Delete up to ``max_delete`` high-ranked supports whose probability passes a threshold.

    This is the episode-level adaptive ``threshold-cap`` policy used in Table 6.
    Ranking is stable and only the first ``max_delete`` candidates are eligible.
    """
    p = _validated_probabilities(prob_corrupt)
    if int(max_delete) < 0:
        raise ValueError("max_delete must be non-negative")
    keep = np.ones(len(p), dtype=np.float32)
    for idx in descending_probability_order(p)[: int(max_delete)]:
        if float(p[int(idx)]) >= float(threshold):
            keep[int(idx)] = 0.0
    return keep


def per_class_topr_keep(
    prob_corrupt: np.ndarray,
    support_labels: np.ndarray,
    *,
    r: int = 1,
    way: int = 5,
) -> np.ndarray:
    """Delete exactly the top-``r`` supports within each observed class."""
    p = _validated_probabilities(prob_corrupt)
    labels = np.asarray(support_labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != len(p):
        raise ValueError("support_labels must match prob_corrupt")
    if int(r) < 0:
        raise ValueError("r must be non-negative")
    keep = np.ones(len(p), dtype=np.float32)
    for c in range(int(way)):
        idx = np.flatnonzero(labels == c)
        if len(idx) == 0:
            raise ValueError(f"class {c} has no supports")
        local_order = np.argsort(-p[idx], kind="mergesort")
        selected = idx[local_order[: min(int(r), len(idx))]]
        keep[selected] = 0.0
    return keep


def probability_confidence(probabilities: np.ndarray, mode: QueryScoreMode) -> float:
    prob = np.asarray(probabilities, dtype=np.float64)
    if prob.ndim != 1 or len(prob) < 2:
        raise ValueError("probabilities must be one-dimensional with at least two classes")
    ranked = np.sort(prob)[::-1]
    if mode == "margin":
        return float(ranked[0] - ranked[1])
    if mode == "maxprob":
        return float(ranked[0])
    raise ValueError(f"unsupported score mode: {mode}")


@dataclass(frozen=True)
class QueryBudgetDecision:
    prediction: int
    baseline_prediction: int
    selected_k: int
    deleted: tuple[int, ...]
    keep_weights: np.ndarray
    confidence: float
    objective: float
    probabilities: np.ndarray

    @property
    def applied(self) -> bool:
        return bool(self.prediction != self.baseline_prediction)


def query_adaptive_budget_decision(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_feature: np.ndarray,
    prob_corrupt: np.ndarray,
    *,
    edit_cost: float,
    score_mode: QueryScoreMode,
    max_delete: int = 3,
    way: int = 5,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> QueryBudgetDecision:
    """Select a prefix deletion budget ``k=0..max_delete`` for one query.

    The candidate order is the stable descending corruption-probability order.
    Candidate objective is either top-1 probability or top-1/top-2 margin minus
    ``edit_cost * k``. The update is strict (``>``), so exact ties retain the
    smaller deletion budget, matching the submitted Table 6 implementation.
    """
    p = _validated_probabilities(prob_corrupt)
    labels = np.asarray(support_labels, dtype=np.int64)
    if len(labels) != len(p):
        raise ValueError("support_labels must match prob_corrupt")
    max_k = min(int(max_delete), len(p))
    if max_k < 0:
        raise ValueError("max_delete must be non-negative")
    order = descending_probability_order(p)[:max_k]

    baseline = prototype_predict(
        support_features,
        labels,
        query_feature,
        way=int(way),
        weights=np.ones(len(p), dtype=np.float32),
        logit_scale=float(logit_scale),
    )
    baseline_prediction = int(np.asarray(baseline.predictions).item())

    best: QueryBudgetDecision | None = None
    for k in range(max_k + 1):
        keep = np.ones(len(p), dtype=np.float32)
        if k:
            keep[order[:k]] = 0.0
        pred = prototype_predict(
            support_features,
            labels,
            query_feature,
            way=int(way),
            weights=keep,
            logit_scale=float(logit_scale),
        )
        probabilities = np.asarray(pred.probabilities, dtype=np.float32)
        confidence = probability_confidence(probabilities, score_mode)
        objective = float(confidence - float(edit_cost) * k)
        candidate = QueryBudgetDecision(
            prediction=int(np.asarray(pred.predictions).item()),
            baseline_prediction=baseline_prediction,
            selected_k=int(k),
            deleted=tuple(int(i) for i in order[:k]),
            keep_weights=keep,
            confidence=float(confidence),
            objective=objective,
            probabilities=probabilities,
        )
        if best is None or candidate.objective > best.objective:
            best = candidate
    assert best is not None
    return best


def query_adaptive_budget_predict(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    prob_corrupt: np.ndarray,
    *,
    edit_cost: float,
    score_mode: QueryScoreMode,
    max_delete: int = 3,
    way: int = 5,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> list[QueryBudgetDecision]:
    queries = np.asarray(query_features, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries[None, :]
    if queries.ndim != 2:
        raise ValueError("query_features must be one- or two-dimensional")
    return [
        query_adaptive_budget_decision(
            support_features,
            support_labels,
            query,
            prob_corrupt,
            edit_cost=float(edit_cost),
            score_mode=score_mode,
            max_delete=int(max_delete),
            way=int(way),
            logit_scale=float(logit_scale),
        )
        for query in queries
    ]
