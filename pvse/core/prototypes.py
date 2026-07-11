from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_LOGIT_SCALE = 10.0
_EPS = 1e-12
_WEIGHT_EPS = 1e-6


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = _EPS) -> np.ndarray:
    """L2-normalize an array using the numerical convention of the paper code."""
    arr = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(denom, float(eps))


def softmax_np(logits: np.ndarray, *, scale: float = 1.0, axis: int = -1) -> np.ndarray:
    """Stable NumPy softmax; ``scale`` is applied before exponentiation."""
    z = np.asarray(logits, dtype=np.float64) * float(scale)
    z = z - np.max(z, axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), _EPS)


def entropy_np(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def _validated_inputs(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    way: int,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels, dtype=np.int64)
    if support.ndim != 2:
        raise ValueError(f"support_features must be 2-D, got {support.shape}")
    if labels.ndim != 1 or len(labels) != len(support):
        raise ValueError("support_labels must be one-dimensional and match support_features")
    if int(way) <= 1:
        raise ValueError("way must be at least 2")
    if np.any(labels < 0) or np.any(labels >= int(way)):
        raise ValueError("support_labels must be in [0, way)")
    missing = [c for c in range(int(way)) if not np.any(labels == c)]
    if missing:
        raise ValueError(f"classes without support examples: {missing}")
    if weights is None:
        w = np.ones(len(labels), dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim != 1 or len(w) != len(labels):
            raise ValueError("weights must be one-dimensional and match support_features")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
    return support, labels, w


def compute_prototypes(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    *,
    way: int = 5,
    weights: np.ndarray | None = None,
    weight_eps: float = _WEIGHT_EPS,
) -> np.ndarray:
    """Compute normalized class prototypes.

    Each support embedding is normalized first, then class means are computed,
    and each resulting prototype is normalized again. For a class whose supplied
    weights sum to at most ``weight_eps``, all supports in that class receive
    weight one. This is the zero-denominator guard used by PVSE-R-Soft.
    """
    support, labels, w = _validated_inputs(support_features, support_labels, way, weights)
    support = l2_normalize(support, axis=1)
    protos: list[np.ndarray] = []
    for c in range(int(way)):
        idx = labels == c
        wc = w[idx].astype(np.float32, copy=True)
        if float(wc.sum()) <= float(weight_eps):
            wc = np.ones_like(wc, dtype=np.float32)
        proto = (support[idx] * wc[:, None]).sum(axis=0) / max(float(wc.sum()), float(weight_eps))
        protos.append(proto.astype(np.float32, copy=False))
    return l2_normalize(np.stack(protos, axis=0), axis=1)


def prototype_logits(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    *,
    way: int = 5,
    weights: np.ndarray | None = None,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> np.ndarray:
    """Return scaled cosine logits for one or more query embeddings."""
    queries = np.asarray(query_features, dtype=np.float32)
    single = queries.ndim == 1
    if single:
        queries = queries.reshape(1, -1)
    if queries.ndim != 2:
        raise ValueError(f"query_features must be 1-D or 2-D, got {queries.shape}")
    protos = compute_prototypes(
        support_features,
        support_labels,
        way=int(way),
        weights=weights,
    )
    q = l2_normalize(queries, axis=1)
    logits = (float(logit_scale) * (q @ protos.T)).astype(np.float32)
    return logits[0] if single else logits


def prototype_probabilities(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    *,
    way: int = 5,
    weights: np.ndarray | None = None,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> np.ndarray:
    logits = prototype_logits(
        support_features,
        support_labels,
        query_features,
        way=way,
        weights=weights,
        logit_scale=logit_scale,
    )
    return softmax_np(logits, axis=-1).astype(np.float32)


@dataclass(frozen=True)
class PrototypePrediction:
    predictions: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray


def prototype_predict(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    *,
    way: int = 5,
    weights: np.ndarray | None = None,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> PrototypePrediction:
    logits = prototype_logits(
        support_features,
        support_labels,
        query_features,
        way=way,
        weights=weights,
        logit_scale=logit_scale,
    )
    probs = softmax_np(logits, axis=-1).astype(np.float32)
    preds = np.argmax(probs, axis=-1).astype(np.int64)
    return PrototypePrediction(predictions=preds, logits=np.asarray(logits), probabilities=probs)


def weights_from_deleted(n_support: int, deleted: Iterable[int]) -> np.ndarray:
    weights = np.ones(int(n_support), dtype=np.float32)
    deleted_idx = np.asarray(tuple(int(i) for i in deleted), dtype=np.int64)
    if deleted_idx.size:
        if np.any(deleted_idx < 0) or np.any(deleted_idx >= int(n_support)):
            raise IndexError("deleted support index out of range")
        weights[deleted_idx] = 0.0
    return weights
