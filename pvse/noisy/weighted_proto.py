from __future__ import annotations

import numpy as np

from pvse.core.prototypes import DEFAULT_LOGIT_SCALE, PrototypePrediction, prototype_predict


def weighted_proto_predict(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    weights: np.ndarray,
    *,
    way: int = 5,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> PrototypePrediction:
    return prototype_predict(
        support_features,
        support_labels,
        query_features,
        way=int(way),
        weights=np.asarray(weights, dtype=np.float32),
        logit_scale=float(logit_scale),
    )
