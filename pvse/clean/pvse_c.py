from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from pvse.clean.verifier import CleanVerifierBundle
from pvse.core.prototypes import DEFAULT_LOGIT_SCALE
from pvse.editing.counterfactual import StrictDeleteDecision, choose_strict_hard_delete


@dataclass(frozen=True)
class CleanQueryDecision:
    query_index: int
    a0_prediction: int
    proposal: int
    verifier_score: float
    accepted: bool
    switch_prediction: int
    hard_delete_a0_only: StrictDeleteDecision | None
    hard_delete_full: StrictDeleteDecision | None


def apply_clean_episode(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    verifier: CleanVerifierBundle,
    *,
    run_a0_only: bool = True,
    run_full: bool = True,
    kmax: int = 3,
    edit_cost: float = 0.0,
    way: int = 5,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> list[CleanQueryDecision]:
    if len(rows) != len(query_features):
        raise ValueError("one verifier row is required for each query")
    scores = verifier.score(rows)
    accepted = scores > float(verifier.config.threshold)
    decisions: list[CleanQueryDecision] = []
    for i, row in enumerate(rows):
        a0 = int(row["a0_pred"])
        proposal = int(row["proposal_class"])
        is_accepted = bool(accepted[i])
        a0_only = None
        full = None
        if is_accepted and run_a0_only:
            a0_only = choose_strict_hard_delete(
                support_features,
                support_labels,
                query_features[i],
                a0_prediction=a0,
                proposal=proposal,
                scope="a0_only",
                kmax=int(kmax),
                edit_cost=float(edit_cost),
                way=int(way),
                logit_scale=float(logit_scale),
            )
        if is_accepted and run_full:
            full = choose_strict_hard_delete(
                support_features,
                support_labels,
                query_features[i],
                a0_prediction=a0,
                proposal=proposal,
                scope="full",
                kmax=int(kmax),
                edit_cost=float(edit_cost),
                way=int(way),
                logit_scale=float(logit_scale),
            )
        decisions.append(
            CleanQueryDecision(
                query_index=int(row.get("query_index", i)),
                a0_prediction=a0,
                proposal=proposal,
                verifier_score=float(scores[i]),
                accepted=is_accepted,
                switch_prediction=proposal if is_accepted else a0,
                hard_delete_a0_only=a0_only,
                hard_delete_full=full,
            )
        )
    return decisions
