from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pvse.core.prototypes import DEFAULT_LOGIT_SCALE, prototype_logits, softmax_np, weights_from_deleted
from pvse.editing.delete_lattice import enumerate_delete_actions

DeletionScope = Literal["a0_only", "full"]


@dataclass(frozen=True)
class DeleteActionEvaluation:
    deleted: tuple[int, ...]
    prediction: int
    logits: np.ndarray
    probabilities: np.ndarray
    proposal_margin: float
    action_score: float


@dataclass(frozen=True)
class StrictDeleteDecision:
    a0_prediction: int
    proposal: int
    scope: DeletionScope
    selected_action: tuple[int, ...]
    selected_k: int
    selected_action_prediction: int
    action_score: float
    proposal_margin: float
    reached_proposal: bool
    applied: bool
    final_prediction: int


def evaluate_delete_action(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_feature: np.ndarray,
    *,
    deleted: tuple[int, ...],
    a0_prediction: int,
    proposal: int,
    edit_cost: float = 0.0,
    way: int = 5,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> DeleteActionEvaluation:
    weights = weights_from_deleted(len(support_labels), deleted)
    logits = np.asarray(
        prototype_logits(
            support_features,
            support_labels,
            query_feature,
            way=way,
            weights=weights,
            logit_scale=logit_scale,
        ),
        dtype=np.float32,
    )
    probs = softmax_np(logits, axis=0).astype(np.float32)
    pred = int(np.argmax(logits))
    margin = float(logits[int(proposal)] - logits[int(a0_prediction)])
    score = float(margin - float(edit_cost) * len(deleted))
    return DeleteActionEvaluation(
        deleted=tuple(int(i) for i in deleted),
        prediction=pred,
        logits=logits,
        probabilities=probs,
        proposal_margin=margin,
        action_score=score,
    )


def choose_strict_hard_delete(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_feature: np.ndarray,
    *,
    a0_prediction: int,
    proposal: int,
    scope: DeletionScope = "a0_only",
    kmax: int = 3,
    edit_cost: float = 0.0,
    way: int = 5,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> StrictDeleteDecision:
    """Select the paper's hard-delete action and apply its strict realization rule.

    The action is selected by the raw scaled-logit proposal margin minus the
    deletion cost. All candidate actions are scored first. Only after selecting
    the highest-scoring action do we check whether its prediction equals the
    verified proposal. If it does not, the method falls back to A0; it does not
    search for a lower-scoring feasible action. Ties retain the first action in
    ``k=0,1,...`` / lexicographic enumeration order, matching the submitted action-ordering protocol.
    """
    labels = np.asarray(support_labels, dtype=np.int64)
    if int(a0_prediction) == int(proposal):
        raise ValueError("proposal must differ from A0 prediction")
    if scope == "a0_only":
        candidates = np.flatnonzero(labels == int(a0_prediction)).tolist()
    elif scope == "full":
        candidates = list(range(len(labels)))
    else:
        raise ValueError(f"unknown deletion scope: {scope}")

    best: DeleteActionEvaluation | None = None
    for deleted in enumerate_delete_actions(candidates, kmax=int(kmax)):
        current = evaluate_delete_action(
            support_features,
            labels,
            query_feature,
            deleted=deleted,
            a0_prediction=int(a0_prediction),
            proposal=int(proposal),
            edit_cost=float(edit_cost),
            way=int(way),
            logit_scale=float(logit_scale),
        )
        if best is None or current.action_score > best.action_score:
            best = current
    if best is None:  # defensive; k=0 always exists
        raise RuntimeError("no delete action was enumerated")
    reached = best.prediction == int(proposal)
    final = int(proposal) if reached else int(a0_prediction)
    return StrictDeleteDecision(
        a0_prediction=int(a0_prediction),
        proposal=int(proposal),
        scope=scope,
        selected_action=best.deleted,
        selected_k=len(best.deleted),
        selected_action_prediction=best.prediction,
        action_score=best.action_score,
        proposal_margin=best.proposal_margin,
        reached_proposal=bool(reached),
        applied=bool(reached),
        final_prediction=final,
    )


def verified_switch(a0_prediction: int, proposal: int, accepted: bool) -> int:
    return int(proposal) if bool(accepted) else int(a0_prediction)
