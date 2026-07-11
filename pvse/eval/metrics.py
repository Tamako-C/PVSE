from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PairedClassificationMetrics:
    query_count: int
    baseline_accuracy: float
    method_accuracy: float
    gain_pp: float
    help: int
    hurt: int
    net: int
    precision: float
    safe_total: int
    rescue_total: int
    safe_damage_rate: float
    rescue_hit_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def paired_classification_metrics(
    true_labels: np.ndarray,
    baseline_predictions: np.ndarray,
    method_predictions: np.ndarray,
) -> PairedClassificationMetrics:
    y = np.asarray(true_labels, dtype=np.int64)
    base = np.asarray(baseline_predictions, dtype=np.int64)
    method = np.asarray(method_predictions, dtype=np.int64)
    if y.shape != base.shape or y.shape != method.shape:
        raise ValueError("true_labels, baseline_predictions, and method_predictions must have equal shape")
    if y.ndim != 1:
        raise ValueError("classification inputs must be one-dimensional")
    base_correct = base == y
    method_correct = method == y
    help_mask = (~base_correct) & method_correct
    hurt_mask = base_correct & (~method_correct)
    help_count = int(help_mask.sum())
    hurt_count = int(hurt_mask.sum())
    safe_total = int(base_correct.sum())
    rescue_total = int((~base_correct).sum())
    return PairedClassificationMetrics(
        query_count=int(len(y)),
        baseline_accuracy=float(base_correct.mean()) if len(y) else float("nan"),
        method_accuracy=float(method_correct.mean()) if len(y) else float("nan"),
        gain_pp=float(100.0 * (method_correct.mean() - base_correct.mean())) if len(y) else float("nan"),
        help=help_count,
        hurt=hurt_count,
        net=help_count - hurt_count,
        precision=float(help_count / max(1, help_count + hurt_count)),
        safe_total=safe_total,
        rescue_total=rescue_total,
        safe_damage_rate=float(hurt_count / max(1, safe_total)),
        rescue_hit_rate=float(help_count / max(1, rescue_total)),
    )


@dataclass(frozen=True)
class SupportRecoveryMetrics:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    selected: int
    corrupted: int

    def to_dict(self) -> dict:
        return asdict(self)


def support_recovery_metrics(
    selected_as_corrupt: np.ndarray,
    true_corrupt: np.ndarray,
) -> SupportRecoveryMetrics:
    selected = np.asarray(selected_as_corrupt, dtype=bool)
    truth = np.asarray(true_corrupt, dtype=bool)
    if selected.shape != truth.shape or selected.ndim != 1:
        raise ValueError("selected_as_corrupt and true_corrupt must be matching one-dimensional arrays")
    tp = int((selected & truth).sum())
    fp = int((selected & ~truth).sum())
    fn = int((~selected & truth).sum())
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2 * precision * recall / max(1e-12, precision + recall))
    return SupportRecoveryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        selected=int(selected.sum()),
        corrupted=int(truth.sum()),
    )


def mean_episode_support_recovery(
    selected_masks: list[np.ndarray],
    true_masks: list[np.ndarray],
) -> dict[str, float]:
    if len(selected_masks) != len(true_masks) or not selected_masks:
        raise ValueError("selected_masks and true_masks must be non-empty and equally sized")
    metrics = [support_recovery_metrics(s, t) for s, t in zip(selected_masks, true_masks)]
    return {
        "support_precision": float(np.mean([m.precision for m in metrics])),
        "support_recall": float(np.mean([m.recall for m in metrics])),
        "support_F1": float(np.mean([m.f1 for m in metrics])),
    }
