from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

CorruptionMode = Literal["per_class", "total"]


@dataclass(frozen=True)
class CorruptionRecord:
    support_index: int
    original_label: int
    source_index: int
    source_label: int
    corruption_mode: str = "within_episode_other_class_copy_replacement"


@dataclass(frozen=True)
class CorruptedSupport:
    features: np.ndarray
    maps: torch.Tensor | None
    corrupt_mask: np.ndarray
    records: tuple[CorruptionRecord, ...]


def _select_per_class_targets(
    labels: np.ndarray,
    corrupt_per_class: int,
    rng: np.random.Generator,
) -> list[int]:
    targets: list[int] = []
    count = int(corrupt_per_class)
    if count < 0:
        raise ValueError("corrupt_per_class must be non-negative")
    for c in sorted(set(labels.astype(int).tolist())):
        idx = np.flatnonzero(labels == c)
        if count > len(idx):
            raise ValueError(f"cannot corrupt {count} supports in class {c} with only {len(idx)} examples")
        if count == 0:
            continue
        # The scalar draw for the 20% protocol is intentionally kept distinct:
        # it reproduces the original NumPy Generator call sequence exactly.
        if count == 1:
            targets.append(int(rng.choice(idx)))
        else:
            targets.extend(rng.choice(idx, size=count, replace=False).astype(int).tolist())
    return targets


def _select_total_targets(labels: np.ndarray, corrupt_total: int, rng: np.random.Generator) -> list[int]:
    count = int(corrupt_total)
    if count < 0 or count > len(labels):
        raise ValueError("corrupt_total is outside the valid range")
    if count == 0:
        return []
    return rng.choice(np.arange(len(labels)), size=count, replace=False).astype(int).tolist()


def apply_main_corruption(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    *,
    rng: np.random.Generator,
    corrupt_per_class: int | None = None,
    corrupt_total: int | None = None,
    support_maps: torch.Tensor | None = None,
) -> CorruptedSupport:
    """Apply the paper's within-episode other-class copy/replacement protocol.

    Exactly one of ``corrupt_per_class`` or ``corrupt_total`` must be supplied.
    All target slots are sampled first; only then are source supports drawn.
    The target's observed label is unchanged and the source remains at its
    original slot. Source draws are independent and may repeat.
    """
    if (corrupt_per_class is None) == (corrupt_total is None):
        raise ValueError("supply exactly one of corrupt_per_class or corrupt_total")
    clean = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels, dtype=np.int64)
    if clean.ndim != 2 or labels.ndim != 1 or len(clean) != len(labels):
        raise ValueError("support_features/support_labels shape mismatch")
    if support_maps is not None and (support_maps.ndim != 4 or len(support_maps) != len(labels)):
        raise ValueError("support_maps must have shape [N,C,H,W]")

    if corrupt_per_class is not None:
        targets = _select_per_class_targets(labels, int(corrupt_per_class), rng)
    else:
        targets = _select_total_targets(labels, int(corrupt_total), rng)

    noisy = clean.copy()
    noisy_maps = support_maps.clone() if support_maps is not None else None
    mask = np.zeros(len(labels), dtype=bool)
    records: list[CorruptionRecord] = []
    # Source sampling starts only after every target has been fixed.
    for target in targets:
        source_pool = np.flatnonzero(labels != labels[target])
        if len(source_pool) == 0:
            raise ValueError("another observed class is required for corruption")
        source = int(rng.choice(source_pool))
        noisy[target] = clean[source]
        if noisy_maps is not None:
            noisy_maps[target] = support_maps[source]
        mask[target] = True
        records.append(
            CorruptionRecord(
                support_index=int(target),
                original_label=int(labels[target]),
                source_index=source,
                source_label=int(labels[source]),
            )
        )
    return CorruptedSupport(
        features=noisy,
        maps=noisy_maps,
        corrupt_mask=mask,
        records=tuple(records),
    )
