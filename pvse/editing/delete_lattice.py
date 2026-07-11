from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DeleteActionBank:
    actions: tuple[tuple[int, ...], ...]
    keep_masks: np.ndarray
    k_values: np.ndarray
    single_delete_action_ids: dict[int, int]

    def __len__(self) -> int:
        return len(self.actions)


def enumerate_delete_actions(indices: Iterable[int], kmax: int = 3) -> tuple[tuple[int, ...], ...]:
    items = tuple(int(i) for i in indices)
    if len(set(items)) != len(items):
        raise ValueError("indices must be unique")
    if int(kmax) < 0:
        raise ValueError("kmax must be non-negative")
    actions: list[tuple[int, ...]] = []
    for k in range(min(int(kmax), len(items)) + 1):
        actions.extend(tuple(int(x) for x in action) for action in combinations(items, k))
    return tuple(actions)


def action_count(n_candidates: int, kmax: int = 3) -> int:
    return sum(comb(int(n_candidates), k) for k in range(min(int(kmax), int(n_candidates)) + 1))


def build_action_bank(n_support: int = 25, kmax: int = 3) -> DeleteActionBank:
    actions = enumerate_delete_actions(range(int(n_support)), kmax=int(kmax))
    keep_masks: list[np.ndarray] = []
    k_values: list[int] = []
    single_delete: dict[int, int] = {}
    for action_id, dropped in enumerate(actions):
        keep = np.ones(int(n_support), dtype=np.float32)
        if dropped:
            keep[np.asarray(dropped, dtype=np.int64)] = 0.0
        keep_masks.append(keep)
        k_values.append(len(dropped))
        if len(dropped) == 1:
            single_delete[dropped[0]] = int(action_id)
    return DeleteActionBank(
        actions=actions,
        keep_masks=np.stack(keep_masks, axis=0),
        k_values=np.asarray(k_values, dtype=np.int64),
        single_delete_action_ids=single_delete,
    )


def keep_mask(n_support: int, deleted: Sequence[int]) -> np.ndarray:
    mask = np.ones(int(n_support), dtype=bool)
    if deleted:
        idx = np.asarray(tuple(int(i) for i in deleted), dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= int(n_support)):
            raise IndexError("deleted support index out of range")
        mask[idx] = False
    return mask
