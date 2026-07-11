from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

TraNFSProtocol = Literal["sym_swap", "pair_swap"]


def normalize_protocol_name(protocol: str) -> TraNFSProtocol:
    aliases = {
        "symmetric_label_swap": "sym_swap",
        "paired_label_swap": "pair_swap",
        "symmetric": "sym_swap",
        "paired": "pair_swap",
        "sym": "sym_swap",
        "pair": "pair_swap",
    }
    normalized = aliases.get(str(protocol).strip(), str(protocol).strip())
    if normalized not in {"sym_swap", "pair_swap"}:
        raise ValueError(f"unsupported TraNFS-style protocol: {protocol}")
    return normalized  # type: ignore[return-value]


def severity_to_per_class(severity: int | str) -> int:
    value = int(severity)
    if value not in {0, 20, 40, 60, 80}:
        raise ValueError(f"unsupported severity: {severity}")
    return value // 20


def gen_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random derangement using the official TraNFS rejection rule."""
    in_order = np.arange(int(n), dtype=np.int16)
    while True:
        candidate = rng.permutation(int(n)).astype(np.int16)
        if 0 not in candidate - in_order:
            return candidate


def gen_swap_indices(
    ways: int,
    num_noise_samples: int,
    num_clean_shots: int,
    indices_to_change: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """NumPy implementation of the TraNFS-style swap-index protocol."""
    ways = int(ways)
    shot = int(num_noise_samples) // ways
    available = np.arange(int(num_noise_samples))
    available_kn = available.reshape(ways, shot)
    swap_indices: dict[int, list[int]] = {}
    class_order = np.arange(ways)
    rng.shuffle(class_order)
    for class_id in class_order:
        c = int(class_id)
        available_c = [int(i) for i in available if i not in available_kn[c]]
        noise_class_count = np.zeros(ways)
        selected: list[int] = []
        for _ in range(len(indices_to_change[c])):
            if not available_c:
                return None
            choice = int(rng.choice(np.asarray(available_c, dtype=np.int64)))
            selected.append(choice)
            available_c.remove(choice)
            choice_class = choice // shot
            noise_class_count[choice_class] += 1
            if (noise_class_count[choice_class] + 1) >= int(num_clean_shots):
                available_c = [int(i) for i in available_c if i not in available_kn[choice_class]]
        swap_indices[c] = selected
        available = np.setdiff1d(available, np.asarray(selected, dtype=np.int64))
    return np.vstack([swap_indices[c] for c in range(ways)]).astype(np.int16)


def gen_valid_swap_indices(
    ways: int,
    shot: int,
    indices_to_change: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    num_noise_samples = int(ways) * int(shot)
    num_clean_shots = int(shot) - len(indices_to_change[0])
    while True:
        swap = gen_swap_indices(
            int(ways),
            num_noise_samples,
            num_clean_shots,
            indices_to_change,
            rng,
        )
        if swap is not None:
            return swap


@dataclass(frozen=True)
class TraNFSCorruption:
    support_features: np.ndarray
    support_maps: torch.Tensor
    support_labels: np.ndarray
    corrupt_mask: np.ndarray
    metadata: tuple[dict[str, Any], ...]


def apply_tranfs_swap(
    support_pool_features: np.ndarray,
    support_pool_maps: torch.Tensor,
    *,
    protocol: str,
    severity: int | str,
    rng: np.random.Generator,
    ways: int = 5,
    shot: int = 5,
) -> TraNFSCorruption:
    """Apply the TraNFS-style sym/pair swap protocol used in submitted Table 8.

    Input pool layout is class-major with ``2*shot`` examples per class: the
    first ``shot`` examples are clean support and the following ``shot`` form
    the extra noise pool. Replaced samples retain the target support position's
    observed class label for downstream prototype grouping.
    """
    protocol_name = normalize_protocol_name(protocol)
    ways = int(ways)
    shot = int(shot)
    noise_num = severity_to_per_class(severity)
    if noise_num > shot:
        raise ValueError("severity requests more corruptions than supports per class")

    features = np.asarray(support_pool_features, dtype=np.float32)
    expected = ways * 2 * shot
    if features.ndim != 2 or len(features) != expected:
        raise ValueError(f"support_pool_features must have shape ({expected}, D)")
    if support_pool_maps.shape[0] != expected:
        raise ValueError(f"support_pool_maps first dimension must be {expected}")

    support_indices = np.concatenate(
        [np.arange(c * 2 * shot, c * 2 * shot + shot) for c in range(ways)]
    )
    noise_indices = np.concatenate(
        [np.arange(c * 2 * shot + shot, c * 2 * shot + 2 * shot) for c in range(ways)]
    )
    noisy_support = features[support_indices].copy()
    noisy_maps = support_pool_maps[support_indices].clone()
    labels = np.repeat(np.arange(ways), shot).astype(np.int64)
    corrupt_mask = np.zeros(ways * shot, dtype=bool)

    indices_to_change = np.empty((0, noise_num), dtype=np.int16)
    for c in range(ways):
        class_local = np.arange(c * shot, (c + 1) * shot, dtype=np.int16)
        selected = rng.choice(class_local, noise_num, replace=False).astype(np.int16)
        indices_to_change = np.vstack((indices_to_change, selected))

    metadata: list[dict[str, Any]] = []
    if noise_num == 0:
        return TraNFSCorruption(noisy_support, noisy_maps, labels, corrupt_mask, tuple(metadata))

    if protocol_name == "sym_swap":
        swap = gen_valid_swap_indices(ways, shot, indices_to_change, rng)
        for c in range(ways):
            for j, target_local in enumerate(indices_to_change[c].astype(int).tolist()):
                source_noise_flat = int(swap[c, j])
                source_class = source_noise_flat // shot
                source_slot = source_noise_flat % shot
                source_pool_idx = int(noise_indices[source_noise_flat])
                noisy_support[target_local] = features[source_pool_idx]
                noisy_maps[target_local] = support_pool_maps[source_pool_idx]
                corrupt_mask[target_local] = True
                metadata.append(
                    {
                        "support_index": int(target_local),
                        "target_support_index": int(target_local),
                        "target_class": int(c),
                        "target_within_class_slot": int(target_local - c * shot),
                        "source_pool_index": source_pool_idx,
                        "source_noise_flat_index": source_noise_flat,
                        "source_class": int(source_class),
                        "source_within_class_slot": int(source_slot),
                        "protocol": protocol_name,
                        "severity": int(severity),
                        "corruption_mode": "official_tranfs_sym_swap",
                        "operation": "copy/replacement_from_extra_noise_pool",
                        "support_group_label_after_corruption": int(c),
                        "copied_data_label_for_audit": int(source_class),
                        "allow_same_class_source": False,
                        "source_class_in_episode": True,
                    }
                )
    else:
        paired_class = gen_derangement(ways, rng)
        for c in range(ways):
            source_class = int(paired_class[c])
            for j, target_local in enumerate(indices_to_change[c].astype(int).tolist()):
                source_noise_flat = source_class * shot + j
                source_pool_idx = int(noise_indices[source_noise_flat])
                noisy_support[target_local] = features[source_pool_idx]
                noisy_maps[target_local] = support_pool_maps[source_pool_idx]
                corrupt_mask[target_local] = True
                metadata.append(
                    {
                        "support_index": int(target_local),
                        "target_support_index": int(target_local),
                        "target_class": int(c),
                        "target_within_class_slot": int(target_local - c * shot),
                        "source_pool_index": source_pool_idx,
                        "source_noise_flat_index": int(source_noise_flat),
                        "source_class": source_class,
                        "source_within_class_slot": int(j),
                        "protocol": protocol_name,
                        "severity": int(severity),
                        "corruption_mode": "official_tranfs_pair_swap",
                        "operation": "copy/replacement_from_extra_noise_pool",
                        "support_group_label_after_corruption": int(c),
                        "copied_data_label_for_audit": source_class,
                        "allow_same_class_source": False,
                        "source_class_in_episode": True,
                    }
                )

    return TraNFSCorruption(noisy_support, noisy_maps, labels, corrupt_mask, tuple(metadata))
