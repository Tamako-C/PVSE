from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    estimate_pp: float
    lower_pp: float
    upper_pp: float
    samples: int
    seed: int
    resampling_unit: str = "episode"

    def to_dict(self) -> dict:
        return asdict(self)


def paired_episode_bootstrap(
    episode_ids: np.ndarray,
    baseline_correct: np.ndarray,
    method_correct: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 5200,
) -> BootstrapInterval:
    """Paired episode bootstrap used for all primary CMT confidence intervals."""
    episodes = np.asarray(episode_ids)
    baseline = np.asarray(baseline_correct, dtype=np.float64)
    method = np.asarray(method_correct, dtype=np.float64)
    if episodes.ndim != 1 or baseline.ndim != 1 or method.ndim != 1:
        raise ValueError("bootstrap inputs must be one-dimensional")
    if not (len(episodes) == len(baseline) == len(method)):
        raise ValueError("bootstrap inputs must have equal length")
    unique = np.unique(episodes)
    if len(unique) == 0:
        raise ValueError("no episodes supplied")

    episode_gain: list[float] = []
    episode_query_count: list[int] = []
    for episode in unique:
        mask = episodes == episode
        episode_gain.append(float((method[mask] - baseline[mask]).sum()))
        episode_query_count.append(int(mask.sum()))
    gain = np.asarray(episode_gain, dtype=np.float64)
    count = np.asarray(episode_query_count, dtype=np.float64)

    estimate = float(100.0 * gain.sum() / count.sum())
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    for i in range(int(samples)):
        sampled = rng.integers(0, len(unique), size=len(unique))
        draws[i] = 100.0 * gain[sampled].sum() / count[sampled].sum()
    lower, upper = np.quantile(draws, [0.025, 0.975]).tolist()
    return BootstrapInterval(
        estimate_pp=estimate,
        lower_pp=float(lower),
        upper_pp=float(upper),
        samples=int(samples),
        seed=int(seed),
    )
