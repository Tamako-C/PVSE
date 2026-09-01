from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from pvse.noisy.features import RELIABILITY_FEATURES


@dataclass(frozen=True)
class ReliabilityConfig:
    severity_percent: int
    solver: str
    penalty: str
    C: float
    beta: float
    w_min: float
    class_weight: str = "balanced"
    max_iter: int = 1000
    feature_set: str = "global_plus_patch"
    random_state: int | None = 3920


PAPER_RELIABILITY_CONFIGS: dict[int, ReliabilityConfig] = {
    20: ReliabilityConfig(20, solver="liblinear", penalty="l1", C=0.5, beta=1.5, w_min=0.0),
    40: ReliabilityConfig(40, solver="lbfgs", penalty="l2", C=1.0, beta=1.5, w_min=0.5),
    60: ReliabilityConfig(60, solver="liblinear", penalty="l1", C=0.5, beta=1.0, w_min=0.1),
}


@dataclass
class ReliabilityBundle:
    model: Pipeline
    features: tuple[str, ...]
    config: ReliabilityConfig

    def predict_corruption_probability(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        x = _mapping_matrix(rows, self.features)
        p = self.model.predict_proba(x)
        classes = np.asarray(self.model.classes_)
        pos = np.flatnonzero(classes == 1)
        if len(pos) != 1:
            return np.zeros(len(x), dtype=np.float32)
        return p[:, int(pos[0])].astype(np.float32)

    def support_weights(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return reliability_weights(
            self.predict_corruption_probability(rows),
            beta=float(self.config.beta),
            w_min=float(self.config.w_min),
        )


def _mapping_matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    missing = sorted({name for name in features if any(name not in row for row in rows)})
    if missing:
        raise KeyError(f"missing reliability features: {missing}")
    x = np.asarray([[row[name] for name in features] for row in rows], dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def fit_reliability_estimator(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: ReliabilityConfig,
    features: Sequence[str] = RELIABILITY_FEATURES,
) -> ReliabilityBundle:
    if not rows:
        raise ValueError("at least one training row is required")
    x = _mapping_matrix(rows, features)
    y = np.asarray([int(row["corrupted_for_analysis_only"]) for row in rows], dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("corruption labels contain fewer than two classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=int(config.max_iter),
            class_weight=config.class_weight,
            penalty=config.penalty,
            solver=config.solver,
            C=float(config.C),
            random_state=config.random_state,
        ),
    )
    model.fit(x, y)
    return ReliabilityBundle(model=model, features=tuple(str(f) for f in features), config=config)


def reliability_weights(prob_corrupt: np.ndarray, *, beta: float, w_min: float) -> np.ndarray:
    p = np.asarray(prob_corrupt, dtype=np.float32)
    if not np.all(np.isfinite(p)):
        raise ValueError("prob_corrupt must be finite")
    return np.clip(1.0 - float(beta) * p, float(w_min), 1.0).astype(np.float32)


def save_reliability_bundle(bundle: ReliabilityBundle, path: str | Path) -> None:
    payload = {
        "format": "pvse_reliability_v1",
        "features": list(bundle.features),
        "config": asdict(bundle.config),
        "model": bundle.model,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_reliability_bundle(path: str | Path) -> ReliabilityBundle:
    payload = joblib.load(Path(path))
    if not isinstance(payload, dict) or payload.get("format") != "pvse_reliability_v1":
        raise ValueError("unrecognized reliability bundle")
    return ReliabilityBundle(
        model=payload["model"],
        features=tuple(payload["features"]),
        config=ReliabilityConfig(**payload["config"]),
    )
