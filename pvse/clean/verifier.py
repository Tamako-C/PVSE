from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from pvse.clean.features import CLEAN_VERIFIER_FEATURES, feature_matrix


@dataclass(frozen=True)
class CleanVerifierConfig:
    penalty: str = "l1"
    solver: str = "liblinear"
    C: float = 0.5
    class_weight: str = "balanced"
    max_iter: int = 1000
    lambda_hurt: float = 2.0
    threshold: float = 5.35311817754396
    feature_set: str = "patch_plus_proposal"
    proposal_source: str = "runnerup_a0"
    random_state: int | None = None


@dataclass
class CleanVerifierBundle:
    help_model: Pipeline
    hurt_model: Pipeline
    features: tuple[str, ...]
    config: CleanVerifierConfig

    def predict_probabilities(self, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        x = _mapping_matrix(rows, self.features)
        p_help = _class_one_probability(self.help_model, x)
        p_hurt = _class_one_probability(self.hurt_model, x)
        return p_help, p_hurt

    def score(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        p_help, p_hurt = self.predict_probabilities(rows)
        return safe_logit(p_help) - float(self.config.lambda_hurt) * safe_logit(p_hurt)

    def accepted(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return self.score(rows) > float(self.config.threshold)


def _new_head(config: CleanVerifierConfig) -> Pipeline:
    return make_pipeline(
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


def _class_one_probability(model: Pipeline, x: np.ndarray) -> np.ndarray:
    p = model.predict_proba(x)
    classes = np.asarray(model.classes_)
    pos = np.flatnonzero(classes == 1)
    if len(pos) != 1:
        return np.zeros(len(x), dtype=np.float64)
    return p[:, int(pos[0])].astype(np.float64)


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _mapping_matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    missing = sorted({name for name in features if any(name not in row for row in rows)})
    if missing:
        raise KeyError(f"missing clean verifier features: {missing}")
    x = np.asarray([[row[name] for name in features] for row in rows], dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def fit_clean_verifier(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: CleanVerifierConfig = CleanVerifierConfig(),
    features: Sequence[str] = CLEAN_VERIFIER_FEATURES,
) -> CleanVerifierBundle:
    if not rows:
        raise ValueError("at least one calibration row is required")
    x = _mapping_matrix(rows, features)
    y_help = np.asarray([int(row["help_label"]) for row in rows], dtype=np.int64)
    y_hurt = np.asarray([int(row["hurt_label"]) for row in rows], dtype=np.int64)
    if len(np.unique(y_help)) < 2:
        raise ValueError("help labels contain fewer than two classes")
    if len(np.unique(y_hurt)) < 2:
        raise ValueError("hurt labels contain fewer than two classes")
    help_model = _new_head(config)
    hurt_model = _new_head(config)
    help_model.fit(x, y_help)
    hurt_model.fit(x, y_hurt)
    return CleanVerifierBundle(
        help_model=help_model,
        hurt_model=hurt_model,
        features=tuple(str(f) for f in features),
        config=config,
    )


def save_clean_verifier(bundle: CleanVerifierBundle, path: str | Path) -> None:
    payload = {
        "format": "pvse_clean_verifier_v1",
        "features": list(bundle.features),
        "config": asdict(bundle.config),
        "help_model": bundle.help_model,
        "hurt_model": bundle.hurt_model,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_clean_verifier(path: str | Path) -> CleanVerifierBundle:
    payload = joblib.load(Path(path))
    if not isinstance(payload, dict) or payload.get("format") != "pvse_clean_verifier_v1":
        raise ValueError("unrecognized clean verifier bundle")
    return CleanVerifierBundle(
        help_model=payload["help_model"],
        hurt_model=payload["hurt_model"],
        features=tuple(payload["features"]),
        config=CleanVerifierConfig(**payload["config"]),
    )
