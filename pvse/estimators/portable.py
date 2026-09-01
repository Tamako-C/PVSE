from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PORTABLE_CLEAN_SCHEMA = "pvse.portable.clean_verifier"
PORTABLE_CLEAN_SCHEMA_VERSION = 1
PORTABLE_BINARY_SCHEMA = "pvse.portable.binary_reliability"
PORTABLE_BINARY_SCHEMA_VERSION = 1
PORTABLE_TWO_HEAD_SCHEMA = "pvse.portable.two_head_gate"
PORTABLE_TWO_HEAD_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


@dataclass(frozen=True)
class PortableBinaryLogisticHead:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray

    def __post_init__(self) -> None:
        feature_count = int(self.mean.size)
        if self.scale.shape != (feature_count,):
            raise ValueError("scaler scale has an invalid shape")
        if self.coef.shape != (1, feature_count):
            raise ValueError("binary logistic coefficient matrix has an invalid shape")
        if self.intercept.shape != (1,):
            raise ValueError("binary logistic intercept has an invalid shape")
        if self.classes.tolist() != [0, 1]:
            raise ValueError("portable clean verifier requires classes [0, 1]")
        arrays = (self.mean, self.scale, self.coef, self.intercept)
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("portable estimator arrays must be finite")
        if np.any(self.scale <= 0):
            raise ValueError("scaler scale must be positive")

    def positive_probability(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError(f"expected feature matrix [N,{self.mean.size}]")
        standardized = (values - self.mean) / self.scale
        logits = standardized @ self.coef[0] + self.intercept[0]
        return _sigmoid(logits)


@dataclass(frozen=True)
class PortableCleanVerifier:
    features: tuple[str, ...]
    help_head: PortableBinaryLogisticHead
    hurt_head: PortableBinaryLogisticHead
    threshold: float
    lambda_hurt: float
    logit_clip_epsilon: float
    logical_role: str
    metadata: Mapping[str, Any]

    def feature_matrix(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        missing = sorted({name for name in self.features if any(name not in row for row in rows)})
        if missing:
            raise KeyError(f"missing clean verifier features: {missing}")
        values = np.asarray([[row[name] for name in self.features] for row in rows], dtype=np.float64)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_probabilities(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.nan_to_num(
            np.asarray(matrix, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return (
            self.help_head.positive_probability(values),
            self.hurt_head.positive_probability(values),
        )

    def score(self, matrix: np.ndarray) -> np.ndarray:
        p_help, p_hurt = self.predict_probabilities(matrix)
        epsilon = float(self.logit_clip_epsilon)
        p_help = np.clip(p_help, epsilon, 1.0 - epsilon)
        p_hurt = np.clip(p_hurt, epsilon, 1.0 - epsilon)
        help_logit = np.log(p_help / (1.0 - p_help))
        hurt_logit = np.log(p_hurt / (1.0 - p_hurt))
        return help_logit - float(self.lambda_hurt) * hurt_logit

    def accepted(self, matrix: np.ndarray) -> np.ndarray:
        return self.score(matrix) > float(self.threshold)


@dataclass(frozen=True)
class PortableReliabilityEstimator:
    features: tuple[str, ...]
    head: PortableBinaryLogisticHead
    logical_role: str
    policy: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def feature_matrix(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        missing = sorted({name for name in self.features if any(name not in row for row in rows)})
        if missing:
            raise KeyError(f"missing reliability features: {missing}")
        values = np.asarray([[row[name] for name in self.features] for row in rows], dtype=np.float64)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_corruption_probability(self, matrix: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(
            np.asarray(matrix, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return self.head.positive_probability(values)

    def support_weights(self, matrix: np.ndarray) -> np.ndarray:
        if "beta" not in self.policy or "w_min" not in self.policy:
            raise ValueError("this estimator does not define a soft-weight policy")
        probability = self.predict_corruption_probability(matrix)
        return np.clip(
            1.0 - float(self.policy["beta"]) * probability,
            float(self.policy["w_min"]),
            1.0,
        )

    def predicted_corrupt(self, matrix: np.ndarray) -> np.ndarray:
        if "threshold" not in self.policy:
            raise ValueError("this estimator does not define a hard-deletion threshold")
        probability = self.predict_corruption_probability(matrix)
        comparison = str(self.policy.get("comparison", ">="))
        if comparison == ">=":
            return probability >= float(self.policy["threshold"])
        if comparison == ">":
            return probability > float(self.policy["threshold"])
        raise ValueError(f"unsupported threshold comparison: {comparison}")


@dataclass(frozen=True)
class PortableTwoHeadGate:
    features: tuple[str, ...]
    help_head: PortableBinaryLogisticHead
    hurt_head: PortableBinaryLogisticHead
    logical_role: str
    score_mode: str
    lambda_hurt: float
    threshold: float
    comparison: str
    logit_clip_epsilon: float
    metadata: Mapping[str, Any]

    def feature_matrix(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        missing = sorted({name for name in self.features if any(name not in row for row in rows)})
        if missing:
            raise KeyError(f"missing gate features: {missing}")
        values = np.asarray([[row[name] for name in self.features] for row in rows], dtype=np.float64)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_probabilities(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.nan_to_num(
            np.asarray(matrix, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return (
            self.help_head.positive_probability(values),
            self.hurt_head.positive_probability(values),
        )

    def score(self, matrix: np.ndarray) -> np.ndarray:
        p_help, p_hurt = self.predict_probabilities(matrix)
        if self.score_mode == "probability_difference":
            return p_help - float(self.lambda_hurt) * p_hurt
        if self.score_mode == "logit_difference":
            epsilon = float(self.logit_clip_epsilon)
            p_help = np.clip(p_help, epsilon, 1.0 - epsilon)
            p_hurt = np.clip(p_hurt, epsilon, 1.0 - epsilon)
            help_logit = np.log(p_help / (1.0 - p_help))
            hurt_logit = np.log(p_hurt / (1.0 - p_hurt))
            return help_logit - float(self.lambda_hurt) * hurt_logit
        raise ValueError(f"unsupported two-head score mode: {self.score_mode}")

    def accepted(self, matrix: np.ndarray) -> np.ndarray:
        score = self.score(matrix)
        if self.comparison == ">=":
            return score >= float(self.threshold)
        if self.comparison == ">":
            return score > float(self.threshold)
        raise ValueError(f"unsupported gate comparison: {self.comparison}")


def _head(arrays: Mapping[str, np.ndarray], prefix: str) -> PortableBinaryLogisticHead:
    return PortableBinaryLogisticHead(
        mean=np.asarray(arrays[f"{prefix}_scaler_mean"], dtype=np.float64),
        scale=np.asarray(arrays[f"{prefix}_scaler_scale"], dtype=np.float64),
        coef=np.asarray(arrays[f"{prefix}_coef"], dtype=np.float64),
        intercept=np.asarray(arrays[f"{prefix}_intercept"], dtype=np.float64),
        classes=np.asarray(arrays[f"{prefix}_classes"], dtype=np.int64),
    )


def load_portable_clean_verifier(metadata_path: str | Path) -> PortableCleanVerifier:
    metadata_file = Path(metadata_path)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("schema") != PORTABLE_CLEAN_SCHEMA:
        raise ValueError("unrecognized portable clean-verifier schema")
    if int(metadata.get("schema_version", -1)) != PORTABLE_CLEAN_SCHEMA_VERSION:
        raise ValueError("unsupported portable clean-verifier schema version")

    arrays_name = str(metadata["arrays_file"])
    if Path(arrays_name).name != arrays_name:
        raise ValueError("portable arrays_file must be a file name in the metadata directory")
    arrays_file = metadata_file.parent / arrays_name
    expected_hash = str(metadata["arrays_sha256"]).lower()
    actual_hash = _sha256_file(arrays_file)
    if actual_hash != expected_hash:
        raise ValueError(
            f"portable array checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )
    with np.load(arrays_file, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}

    features = tuple(str(name) for name in metadata["features"])
    help_head = _head(arrays, "help")
    hurt_head = _head(arrays, "hurt")
    if help_head.mean.size != len(features) or hurt_head.mean.size != len(features):
        raise ValueError("portable head feature count does not match metadata")

    policy = metadata["policy"]
    return PortableCleanVerifier(
        features=features,
        help_head=help_head,
        hurt_head=hurt_head,
        threshold=float(policy["threshold"]),
        lambda_hurt=float(policy["lambda_hurt"]),
        logit_clip_epsilon=float(policy.get("logit_clip_epsilon", 1e-6)),
        logical_role=str(metadata["logical_role"]),
        metadata=metadata,
    )


def _load_arrays(metadata_file: Path, metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays_name = str(metadata["arrays_file"])
    if Path(arrays_name).name != arrays_name:
        raise ValueError("portable arrays_file must be a file name in the metadata directory")
    arrays_file = metadata_file.parent / arrays_name
    expected_hash = str(metadata["arrays_sha256"]).lower()
    actual_hash = _sha256_file(arrays_file)
    if actual_hash != expected_hash:
        raise ValueError(
            f"portable array checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )
    with np.load(arrays_file, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def load_portable_reliability_estimator(
    metadata_path: str | Path,
) -> PortableReliabilityEstimator:
    metadata_file = Path(metadata_path)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("schema") != PORTABLE_BINARY_SCHEMA:
        raise ValueError("unrecognized portable reliability-estimator schema")
    if int(metadata.get("schema_version", -1)) != PORTABLE_BINARY_SCHEMA_VERSION:
        raise ValueError("unsupported portable reliability-estimator schema version")
    arrays = _load_arrays(metadata_file, metadata)
    features = tuple(str(name) for name in metadata["features"])
    head = _head(arrays, "model")
    if head.mean.size != len(features):
        raise ValueError("portable reliability head feature count does not match metadata")
    return PortableReliabilityEstimator(
        features=features,
        head=head,
        logical_role=str(metadata["logical_role"]),
        policy=dict(metadata.get("policy", {})),
        metadata=metadata,
    )


def load_portable_two_head_gate(metadata_path: str | Path) -> PortableTwoHeadGate:
    metadata_file = Path(metadata_path)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("schema") != PORTABLE_TWO_HEAD_SCHEMA:
        raise ValueError("unrecognized portable two-head gate schema")
    if int(metadata.get("schema_version", -1)) != PORTABLE_TWO_HEAD_SCHEMA_VERSION:
        raise ValueError("unsupported portable two-head gate schema version")
    arrays = _load_arrays(metadata_file, metadata)
    features = tuple(str(name) for name in metadata["features"])
    help_head = _head(arrays, "help")
    hurt_head = _head(arrays, "hurt")
    if help_head.mean.size != len(features) or hurt_head.mean.size != len(features):
        raise ValueError("portable two-head feature count does not match metadata")
    policy = metadata["policy"]
    return PortableTwoHeadGate(
        features=features,
        help_head=help_head,
        hurt_head=hurt_head,
        logical_role=str(metadata["logical_role"]),
        score_mode=str(policy["score_mode"]),
        lambda_hurt=float(policy["lambda_hurt"]),
        threshold=float(policy["threshold"]),
        comparison=str(policy.get("comparison", ">=")),
        logit_clip_epsilon=float(policy.get("logit_clip_epsilon", 1e-6)),
        metadata=metadata,
    )


def load_portable_estimator(
    metadata_path: str | Path,
) -> PortableReliabilityEstimator | PortableTwoHeadGate:
    metadata_file = Path(metadata_path)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    schema = metadata.get("schema")
    if schema == PORTABLE_BINARY_SCHEMA:
        return load_portable_reliability_estimator(metadata_file)
    if schema == PORTABLE_TWO_HEAD_SCHEMA:
        return load_portable_two_head_gate(metadata_file)
    raise ValueError(f"unsupported portable estimator schema: {schema}")
