from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pytest

from pvse.estimators.portable import (
    load_portable_clean_verifier,
    load_portable_estimator,
    load_portable_reliability_estimator,
    load_portable_two_head_gate,
)
from pvse.cli.verify_estimator_pack import _verify_checksums


def _write_toy_bundle(tmp_path):
    arrays = tmp_path / "clean_verifier.npz"
    np.savez_compressed(
        arrays,
        help_scaler_mean=np.array([1.0, -1.0]),
        help_scaler_scale=np.array([2.0, 4.0]),
        help_coef=np.array([[0.5, -0.25]]),
        help_intercept=np.array([0.1]),
        help_classes=np.array([0, 1]),
        hurt_scaler_mean=np.array([0.0, 0.0]),
        hurt_scaler_scale=np.array([1.0, 2.0]),
        hurt_coef=np.array([[-0.2, 0.4]]),
        hurt_intercept=np.array([-0.3]),
        hurt_classes=np.array([0, 1]),
    )
    digest = sha256(arrays.read_bytes()).hexdigest()
    metadata = {
        "schema": "pvse.portable.clean_verifier",
        "schema_version": 1,
        "logical_role": "clean.resnet12.table4",
        "arrays_file": arrays.name,
        "arrays_sha256": digest,
        "features": ["f0", "f1"],
        "policy": {
            "threshold": 0.0,
            "lambda_hurt": 2.0,
            "logit_clip_epsilon": 1e-6,
        },
    }
    path = tmp_path / "clean_verifier.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_portable_clean_verifier_math(tmp_path):
    bundle = load_portable_clean_verifier(_write_toy_bundle(tmp_path))
    matrix = np.array([[1.0, -1.0], [3.0, 3.0]], dtype=np.float64)
    p_help, p_hurt = bundle.predict_probabilities(matrix)
    expected_help_logits = np.array([0.1, 0.35])
    expected_hurt_logits = np.array([-0.7, -0.3])
    np.testing.assert_allclose(p_help, 1.0 / (1.0 + np.exp(-expected_help_logits)))
    np.testing.assert_allclose(p_hurt, 1.0 / (1.0 + np.exp(-expected_hurt_logits)))
    np.testing.assert_allclose(bundle.score(matrix), expected_help_logits - 2.0 * expected_hurt_logits)


def test_portable_clean_verifier_mapping_order(tmp_path):
    bundle = load_portable_clean_verifier(_write_toy_bundle(tmp_path))
    rows = [{"f1": -1.0, "f0": 1.0}, {"f1": 3.0, "f0": 3.0}]
    np.testing.assert_allclose(bundle.feature_matrix(rows), [[1.0, -1.0], [3.0, 3.0]])


def test_portable_clean_verifier_rejects_modified_arrays(tmp_path):
    metadata = _write_toy_bundle(tmp_path)
    with (tmp_path / "clean_verifier.npz").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_portable_clean_verifier(metadata)


def test_pack_checksum_verifier_rejects_uncovered_file(tmp_path):
    covered = tmp_path / "covered.txt"
    covered.write_text("covered", encoding="utf-8")
    digest = sha256(covered.read_bytes()).hexdigest()
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    (tmp_path / "SHA256SUMS.txt").write_text(
        f"{digest}  covered.txt\n",
        encoding="utf-8",
    )
    checked, failures = _verify_checksums(tmp_path)
    assert checked == 1
    assert failures == ["uncovered:extra.txt"]


def test_portable_reliability_soft_and_hard_policy(tmp_path):
    arrays = tmp_path / "reliability.npz"
    np.savez_compressed(
        arrays,
        model_scaler_mean=np.array([0.0, 1.0]),
        model_scaler_scale=np.array([1.0, 2.0]),
        model_coef=np.array([[0.5, -1.0]]),
        model_intercept=np.array([0.25]),
        model_classes=np.array([0, 1]),
    )
    metadata = {
        "schema": "pvse.portable.binary_reliability",
        "schema_version": 1,
        "logical_role": "noisy.test.reliability",
        "arrays_file": arrays.name,
        "arrays_sha256": sha256(arrays.read_bytes()).hexdigest(),
        "features": ["f0", "f1"],
        "policy": {"beta": 1.5, "w_min": 0.2, "threshold": 0.5, "comparison": ">="},
    }
    path = tmp_path / "reliability.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    estimator = load_portable_reliability_estimator(path)
    assert load_portable_estimator(path).logical_role == estimator.logical_role
    matrix = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float64)
    logits = np.array([0.25, 2.25])
    probability = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(estimator.predict_corruption_probability(matrix), probability)
    np.testing.assert_allclose(estimator.support_weights(matrix), np.clip(1.0 - 1.5 * probability, 0.2, 1.0))
    np.testing.assert_array_equal(estimator.predicted_corrupt(matrix), probability >= 0.5)


def test_portable_two_head_probability_difference(tmp_path):
    arrays = tmp_path / "gate.npz"
    np.savez_compressed(
        arrays,
        help_scaler_mean=np.array([0.0]),
        help_scaler_scale=np.array([1.0]),
        help_coef=np.array([[1.0]]),
        help_intercept=np.array([0.0]),
        help_classes=np.array([0, 1]),
        hurt_scaler_mean=np.array([0.0]),
        hurt_scaler_scale=np.array([1.0]),
        hurt_coef=np.array([[-1.0]]),
        hurt_intercept=np.array([0.0]),
        hurt_classes=np.array([0, 1]),
    )
    metadata = {
        "schema": "pvse.portable.two_head_gate",
        "schema_version": 1,
        "logical_role": "clean.test.gate",
        "arrays_file": arrays.name,
        "arrays_sha256": sha256(arrays.read_bytes()).hexdigest(),
        "features": ["f0"],
        "policy": {
            "score_mode": "probability_difference",
            "lambda_hurt": 1.5,
            "threshold": 0.0,
            "comparison": ">=",
        },
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    gate = load_portable_two_head_gate(path)
    assert load_portable_estimator(path).logical_role == gate.logical_role
    matrix = np.array([[-2.0], [2.0]], dtype=np.float64)
    p_help = 1.0 / (1.0 + np.exp(-matrix[:, 0]))
    p_hurt = 1.0 / (1.0 + np.exp(matrix[:, 0]))
    expected = p_help - 1.5 * p_hurt
    np.testing.assert_allclose(gate.score(matrix), expected)
    np.testing.assert_array_equal(gate.accepted(matrix), expected >= 0.0)
