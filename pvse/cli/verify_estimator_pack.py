from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from pvse.estimators import (
    PortableReliabilityEstimator,
    PortableTwoHeadGate,
    load_portable_clean_verifier,
    load_portable_estimator,
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_checksums(root: Path) -> tuple[int, list[str]]:
    checksum_file = root / "SHA256SUMS.txt"
    failures: list[str] = []
    if not checksum_file.is_file():
        return 0, ["missing:SHA256SUMS.txt"]
    checked = 0
    covered: set[str] = set()
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(None, 1)
        if len(fields) != 2:
            failures.append("malformed-checksum-line")
            continue
        expected, relative = fields
        relative_path = Path(relative.strip())
        normalized = relative_path.as_posix()
        if relative_path.is_absolute() or relative_path.drive or ".." in relative_path.parts:
            failures.append(f"unsafe-path:{normalized}")
            continue
        if normalized in covered:
            failures.append(f"duplicate:{normalized}")
            continue
        covered.add(normalized)
        path = root / normalized
        if not path.is_file():
            failures.append(f"missing:{normalized}")
            continue
        checked += 1
        actual = _sha256_file(path)
        if actual != expected.lower():
            failures.append(f"sha256:{normalized}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    for missing in sorted(actual_files - covered):
        failures.append(f"uncovered:{missing}")
    for extra in sorted(covered - actual_files):
        failures.append(f"nonexistent:{extra}")
    return checked, failures


def _safe_logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a PVSE frozen-estimator release asset")
    parser.add_argument("root", help="extracted estimator-pack root")
    parser.add_argument(
        "--allow-joblib",
        action="store_true",
        help="load the trusted author-environment snapshot and compare it with portable inference",
    )
    return parser


def _verify_single_estimator(root: Path, *, allow_joblib: bool) -> dict[str, object]:
    metadata_files = sorted((root / "portable").glob("*.json"))
    if len(metadata_files) != 1:
        raise ValueError("an estimator pack must contain exactly one portable metadata file")
    estimator = load_portable_estimator(metadata_files[0])
    random = np.random.default_rng(3920)
    matrix = random.normal(size=(257, len(estimator.features))).astype(np.float64)
    report: dict[str, object] = {
        "logical_role": estimator.logical_role,
        "rows_checked": int(len(matrix)),
    }
    passed = True
    if isinstance(estimator, PortableReliabilityEstimator):
        probability = estimator.predict_corruption_probability(matrix)
        report["portable_outputs_finite"] = bool(np.all(np.isfinite(probability)))
    elif isinstance(estimator, PortableTwoHeadGate):
        score = estimator.score(matrix)
        report["portable_outputs_finite"] = bool(np.all(np.isfinite(score)))
    else:  # pragma: no cover - guarded by the loader
        raise TypeError(f"unsupported portable estimator type: {type(estimator)!r}")
    passed = passed and bool(report["portable_outputs_finite"])

    if allow_joblib:
        import joblib

        relative = Path(str(estimator.metadata["joblib_file"]))
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError(f"unsafe joblib path: {relative}")
        payload = joblib.load(root / relative)
        if isinstance(estimator, PortableReliabilityEstimator):
            probability_joblib = payload["model"].predict_proba(matrix)[:, 1]
            probability_portable = estimator.predict_corruption_probability(matrix)
            maximum_error = float(np.max(np.abs(probability_joblib - probability_portable)))
        else:
            p_help = payload["help_model"].predict_proba(matrix)[:, 1]
            p_hurt = payload["hurt_model"].predict_proba(matrix)[:, 1]
            portable_help, portable_hurt = estimator.predict_probabilities(matrix)
            maximum_error = max(
                float(np.max(np.abs(p_help - portable_help))),
                float(np.max(np.abs(p_hurt - portable_hurt))),
            )
        report["maximum_probability_error"] = maximum_error
        report["joblib_loaded"] = True
        passed = passed and maximum_error <= 1e-6
    report["passed"] = passed
    return report


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    checked, failures = _verify_checksums(root)
    generic_metadata = sorted((root / "portable").glob("*.json"))
    if generic_metadata and not (root / "portable" / "clean_verifier.json").is_file():
        report = _verify_single_estimator(root, allow_joblib=bool(args.allow_joblib))
        report.update(
            {
                "checksum_files_checked": checked,
                "checksum_failures": failures,
                "passed": bool(report["passed"]) and not failures,
            }
        )
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["passed"] else 1)

    portable = load_portable_clean_verifier(root / "portable" / "clean_verifier.json")
    matrix = np.random.default_rng(3920).normal(size=(257, len(portable.features))).astype(np.float64)
    score = portable.score(matrix)
    report: dict[str, object] = {
        "passed": not failures,
        "logical_role": portable.logical_role,
        "checksum_files_checked": checked,
        "checksum_failures": failures,
        "portable_rows_checked": int(len(matrix)),
        "portable_scores_finite": bool(np.all(np.isfinite(score))),
    }
    if not report["portable_scores_finite"]:
        report["passed"] = False

    if args.allow_joblib:
        import joblib

        payload = joblib.load(root / "joblib_author_env" / "clean_verifier.joblib")
        p_help = payload["help_model"].predict_proba(matrix)[:, 1]
        p_hurt = payload["hurt_model"].predict_proba(matrix)[:, 1]
        portable_help, portable_hurt = portable.predict_probabilities(matrix)
        score_joblib = _safe_logit(p_help) - float(portable.lambda_hurt) * _safe_logit(p_hurt)
        max_probability_error = max(
            float(np.max(np.abs(p_help - portable_help))),
            float(np.max(np.abs(p_hurt - portable_hurt))),
        )
        max_score_error = float(np.max(np.abs(score_joblib - score)))
        decision_mismatches = int(
            np.sum((score_joblib > portable.threshold) != (score > portable.threshold))
        )
        report.update(
            {
                "joblib_loaded": True,
                "max_probability_error": max_probability_error,
                "max_score_error": max_score_error,
                "decision_mismatches": decision_mismatches,
            }
        )
        if max_probability_error > 1e-6 or max_score_error > 1e-4 or decision_mismatches:
            report["passed"] = False

    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
