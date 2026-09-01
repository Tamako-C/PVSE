from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
import sklearn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pvse.clean.features import CLEAN_VERIFIER_FEATURES
from pvse.clean.verifier import (
    CleanVerifierConfig,
    fit_clean_verifier,
    safe_logit,
    save_clean_verifier,
)
from pvse.estimators.portable import load_portable_clean_verifier


LOGICAL_ROLE = "clean.resnet12.table4"
ASSET_NAME = "pvse_clean_verifier_table4_v1"
CALIBRATION_SOURCE_SHA256 = "3e596f16a1694de46b536b8bd75de6628f9bc861f3bad1577b85edb767ef4400"
FIT_RANDOM_STATE = 3920
EXPECTED_PARAMETERS = {
    "feature_set": "patch_plus_proposal",
    "model": "l1_logreg",
    "lambda_hurt": 2.0,
    "proposal_source": "runnerup_a0",
    "threshold": 5.35311817754396,
    "diagnostic_only": False,
    "not_for_formal_claim": False,
}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_selection(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("val_selected_max_net")
    if not isinstance(selected, dict):
        raise ValueError("selection record does not contain val_selected_max_net")
    mismatches = {
        key: (selected.get(key), expected)
        for key, expected in EXPECTED_PARAMETERS.items()
        if selected.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"selected clean-verifier configuration mismatch: {mismatches}")
    return selected


def _load_calibration(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if _sha256_file(path) != CALIBRATION_SOURCE_SHA256:
        raise ValueError("calibration source SHA-256 does not match the formal source")
    columns = [
        "split",
        "episode",
        "proposal_source",
        "help_label",
        "hurt_label",
        *CLEAN_VERIFIER_FEATURES,
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame.loc[frame["proposal_source"] == "runnerup_a0"].reset_index(drop=True)
    if len(frame) != 22_500:
        raise ValueError(f"expected 22,500 runner-up calibration rows, found {len(frame)}")
    if set(frame["split"].astype(str)) != {"val"}:
        raise ValueError("calibration rows are not exclusively from the validation split")
    if int(frame["episode"].nunique()) != 300:
        raise ValueError("expected 300 calibration episodes")
    counts = {
        "rows": int(len(frame)),
        "episodes": int(frame["episode"].nunique()),
        "help_positive": int(frame["help_label"].sum()),
        "hurt_positive": int(frame["hurt_label"].sum()),
    }
    return frame, counts


def _load_formal_test(path: Path) -> pd.DataFrame:
    columns = [
        "split",
        "episode",
        "query_index",
        "proposal_source",
        "a0_pred",
        "proposal_class",
        "analysis_only_true_label",
        *CLEAN_VERIFIER_FEATURES,
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    if len(frame) != 75_000:
        raise ValueError(f"expected 75,000 formal test rows, found {len(frame)}")
    if set(frame["split"].astype(str)) != {"test"}:
        raise ValueError("formal evaluation rows are not exclusively from the test split")
    if set(frame["proposal_source"].astype(str)) != {"runnerup_a0"}:
        raise ValueError("formal evaluation rows do not exclusively use the A0 runner-up")
    if int(frame["episode"].nunique()) != 1_000:
        raise ValueError("expected 1,000 formal test episodes")
    per_episode = frame.groupby("episode", sort=False).size()
    if set(per_episode.astype(int)) != {75}:
        raise ValueError("formal test episodes do not contain exactly 75 queries")
    if frame.duplicated(["episode", "query_index"]).any():
        raise ValueError("formal test episode/query identifiers are not unique")
    return frame.sort_values(["episode", "query_index"], kind="stable").reset_index(drop=True)


def _load_archived_switch(path: Path) -> pd.DataFrame:
    columns = [
        "episode",
        "query_index",
        "proposal_source",
        "a0_pred",
        "proposal_class",
        "analysis_only_true_label",
        "score",
        "applied",
        "final_pred",
        "a0_correct",
        "final_correct",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    candidate = frame.loc[
        (frame["proposal_source"].astype(str) == "runnerup_a0")
        & frame["score"].notna()
        & (frame["proposal_class"] >= 0)
    ].copy()
    if len(candidate) != 75_000:
        raise ValueError(f"expected 75,000 archived switch rows, found {len(candidate)}")
    if candidate.duplicated(["episode", "query_index"]).any():
        raise ValueError("archived switch episode/query identifiers are not unique")
    return candidate.sort_values(["episode", "query_index"], kind="stable").reset_index(drop=True)


def _load_submitted_switch(path: Path) -> dict[str, object]:
    table = pd.read_csv(path)
    row = table.loc[table["Method"].astype(str).str.startswith("PVSE-C-Switch")]
    if len(row) != 1:
        raise ValueError("submitted Table 4 does not contain one PVSE-C-Switch row")
    item = row.iloc[0]
    help_count, hurt_count = (int(value) for value in str(item["Help/Hurt"]).split("/"))
    gain = float(str(item["Gain (episode CI)"]).split()[0])
    return {
        "accuracy_percent_display": float(str(item["Acc."]).rstrip("%")),
        "gain_pp_display": gain,
        "help": help_count,
        "hurt": hurt_count,
        "net": int(str(item["Net"]).lstrip("+")),
        "precision_percent_display": float(str(item["Precision"]).rstrip("%")),
    }


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, list(CLEAN_VERIFIER_FEATURES)].to_numpy(dtype=np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _joblib_scores(bundle, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_help = bundle.help_model.predict_proba(matrix)[:, 1].astype(np.float64)
    p_hurt = bundle.hurt_model.predict_proba(matrix)[:, 1].astype(np.float64)
    score = safe_logit(p_help) - float(bundle.config.lambda_hurt) * safe_logit(p_hurt)
    return p_help, p_hurt, score


def _aggregate(
    truth: np.ndarray,
    a0_prediction: np.ndarray,
    proposal: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, object]:
    final = np.where(accepted, proposal, a0_prediction)
    a0_correct = a0_prediction == truth
    final_correct = final == truth
    help_mask = ~a0_correct & final_correct
    hurt_mask = a0_correct & ~final_correct
    help_count = int(help_mask.sum())
    hurt_count = int(hurt_mask.sum())
    return {
        "queries": int(len(truth)),
        "baseline_accuracy": float(a0_correct.mean()),
        "method_accuracy": float(final_correct.mean()),
        "gain_pp": float(100.0 * np.mean(final_correct.astype(np.int8) - a0_correct.astype(np.int8))),
        "help": help_count,
        "hurt": hurt_count,
        "net": help_count - hurt_count,
        "precision": float(help_count / max(1, help_count + hurt_count)),
    }


def _assert_submitted_match(metrics: dict[str, object], submitted: dict[str, object]) -> None:
    checks = {
        "accuracy": round(float(metrics["method_accuracy"]) * 100.0, 3)
        == float(submitted["accuracy_percent_display"]),
        "gain": round(float(metrics["gain_pp"]), 3) == float(submitted["gain_pp_display"]),
        "help": int(metrics["help"]) == int(submitted["help"]),
        "hurt": int(metrics["hurt"]) == int(submitted["hurt"]),
        "net": int(metrics["net"]) == int(submitted["net"]),
        "precision": round(float(metrics["precision"]) * 100.0, 1)
        == float(submitted["precision_percent_display"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"rebuilt estimator does not reproduce submitted Table 4 fields: {failed}")


def _pipeline_arrays(model) -> dict[str, np.ndarray]:
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    return {
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "scaler_var": np.asarray(scaler.var_, dtype=np.float64),
        "coef": np.asarray(classifier.coef_, dtype=np.float64),
        "intercept": np.asarray(classifier.intercept_, dtype=np.float64),
        "classes": np.asarray(classifier.classes_, dtype=np.int64),
    }


def _write_portable(root: Path, bundle) -> Path:
    portable_dir = root / "portable"
    portable_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = portable_dir / "clean_verifier.npz"
    help_arrays = _pipeline_arrays(bundle.help_model)
    hurt_arrays = _pipeline_arrays(bundle.hurt_model)
    arrays = {
        **{f"help_{key}": value for key, value in help_arrays.items()},
        **{f"hurt_{key}": value for key, value in hurt_arrays.items()},
    }
    np.savez_compressed(arrays_path, **arrays)
    metadata = {
        "schema": "pvse.portable.clean_verifier",
        "schema_version": 1,
        "logical_role": LOGICAL_ROLE,
        "paper_scope": [
            "Table 4 PVSE-C-Switch verifier",
            "Table 4 PVSE-C-HD verifier gate",
            "Table 9 fixed-transfer clean verifier",
        ],
        "arrays_file": arrays_path.name,
        "arrays_sha256": _sha256_file(arrays_path),
        "features": list(bundle.features),
        "model": {
            "heads": ["help", "hurt"],
            "estimator": "LogisticRegression",
            "scaler": "StandardScaler",
            "penalty": bundle.config.penalty,
            "solver": bundle.config.solver,
            "C": bundle.config.C,
            "class_weight": bundle.config.class_weight,
            "max_iter": bundle.config.max_iter,
            "fit_random_state": bundle.config.random_state,
        },
        "policy": {
            "proposal_source": "A0 runner-up",
            "feature_set": bundle.config.feature_set,
            "score": "logit(p_help) - lambda_hurt * logit(p_hurt)",
            "lambda_hurt": bundle.config.lambda_hurt,
            "threshold": bundle.config.threshold,
            "comparison": "score > threshold",
            "logit_clip_epsilon": 1e-6,
        },
        "calibration": {
            "split": "official validation",
            "episodes": 300,
            "runner_up_rows": 22_500,
            "test_selection": False,
        },
    }
    metadata_path = portable_dir / "clean_verifier.json"
    _json_dump(metadata_path, metadata)
    return metadata_path


def _write_readme(root: Path) -> None:
    text = """# PVSE-C Clean Verifier (Table 4)

This asset contains the frozen two-head clean verifier used by the PVSE-C methods. It was fitted on 300 official miniImageNet validation episodes (22,500 A0 runner-up rows) with the paper-fixed configuration. Formal test episodes were used only for parity evaluation.

## Recommended format

`portable/clean_verifier.json` and `portable/clean_verifier.npz` provide a NumPy-only representation. The metadata fixes feature order, preprocessing, model parameters, score composition, and threshold. The loader verifies the array checksum before inference.

```python
from pvse.estimators import load_portable_clean_verifier

verifier = load_portable_clean_verifier("portable/clean_verifier.json")
scores = verifier.score(feature_matrix)
accepted = verifier.accepted(feature_matrix)
```

## Author-environment snapshot

`joblib_author_env/clean_verifier.joblib` preserves the original scikit-learn pipeline structure. Joblib files can execute Python objects while loading; load this snapshot only when it comes from a trusted release. The portable representation is preferred for inspection and exchange.

## Verification

After extracting the archive, run:

```bash
pvse-verify-estimator-pack pvse_clean_verifier_table4_v1
pvse-verify-estimator-pack pvse_clean_verifier_table4_v1 --allow-joblib
```

The formal parity record covers all 1,000 episodes and 75,000 query predictions. Rebuilding with the fixed configuration reproduces the submitted PVSE-C-Switch accuracy, gain, help, hurt, net, and precision exactly at the reported precision.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _write_provenance(
    root: Path,
    sources: list[tuple[str, str, Path, int, bool, bool]],
    calibration_counts: dict[str, int],
    config: CleanVerifierConfig,
) -> None:
    provenance = root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    with (provenance / "source_files.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "role",
                "sha256",
                "rows_used",
                "used_for_fit",
                "used_for_model_selection",
            ],
        )
        writer.writeheader()
        for source_id, role, path, rows, fit, selection in sources:
            writer.writerow(
                {
                    "source_id": source_id,
                    "role": role,
                    "sha256": _sha256_file(path),
                    "rows_used": rows,
                    "used_for_fit": str(fit).lower(),
                    "used_for_model_selection": str(selection).lower(),
                }
            )
    _json_dump(
        provenance / "fit_protocol.json",
        {
            "logical_role": LOGICAL_ROLE,
            "configuration": asdict(config),
            "calibration": calibration_counts,
            "calibration_split": "official validation",
            "test_selection": False,
            "test_rows_used_for_fit": 0,
            "test_rows_used_for_selection": 0,
        },
    )
    _json_dump(
        provenance / "build_environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.system(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    )


def _role(relative: str) -> str:
    if relative == "README.md":
        return "usage documentation"
    if relative.startswith("portable/"):
        return "portable estimator"
    if relative.startswith("joblib_author_env/"):
        return "author-environment estimator snapshot"
    if relative.startswith("parity/"):
        return "formal parity evidence"
    if relative.startswith("provenance/"):
        return "fit and source provenance"
    return "release metadata"


def _write_manifests(root: Path) -> None:
    excluded = {"MANIFEST.json", "MANIFEST.csv", "SHA256SUMS.txt"}
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "role": _role(relative),
            }
        )
    _json_dump(
        root / "MANIFEST.json",
        {
            "schema": "pvse.estimator_release_manifest",
            "schema_version": 1,
            "asset": ASSET_NAME,
            "logical_role": LOGICAL_ROLE,
            "files": rows,
        },
    )
    with (root / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256", "role"])
        writer.writeheader()
        writer.writerows(rows)
    with (root / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            if relative == "SHA256SUMS.txt":
                continue
            handle.write(f"{_sha256_file(path)}  {relative}\n")


def _write_deterministic_zip(root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.zip")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.comment = b""
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(f"{root.name}/{relative}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(temporary, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("generated ZIP failed its integrity check")
        if archive.comment:
            raise RuntimeError("generated ZIP contains an unexpected comment")
    temporary.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate the frozen PVSE-C clean verifier release asset"
    )
    parser.add_argument("--calibration-rows", required=True)
    parser.add_argument("--selected-configuration", required=True)
    parser.add_argument("--formal-test-rows", required=True)
    parser.add_argument("--formal-switch-outputs", required=True)
    parser.add_argument("--submitted-table4", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    calibration_path = Path(args.calibration_rows).resolve()
    selection_path = Path(args.selected_configuration).resolve()
    test_path = Path(args.formal_test_rows).resolve()
    switch_path = Path(args.formal_switch_outputs).resolve()
    submitted_path = Path(args.submitted_table4).resolve()
    output = Path(args.output_dir).resolve()
    zip_path = Path(args.zip_path).resolve()
    if output.name != ASSET_NAME:
        raise ValueError(f"output directory must be named {ASSET_NAME}")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    selected = _read_selection(selection_path)
    calibration, calibration_counts = _load_calibration(calibration_path)
    config = CleanVerifierConfig(
        penalty="l1",
        solver="liblinear",
        C=0.5,
        class_weight="balanced",
        max_iter=1000,
        lambda_hurt=float(selected["lambda_hurt"]),
        threshold=float(selected["threshold"]),
        feature_set=str(selected["feature_set"]),
        proposal_source=str(selected["proposal_source"]),
        random_state=FIT_RANDOM_STATE,
    )
    fit_rows = calibration.loc[
        :, [*CLEAN_VERIFIER_FEATURES, "help_label", "hurt_label"]
    ].to_dict(orient="records")
    bundle = fit_clean_verifier(fit_rows, config=config)

    joblib_path = output / "joblib_author_env" / "clean_verifier.joblib"
    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    save_clean_verifier(bundle, joblib_path)
    portable_metadata = _write_portable(output, bundle)

    formal = _load_formal_test(test_path)
    archived = _load_archived_switch(switch_path)
    merged = formal.merge(
        archived,
        on=["episode", "query_index"],
        how="inner",
        validate="one_to_one",
        suffixes=("_features", "_archived"),
    )
    if len(merged) != 75_000:
        raise ValueError("formal feature and archived prediction rows are not fully aligned")
    for field in ("a0_pred", "proposal_class", "analysis_only_true_label"):
        if not np.array_equal(
            merged[f"{field}_features"].to_numpy(), merged[f"{field}_archived"].to_numpy()
        ):
            raise ValueError(f"formal rows disagree on aligned field {field}")

    matrix = _feature_matrix(merged)
    p_help, p_hurt, rebuilt_score = _joblib_scores(bundle, matrix)
    rebuilt_accepted = rebuilt_score > float(config.threshold)
    truth = merged["analysis_only_true_label_features"].to_numpy(dtype=np.int64)
    a0_prediction = merged["a0_pred_features"].to_numpy(dtype=np.int64)
    proposal = merged["proposal_class_features"].to_numpy(dtype=np.int64)
    rebuilt_metrics = _aggregate(truth, a0_prediction, proposal, rebuilt_accepted)
    archived_accepted = merged["applied"].to_numpy(dtype=np.int64).astype(bool)
    archived_metrics = _aggregate(truth, a0_prediction, proposal, archived_accepted)
    submitted = _load_submitted_switch(submitted_path)
    _assert_submitted_match(rebuilt_metrics, submitted)
    _assert_submitted_match(archived_metrics, submitted)

    archived_score = merged["score"].to_numpy(dtype=np.float64)
    score_error = np.abs(rebuilt_score - archived_score)
    gate_mismatches = int(np.sum(rebuilt_accepted != archived_accepted))
    formal_parity = {
        "schema": "pvse.clean_verifier.formal_test1000_parity",
        "schema_version": 1,
        "logical_role": LOGICAL_ROLE,
        "fit_scope": {
            "split": "official validation",
            "episodes": 300,
            "rows": 22_500,
            "test_rows_used_for_fit": 0,
            "test_rows_used_for_selection": 0,
            "fit_random_state": FIT_RANDOM_STATE,
        },
        "evaluation_scope": {"split": "test", "episodes": 1_000, "queries": 75_000},
        "submitted_display_values": submitted,
        "rebuilt_metrics": rebuilt_metrics,
        "archived_metrics": archived_metrics,
        "verification": {
            "submitted_display_fields_exact": True,
            "historical_numeric_check_passed": True,
        },
    }
    gate_decision_agreement = float(1.0 - gate_mismatches / len(merged))
    if gate_decision_agreement < 0.9999:
        raise AssertionError("historical gate-decision agreement is below 99.99%")
    if float(score_error.max()) > 0.05:
        raise AssertionError("historical clean-verifier score error exceeds tolerance")
    _json_dump(output / "parity" / "formal_test1000_parity.json", formal_parity)

    portable = load_portable_clean_verifier(portable_metadata)
    portable_help, portable_hurt = portable.predict_probabilities(matrix)
    portable_score = portable.score(matrix)
    portable_accepted = portable.accepted(matrix)
    portable_parity = {
        "schema": "pvse.clean_verifier.portable_joblib_parity",
        "schema_version": 1,
        "rows": int(len(matrix)),
        "maximum_probability_absolute_error": float(
            max(np.max(np.abs(p_help - portable_help)), np.max(np.abs(p_hurt - portable_hurt)))
        ),
        "maximum_score_absolute_error": float(np.max(np.abs(rebuilt_score - portable_score))),
        "decision_mismatches": int(np.sum(rebuilt_accepted != portable_accepted)),
        "tolerances": {
            "maximum_probability_absolute_error": 1e-6,
            "maximum_score_absolute_error": 1e-4,
            "decision_mismatches": 0,
        },
    }
    if portable_parity["maximum_probability_absolute_error"] > 1e-6:
        raise AssertionError("portable probability parity exceeds tolerance")
    if portable_parity["maximum_score_absolute_error"] > 1e-4:
        raise AssertionError("portable score parity exceeds tolerance")
    if portable_parity["decision_mismatches"] != 0:
        raise AssertionError("portable and joblib verifier decisions differ")
    _json_dump(output / "parity" / "portable_joblib_parity.json", portable_parity)

    _write_readme(output)
    _write_provenance(
        output,
        [
            (
                "official_validation_runnerup_rows",
                "clean verifier fitting",
                calibration_path,
                22_500,
                True,
                False,
            ),
            (
                "validation_selected_configuration",
                "fixed clean verifier configuration",
                selection_path,
                1,
                False,
                True,
            ),
            (
                "formal_test1000_feature_rows",
                "formal parity evaluation",
                test_path,
                75_000,
                False,
                False,
            ),
            (
                "formal_test1000_switch_outputs",
                "historical parity reference",
                switch_path,
                75_000,
                False,
                False,
            ),
            (
                "submitted_table4",
                "submitted display-value reference",
                submitted_path,
                4,
                False,
                False,
            ),
        ],
        calibration_counts,
        config,
    )
    _write_manifests(output)
    _write_deterministic_zip(output, zip_path)
    zip_digest = _sha256_file(zip_path)
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(
        f"{zip_digest}  {zip_path.name}\n", encoding="utf-8"
    )
    report = {
        "passed": True,
        "asset_directory": str(output),
        "zip_path": str(zip_path),
        "zip_sha256": zip_digest,
        "formal_test1000_parity": formal_parity,
        "portable_joblib_parity": portable_parity,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
