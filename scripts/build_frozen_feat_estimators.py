from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pvse.estimators import load_portable_reliability_estimator, load_portable_two_head_gate
from pvse.plugins.feat import (
    FEAT_CLEAN_FEATURES,
    FEAT_NOISY_FEATURE_SETS,
    FeatCleanGatePolicy,
    fit_feat_clean_gate,
    fit_select_feat_reliability,
)
from pvse.release.estimator_pack import (
    prepare_output,
    sha256_file,
    write_binary_estimator,
    write_deterministic_zip,
    write_json,
    write_pack_manifests,
    write_source_manifest,
    write_two_head_gate,
)


CLEAN_ASSET = "pvse_c_feat_clean_gate_v1"


def _noisy_asset(severity: int, feature_set: str) -> str:
    suffix = "global_patch" if feature_set == "global_plus_patch" else "global_only"
    return f"pvse_r_hd_feat_noise{severity}_{suffix}_v1"


def _clean_metrics(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if len(frame) != 45_000 or frame["episode"].nunique() != 600:
        raise ValueError("FEAT clean formal query coverage is incomplete")
    truth = frame["true_label"].to_numpy(dtype=np.int64)
    baseline = frame["a0_pred"].to_numpy(dtype=np.int64) == truth
    edited = frame["hd_pred"].to_numpy(dtype=np.int64) == truth
    return {
        "episodes": 600,
        "queries": 45_000,
        "baseline_accuracy_percent": float(100.0 * baseline.mean()),
        "method_accuracy_percent": float(100.0 * edited.mean()),
        "gain_pp": float(100.0 * np.mean(edited.astype(int) - baseline.astype(int))),
        "help": int((~baseline & edited).sum()),
        "hurt": int((baseline & ~edited).sum()),
    }


def _noisy_metrics(path: Path, prediction_column: str) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if len(frame) != 75_000 or frame["episode"].nunique() != 1_000:
        raise ValueError(f"FEAT noisy formal query coverage is incomplete in {path.name}")
    truth = frame["true_label"].to_numpy(dtype=np.int64)
    baseline = frame["noisy_pred"].to_numpy(dtype=np.int64) == truth
    edited = frame[prediction_column].to_numpy(dtype=np.int64) == truth
    return {
        "episodes": 1_000,
        "queries": 75_000,
        "baseline_accuracy_percent": float(100.0 * baseline.mean()),
        "method_accuracy_percent": float(100.0 * edited.mean()),
        "gain_pp": float(100.0 * np.mean(edited.astype(int) - baseline.astype(int))),
    }


def _table10(path: Path) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for _, row in pd.read_csv(path).iterrows():
        noise = str(row["Noise"])
        severity = 0 if noise.startswith("0%") else int(noise.rstrip("%"))
        output[severity] = {
            "baseline": float(str(row["FEAT native"]).rstrip("%")),
            "accuracy": float(str(row["FEAT+PVSE"]).rstrip("%")),
            "gain": float(str(row["Gain (episode CI)"]).split()[0]),
        }
    return output


def _table7(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    output: dict[tuple[str, int], tuple[float, float]] = {}
    for _, row in pd.read_csv(path).iterrows():
        output[(str(row["Setting"]), int(str(row["Noise"]).rstrip("%")))] = (
            float(row["Global-only"]),
            float(row["Global+Patch"]),
        )
    return output


def _assert_table10(actual: Mapping[str, Any], expected: Mapping[str, float]) -> None:
    if not (
        round(float(actual["baseline_accuracy_percent"]), 3) == expected["baseline"]
        and round(float(actual["method_accuracy_percent"]), 3) == expected["accuracy"]
        and round(float(actual["gain_pp"]), 3) == expected["gain"]
    ):
        raise AssertionError("formal Table 10 parity failed")


def _source(source_id: str, role: str, path: Path, rows: int, *, fit: bool = False, selection: bool = False) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "role": role,
        "sha256": sha256_file(path),
        "rows_used": rows,
        "used_for_fit": str(fit).lower(),
        "used_for_model_selection": str(selection).lower(),
    }


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _finish(root: Path, *, asset: str, role: str, readme: str, output_root: Path) -> dict[str, str]:
    (root / "README.md").write_text(readme, encoding="utf-8")
    write_pack_manifests(root, asset=asset, logical_role=role)
    zip_path = output_root / f"{asset}.zip"
    return {"asset": asset, "zip": str(zip_path), "sha256": write_deterministic_zip(root, zip_path)}


def _build_clean(
    *,
    clean_root: Path,
    table10_path: Path,
    expected: Mapping[str, float],
    output_root: Path,
    force: bool,
) -> dict[str, str]:
    root = prepare_output(output_root / CLEAN_ASSET, expected_name=CLEAN_ASSET, force=force)
    rows_path = clean_root / "feat_val_calibration_rows.csv"
    selection_path = clean_root / "selected_model.json"
    query_path = clean_root / "feat_query_outputs.csv"
    rows = pd.read_csv(rows_path)
    if len(rows) != 22_500 or rows["episode"].nunique() != 300:
        raise ValueError("FEAT clean calibration must contain 300 episodes and 22,500 rows")
    selected = json.loads(selection_path.read_text(encoding="utf-8"))["pvse_gate"]
    policy = FeatCleanGatePolicy(float(selected["lambda_hurt"]), float(selected["tau"]), 517, 518)
    bundle = fit_feat_clean_gate(rows.to_dict(orient="records"), policy=policy)
    joblib_relative = "joblib_author_env/feat_clean_gate.joblib"
    joblib_path = root / joblib_relative
    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"format": "pvse_feat_clean_gate_v1", "features": list(bundle.features), "policy": asdict(bundle.policy), "help_model": bundle.help_model, "hurt_model": bundle.hurt_model}, joblib_path)
    metadata = write_two_head_gate(
        root / "portable",
        estimator_id="feat_plugin.table10.clean_gate",
        file_stem="feat_clean_gate",
        logical_role="feat_plugin.table10.clean_gate",
        help_model=bundle.help_model,
        hurt_model=bundle.hurt_model,
        features=FEAT_CLEAN_FEATURES,
        model_metadata={"heads": ["help", "hurt"], "estimator": "LogisticRegression", "scaler": "StandardScaler", "solver": "liblinear", "penalty": "l2", "C": 1.0, "class_weight": "balanced", "max_iter": 1000, "fit_random_states": {"help": 517, "hurt": 518}},
        policy={"score_mode": "probability_difference", "lambda_hurt": policy.lambda_hurt, "threshold": policy.threshold, "comparison": ">=", "logit_clip_epsilon": 1e-6},
        calibration={"split": "official validation", "episodes": 300, "query_rows": 22_500, "test_selection": False},
        paper_scope=("Table 10 FEAT clean plug-in",),
        joblib_file=joblib_relative,
    )
    portable = load_portable_two_head_gate(metadata)
    matrix = np.random.default_rng(3920).normal(size=(257, len(portable.features)))
    p_help, p_hurt = portable.predict_probabilities(matrix)
    maximum_error = max(float(np.max(np.abs(p_help - bundle.help_model.predict_proba(matrix)[:, 1]))), float(np.max(np.abs(p_hurt - bundle.hurt_model.predict_proba(matrix)[:, 1]))))
    if maximum_error > 1e-6:
        raise AssertionError("portable and joblib FEAT clean probabilities differ")
    metrics = _clean_metrics(query_path)
    _assert_table10(metrics, expected)
    write_json(root / "parity" / "formal_parity.json", {"formal_results": metrics, "portable_joblib_max_error": maximum_error})
    write_json(root / "provenance" / "build_environment.json", _environment())
    write_source_manifest(root / "provenance" / "source_files.csv", [
        _source("validation_rows", "gate fitting and selection", rows_path, 22_500, fit=True, selection=True),
        _source("selected_configuration", "fixed selected policy", selection_path, 1, selection=True),
        _source("formal_query_outputs", "formal parity", query_path, 45_000),
        _source("submitted_table10", "display-value reference", table10_path, 4),
    ])
    readme = f"""# {CLEAN_ASSET}

This asset contains the frozen two-head PVSE-C gate used with the frozen FEAT checkpoint for the submitted clean plug-in result.

## Recommended format

`portable/feat_clean_gate.json` and `portable/feat_clean_gate.npz` provide a NumPy-only representation. The metadata fixes feature order, preprocessing, both logistic heads, score composition, and threshold.

```python
from pvse.estimators import load_portable_two_head_gate

gate = load_portable_two_head_gate("portable/feat_clean_gate.json")
accepted = gate.accepted(feature_matrix)
```

## Author-environment snapshot

`joblib_author_env/feat_clean_gate.joblib` preserves the fitted scikit-learn pipelines. Load it only from a trusted release; the portable representation is preferred for inspection and exchange.

## Verification

```bash
pvse-verify-estimator-pack {CLEAN_ASSET}
pvse-verify-estimator-pack {CLEAN_ASSET} --allow-joblib
```

The parity record checks the submitted clean FEAT plug-in result and portable/joblib numerical agreement.
"""
    return _finish(root, asset=CLEAN_ASSET, role="feat_plugin.table10.clean_gate", readme=readme, output_root=output_root)


def _build_noisy(
    *,
    noisy_root: Path,
    table7_path: Path,
    table10_path: Path,
    table7: Mapping[tuple[str, int], tuple[float, float]],
    table10: Mapping[int, Mapping[str, float]],
    severity: int,
    per_class: int,
    feature_set: str,
    output_root: Path,
    force: bool,
) -> dict[str, str]:
    asset = _noisy_asset(severity, feature_set)
    role = f"feat_plugin.noisy_{severity}.{feature_set}"
    root = prepare_output(output_root / asset, expected_name=asset, force=force)
    train_path = noisy_root / f"feat_support_rows_train_r{per_class}.csv"
    val_path = noisy_root / f"feat_support_rows_val_r{per_class}.csv"
    query_path = noisy_root / f"feat_noisy_query_outputs_r{per_class}.csv"
    selection_path = noisy_root / "selected_configs.csv"
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    if len(train) != 1_250 or len(val) != 1_250:
        raise ValueError("FEAT noisy calibration must contain 1,250 train and validation rows")
    selected_all = pd.read_csv(selection_path)
    selected = selected_all.loc[(selected_all["severity_r_per_class"].astype(int) == per_class) & (selected_all["feature_set"].astype(str) == feature_set)]
    if len(selected) != 1:
        raise ValueError(f"missing selected configuration for {severity}/{feature_set}")
    selected_row = selected.iloc[0]
    bundle, _ = fit_select_feat_reliability(train.to_dict(orient="records"), val.to_dict(orient="records"), feature_sets=(feature_set,))
    if bundle.selection.model_type != str(selected_row["model_type"]) or bundle.selection.threshold_quantile != int(selected_row["threshold_quantile"]) or abs(bundle.selection.threshold - float(selected_row["threshold"])) > 1e-9:
        raise AssertionError(f"FEAT selection parity failed for {severity}/{feature_set}")
    joblib_relative = "joblib_author_env/reliability_estimator.joblib"
    joblib_path = root / joblib_relative
    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"format": "pvse_feat_reliability_v1", "features": list(bundle.features), "selection": asdict(bundle.selection), "model": bundle.model}, joblib_path)
    metadata = write_binary_estimator(
        root / "portable",
        estimator_id=role,
        file_stem="reliability_estimator",
        arrays_stem="reliability_estimator",
        logical_role=role,
        model=bundle.model,
        features=FEAT_NOISY_FEATURE_SETS[feature_set],
        model_metadata={"estimator": "LogisticRegression", "scaler": "StandardScaler", "penalty": "l1" if bundle.selection.model_type == "l1_logreg" else "l2", "solver": "liblinear", "C": 1.0, "class_weight": "balanced", "max_iter": 1000, "fit_random_state": 519},
        policy={"threshold": bundle.selection.threshold, "threshold_quantile": bundle.selection.threshold_quantile, "comparison": ">=", "max_delete": 3},
        calibration={"train_split": "miniImageNet train", "validation_split": "miniImageNet validation", "train_episodes": 50, "validation_episodes": 50, "severity_percent": severity, "test_selection": False},
        paper_scope=("Table 7 patch ablation", "Table 10 FEAT noisy plug-in"),
        joblib_file=joblib_relative,
    )
    portable = load_portable_reliability_estimator(metadata)
    matrix = np.random.default_rng(3920).normal(size=(257, len(portable.features)))
    maximum_error = float(np.max(np.abs(portable.predict_corruption_probability(matrix) - bundle.model.predict_proba(matrix)[:, 1])))
    if maximum_error > 1e-6:
        raise AssertionError("portable and joblib FEAT reliability probabilities differ")
    prediction_column = "pvse_hd_pred" if feature_set == "global_plus_patch" else "global_only_pred"
    metrics = _noisy_metrics(query_path, prediction_column)
    gain_index = 1 if feature_set == "global_plus_patch" else 0
    if round(metrics["gain_pp"], 3) != table7[("FEAT plug-in", severity)][gain_index]:
        raise AssertionError("formal Table 7 parity failed")
    if feature_set == "global_plus_patch":
        _assert_table10(metrics, table10[severity])
    write_json(root / "parity" / "formal_parity.json", {"formal_results": metrics, "portable_joblib_max_error": maximum_error})
    write_json(root / "provenance" / "build_environment.json", _environment())
    write_source_manifest(root / "provenance" / "source_files.csv", [
        _source("training_rows", "estimator fitting", train_path, 1_250, fit=True),
        _source("validation_rows", "model and threshold selection", val_path, 1_250, selection=True),
        _source("selected_configuration", "fixed selected configuration", selection_path, 6, selection=True),
        _source("formal_query_outputs", "formal parity", query_path, 75_000),
        _source("submitted_table7", "display-value reference", table7_path, 6),
        _source("submitted_table10", "display-value reference", table10_path, 4),
    ])
    readme = f"""# {asset}

This asset contains one frozen PVSE-R hard-deletion estimator used with the frozen FEAT checkpoint at {severity}% support contamination ({feature_set.replace('_', ' ')} features).

## Recommended format

`portable/reliability_estimator.json` and `portable/reliability_estimator.npz` provide a NumPy-only representation. The metadata fixes feature order, preprocessing, model parameters, threshold, and deletion cap.

```python
from pvse.estimators import load_portable_reliability_estimator

estimator = load_portable_reliability_estimator(
    "portable/reliability_estimator.json"
)
deleted = estimator.predicted_corrupt(feature_matrix)
```

## Author-environment snapshot

`joblib_author_env/reliability_estimator.joblib` preserves the fitted scikit-learn pipeline. Load it only from a trusted release; the portable representation is preferred for inspection and exchange.

## Verification

```bash
pvse-verify-estimator-pack {asset}
pvse-verify-estimator-pack {asset} --allow-joblib
```

The parity record checks the corresponding submitted FEAT plug-in result and portable/joblib numerical agreement.
"""
    return _finish(root, asset=asset, role=role, readme=readme, output_root=output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build individual frozen FEAT plug-in estimator assets")
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--noisy-root", required=True)
    parser.add_argument("--submitted-table7", required=True)
    parser.add_argument("--submitted-table10", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    clean_root = Path(args.clean_root).resolve()
    noisy_root = Path(args.noisy_root).resolve()
    table7_path = Path(args.submitted_table7).resolve()
    table10_path = Path(args.submitted_table10).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    table7 = _table7(table7_path)
    table10 = _table10(table10_path)
    built = [_build_clean(clean_root=clean_root, table10_path=table10_path, expected=table10[0], output_root=output_root, force=args.force)]
    for per_class, severity in ((1, 20), (2, 40), (3, 60)):
        for feature_set in ("global_plus_patch", "global_only"):
            built.append(_build_noisy(noisy_root=noisy_root, table7_path=table7_path, table10_path=table10_path, table7=table7, table10=table10, severity=severity, per_class=per_class, feature_set=feature_set, output_root=output_root, force=args.force))
    print(json.dumps({"passed": True, "assets": built}, indent=2))


if __name__ == "__main__":
    main()
