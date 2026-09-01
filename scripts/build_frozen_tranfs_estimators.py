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

from pvse.estimators import load_portable_reliability_estimator
from pvse.noisy.features import RELIABILITY_FEATURES
from pvse.noisy.reliability import ReliabilityConfig, fit_reliability_estimator
from pvse.release.estimator_pack import (
    prepare_output,
    sha256_file,
    write_binary_estimator,
    write_deterministic_zip,
    write_json,
    write_pack_manifests,
    write_source_manifest,
)


FIT_RANDOM_STATE = 3920
PROTOCOLS = ("sym_swap", "pair_swap")
SEVERITIES = (20, 40, 60)


def _asset_name(protocol: str, severity: int, branch: str) -> str:
    protocol_name = "symmetric" if protocol == "sym_swap" else "paired"
    return f"pvse_r_{branch}_tranfs_{protocol_name}_noise{severity}_v1"


def _model_config(kind: str, severity: int) -> ReliabilityConfig:
    if kind == "l1_logreg":
        return ReliabilityConfig(severity, "liblinear", "l1", 0.5, 1.0, 0.0, random_state=FIT_RANDOM_STATE)
    if kind == "logreg":
        return ReliabilityConfig(severity, "lbfgs", "l2", 1.0, 1.0, 0.0, random_state=FIT_RANDOM_STATE)
    raise ValueError(f"unsupported model kind: {kind}")


def _fit(path: Path, *, kind: str, severity: int):
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != 1_250 or frame["episode"].nunique() != 50:
        raise ValueError(f"expected 50 episodes and 1,250 rows in {path.name}")
    config = _model_config(kind, severity)
    rows = frame.loc[:, [*RELIABILITY_FEATURES, "corrupted_for_analysis_only"]].to_dict(orient="records")
    return fit_reliability_estimator(rows, config=config, features=RELIABILITY_FEATURES)


def _model_metadata(config: ReliabilityConfig) -> dict[str, Any]:
    return {
        "estimator": "LogisticRegression",
        "scaler": "StandardScaler",
        "penalty": config.penalty,
        "solver": config.solver,
        "C": config.C,
        "class_weight": config.class_weight,
        "max_iter": config.max_iter,
        "fit_random_state": config.random_state,
    }


def _query_metrics(path: Path, method: str) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=["episode", "query_index", "method", "correct"])
    baseline = frame.loc[frame["method"].astype(str) == "Noisy A0"]
    edited = frame.loc[frame["method"].astype(str) == method]
    paired = baseline[["episode", "query_index", "correct"]].merge(
        edited[["episode", "query_index", "correct"]],
        on=["episode", "query_index"],
        validate="one_to_one",
        suffixes=("_baseline", "_method"),
    )
    if len(paired) != 75_000 or paired["episode"].nunique() != 1_000:
        raise ValueError(f"formal query coverage is incomplete in {path.name}")
    base = paired["correct_baseline"].astype(bool).to_numpy()
    edit = paired["correct_method"].astype(bool).to_numpy()
    return {
        "episodes": 1_000,
        "queries": 75_000,
        "baseline_accuracy_percent": float(100.0 * base.mean()),
        "method_accuracy_percent": float(100.0 * edit.mean()),
        "gain_pp": float(100.0 * np.mean(edit.astype(int) - base.astype(int))),
    }


def _paper_rows(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    protocol_names = {"symmetric replacement": "sym_swap", "paired replacement": "pair_swap"}
    output: dict[tuple[str, int], dict[str, float]] = {}
    for _, row in pd.read_csv(path).iterrows():
        key = (protocol_names[str(row["Protocol"])], int(str(row["Noise"]).rstrip("%")))
        output[key] = {
            "baseline": float(str(row["Noisy A0"]).rstrip("%")),
            "soft_accuracy": float(str(row["PVSE-R-Soft acc."]).rstrip("%")),
            "soft_gain": float(str(row["PVSE-R-Soft gain (episode CI)"]).split()[0]),
            "hd_accuracy": float(str(row["PVSE-R-HD acc."]).rstrip("%")),
            "hd_gain": float(str(row["PVSE-R-HD gain (episode CI)"]).split()[0]),
        }
    expected_keys = {(protocol, severity) for protocol in PROTOCOLS for severity in SEVERITIES}
    if set(output) != expected_keys:
        raise ValueError("submitted Table 8 does not contain the expected six rows")
    return output


def _assert_metrics(actual: Mapping[str, Any], expected: Mapping[str, float], branch: str) -> None:
    if not (
        round(float(actual["baseline_accuracy_percent"]), 3) == expected["baseline"]
        and round(float(actual["method_accuracy_percent"]), 3) == expected[f"{branch}_accuracy"]
        and round(float(actual["gain_pp"]), 3) == expected[f"{branch}_gain"]
    ):
        raise AssertionError(f"formal Table 8 parity failed for {branch}")


def _source(source_id: str, role: str, path: Path, rows: int, *, fit: bool = False, selection: bool = False) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "role": role,
        "sha256": sha256_file(path),
        "rows_used": rows,
        "used_for_fit": str(fit).lower(),
        "used_for_model_selection": str(selection).lower(),
    }


def _build_asset(
    *,
    asset_name: str,
    logical_role: str,
    bundle: Any,
    policy: Mapping[str, Any],
    calibration: Mapping[str, Any],
    parity: Mapping[str, Any],
    sources: list[dict[str, Any]],
    output_root: Path,
    force: bool,
) -> dict[str, str]:
    root = prepare_output(output_root / asset_name, expected_name=asset_name, force=force)
    joblib_relative = "joblib_author_env/reliability_estimator.joblib"
    joblib_path = root / joblib_relative
    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"format": "pvse_reliability_v1", "features": list(bundle.features), "config": asdict(bundle.config), "model": bundle.model}, joblib_path)
    metadata = write_binary_estimator(
        root / "portable",
        estimator_id=logical_role,
        file_stem="reliability_estimator",
        arrays_stem="reliability_estimator",
        logical_role=logical_role,
        model=bundle.model,
        features=bundle.features,
        model_metadata=_model_metadata(bundle.config),
        policy=policy,
        calibration=calibration,
        paper_scope=("Table 8 TraNFS-style support replacement",),
        joblib_file=joblib_relative,
    )
    portable = load_portable_reliability_estimator(metadata)
    matrix = np.random.default_rng(3920).normal(size=(257, len(portable.features)))
    maximum_error = float(np.max(np.abs(portable.predict_corruption_probability(matrix) - bundle.model.predict_proba(matrix)[:, 1])))
    if maximum_error > 1e-6:
        raise AssertionError("portable and joblib probabilities differ")
    write_json(root / "parity" / "formal_test1000_parity.json", {"formal_results": dict(parity), "portable_joblib_max_error": maximum_error})
    write_json(
        root / "provenance" / "build_environment.json",
        {"python": platform.python_version(), "platform": platform.system(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    )
    write_source_manifest(root / "provenance" / "source_files.csv", sources)
    (root / "README.md").write_text(
        f"""# {asset_name}

This asset contains one frozen PVSE-R estimator and its fixed policy for a submitted TraNFS-style support-replacement row. Calibration uses the matching training and validation severity; formal test episodes are used only for parity evaluation.

## Recommended format

`portable/reliability_estimator.json` and `portable/reliability_estimator.npz` provide a NumPy-only representation. The metadata fixes feature order, preprocessing, model parameters, and the severity-specific soft-weight or hard-deletion policy.

```python
from pvse.estimators import load_portable_reliability_estimator

estimator = load_portable_reliability_estimator(
    "portable/reliability_estimator.json"
)
probability = estimator.predict_corruption_probability(feature_matrix)
```

## Author-environment snapshot

`joblib_author_env/reliability_estimator.joblib` preserves the fitted scikit-learn pipeline. Load it only from a trusted release; the portable representation is preferred for inspection and exchange.

## Verification

```bash
pvse-verify-estimator-pack {asset_name}
pvse-verify-estimator-pack {asset_name} --allow-joblib
```

The first command verifies packaged checksums and portable inference. The second additionally checks portable output against the author-environment snapshot. The parity record covers the corresponding 1,000-episode Table 8 evaluation row.
""",
        encoding="utf-8",
    )
    write_pack_manifests(root, asset=asset_name, logical_role=logical_role)
    zip_path = output_root / f"{asset_name}.zip"
    return {"asset": asset_name, "zip": str(zip_path), "sha256": write_deterministic_zip(root, zip_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build individual severity-specific TraNFS-style PVSE-R assets")
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--submitted-table8", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    formal_root = Path(args.formal_root).resolve()
    table_path = Path(args.submitted_table8).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    expected_rows = _paper_rows(table_path)
    built: list[dict[str, str]] = []

    for protocol in PROTOCOLS:
        for severity in SEVERITIES:
            level = f"{protocol}_{severity}"
            train_path = formal_root / f"support_rows_{level}_train64.csv"
            val_path = formal_root / f"support_rows_{level}_val.csv"
            selection_path = formal_root / f"selected_configs_{level}.csv"
            query_path = formal_root / f"query_outputs_{level}.csv"
            selection = pd.read_csv(selection_path)
            val = pd.read_csv(val_path, usecols=["episode"])
            if len(val) != 500 or val["episode"].nunique() != 20:
                raise ValueError(f"expected 20 validation episodes in {val_path.name}")
            for selector, method, branch in (
                ("soft_weighted", "PVSE-R-Soft global_plus_patch", "soft"),
                ("budget_le3_hd", "PVSE-R-HD budget<=3 global_plus_patch", "hd"),
            ):
                selected = selection.loc[selection["selector_type"].astype(str) == selector]
                if len(selected) != 1:
                    raise ValueError(f"invalid {selector} selection in {selection_path.name}")
                row = selected.iloc[0]
                bundle = _fit(train_path, kind=str(row["model"]), severity=severity)
                policy = (
                    {"beta": float(row["beta"]), "w_min": float(row["w_min"])}
                    if branch == "soft"
                    else {"threshold": float(row["threshold"]), "comparison": ">", "max_delete": 3}
                )
                metrics = _query_metrics(query_path, method)
                _assert_metrics(metrics, expected_rows[(protocol, severity)], branch)
                asset_name = _asset_name(protocol, severity, branch)
                built.append(
                    _build_asset(
                        asset_name=asset_name,
                        logical_role=f"tranfs_style.resnet12.table8.{level}.{branch}",
                        bundle=bundle,
                        policy=policy,
                        calibration={"split": "train64 and validation", "train_episodes": 50, "validation_episodes": 20, "protocol": protocol, "severity_percent": severity, "test_selection": False},
                        parity=metrics,
                        sources=[
                            _source("training_rows", "estimator fitting", train_path, 1_250, fit=True),
                            _source("validation_rows", "model and policy selection", val_path, 500, selection=True),
                            _source("selected_configuration", "fixed selected configuration", selection_path, 2, selection=True),
                            _source("formal_query_outputs", "formal parity", query_path, 450_000),
                            _source("submitted_table8", "display-value reference", table_path, 6),
                        ],
                        output_root=output_root,
                        force=args.force,
                    )
                )
    print(json.dumps({"passed": True, "assets": built}, indent=2))


if __name__ == "__main__":
    main()
