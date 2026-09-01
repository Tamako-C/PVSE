from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pvse.estimators import load_portable_reliability_estimator
from pvse.noisy.features import RELIABILITY_FEATURES
from pvse.noisy.reliability import PAPER_RELIABILITY_CONFIGS, ReliabilityConfig, fit_reliability_estimator
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
MAIN_ASSETS = {
    20: "pvse_r_soft_minimagenet_noise20_v1",
    40: "pvse_r_soft_minimagenet_noise40_v1",
    60: "pvse_r_soft_minimagenet_noise60_v1",
}
BUDGET_ASSETS = {
    "perclass1": "pvse_r_budget_perclass1_v1",
    "total3": "pvse_r_budget_total3_v1",
}


def _read_frame(path: Path, *, expected_rows: int, level: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "corruption_level" in frame.columns:
        frame = frame.loc[frame["corruption_level"].astype(str) == level].copy()
    frame = frame.reset_index(drop=True)
    if len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows in {path.name}, found {len(frame)}")
    missing = sorted(set(RELIABILITY_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"missing reliability features in {path.name}: {missing}")
    return frame


def _fit(frame: pd.DataFrame, config: ReliabilityConfig):
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


def _query_metrics(path: Path, *, baseline_method: str, method: str) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=["episode", "query_index", "method", "correct"])
    baseline = frame.loc[frame["method"].astype(str) == baseline_method]
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
    help_count = int((~base & edit).sum())
    hurt_count = int((base & ~edit).sum())
    return {
        "episodes": 1_000,
        "queries": 75_000,
        "baseline_accuracy_percent": float(100.0 * base.mean()),
        "method_accuracy_percent": float(100.0 * edit.mean()),
        "gain_pp": float(100.0 * np.mean(edit.astype(int) - base.astype(int))),
        "help": help_count,
        "hurt": hurt_count,
        "net": help_count - hurt_count,
    }


def _table5(path: Path) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for _, row in pd.read_csv(path).iterrows():
        severity = int(str(row["Noise"]).rstrip("%"))
        output[severity] = {
            "baseline": float(str(row["Noisy A0"]).rstrip("%")),
            "accuracy": float(str(row["PVSE-R-Soft"]).rstrip("%")),
            "gain": float(str(row["Gain (episode CI)"]).split()[0]),
            "net": int(str(row["Net"]).lstrip("+")),
            "support_f1": float(row["Support F1"]),
        }
    if set(output) != {20, 40, 60}:
        raise ValueError("submitted Table 5 does not contain 20/40/60 rows")
    return output


def _assert_table5(metrics: Mapping[str, Any], expected: Mapping[str, float]) -> None:
    checks = (
        round(float(metrics["baseline_accuracy_percent"]), 3) == expected["baseline"],
        round(float(metrics["method_accuracy_percent"]), 3) == expected["accuracy"],
        round(float(metrics["gain_pp"]), 3) == expected["gain"],
        int(metrics["net"]) == expected["net"],
    )
    if not all(checks):
        raise AssertionError("formal Table 5 parity failed")


def _budget_metrics(budget_root: Path, supplement_root: Path) -> dict[str, dict[str, Any]]:
    mappings = {
        "perclass_soft": (supplement_root / "perclass1_budget_le3_query_outputs.csv", "Noisy A0 corrupt_per_class_1", "V418F soft weighted corrupt_per_class_1"),
        "perclass_budget": (supplement_root / "perclass1_budget_le3_query_outputs.csv", "Noisy A0 corrupt_per_class_1", "N-HD-budget<=3 threshold-cap val-selected"),
        "perclass_top3": (supplement_root / "perclass1_budget_le3_query_outputs.csv", "Noisy A0 corrupt_per_class_1", "N-HD-topk-global k=3 reference"),
        "perclass_r1": (budget_root / "noisy_harddelete_query_outputs.csv", "Noisy A0", "N-HD-perclass-r=1"),
        "total3_soft_transfer": (supplement_root / "total3_budget_le3_query_outputs.csv", "Noisy A0 total3", "V418F soft weighted perclass-model transfer to total3"),
        "total3_budget": (supplement_root / "total3_budget_le3_query_outputs.csv", "Noisy A0 total3", "N-HD-budget<=3 threshold-cap total3 val-selected"),
        "total3_top3": (supplement_root / "total3_budget_le3_query_outputs.csv", "Noisy A0 total3", "N-HD-topk-global k=3 total3 reference"),
    }
    return {key: _query_metrics(path, baseline_method=baseline, method=method) for key, (path, baseline, method) in mappings.items()}


def _assert_table6(path: Path, metrics: Mapping[str, Mapping[str, Any]]) -> None:
    mapping = {
        ("per-class-1", "Soft weighting"): "perclass_soft",
        ("per-class-1", "Budget<=3 hard delete"): "perclass_budget",
        ("per-class-1", "Fixed top-3 delete"): "perclass_top3",
        ("per-class-1", "Per-class delete r = 1"): "perclass_r1",
        ("corrupt-total-3", "Soft transfer"): "total3_soft_transfer",
        ("corrupt-total-3", "Budget<=3 hard delete"): "total3_budget",
        ("corrupt-total-3", "Fixed top-3 delete"): "total3_top3",
    }
    frame = pd.read_csv(path)
    if len(frame) != 7:
        raise ValueError("submitted Table 6 must contain seven rows")
    for _, row in frame.iterrows():
        key = mapping[(str(row["Protocol"]), str(row["Method"]))]
        expected = float(str(row["Gain"]).lstrip("+"))
        if round(float(metrics[key]["gain_pp"]), 3) != expected:
            raise AssertionError(f"formal Table 6 parity failed for {key}")


def _source_row(source_id: str, role: str, path: Path, rows: int, *, fit: bool = False, selection: bool = False) -> dict[str, Any]:
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
    paper_scope: Sequence[str],
    parity: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
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
        paper_scope=paper_scope,
        joblib_file=joblib_relative,
    )
    portable = load_portable_reliability_estimator(metadata)
    matrix = np.random.default_rng(3920).normal(size=(257, len(portable.features)))
    maximum_error = float(np.max(np.abs(portable.predict_corruption_probability(matrix) - bundle.model.predict_proba(matrix)[:, 1])))
    if maximum_error > 1e-6:
        raise AssertionError("portable and joblib probabilities differ")
    write_json(root / "parity" / "formal_parity.json", {"formal_results": dict(parity), "portable_joblib_max_error": maximum_error})
    write_json(
        root / "provenance" / "build_environment.json",
        {"python": platform.python_version(), "platform": platform.system(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    )
    write_source_manifest(root / "provenance" / "source_files.csv", sources)
    (root / "README.md").write_text(
        f"""# {asset_name}

This asset contains one frozen PVSE support-reliability estimator used by {', '.join(paper_scope)}.

## Recommended format

`portable/reliability_estimator.json` and `portable/reliability_estimator.npz` provide a NumPy-only representation. The metadata fixes feature order, preprocessing, model parameters, and the associated soft-weight or hard-deletion policy.

```python
from pvse.estimators import load_portable_reliability_estimator

estimator = load_portable_reliability_estimator(
    "portable/reliability_estimator.json"
)
probability = estimator.predict_corruption_probability(feature_matrix)
```

## Author-environment snapshot

`joblib_author_env/reliability_estimator.joblib` preserves the fitted scikit-learn pipeline. Joblib files can execute Python objects while loading; load this snapshot only when it comes from a trusted release. The portable representation is preferred for inspection and exchange.

## Verification

```bash
pvse-verify-estimator-pack {asset_name}
pvse-verify-estimator-pack {asset_name} --allow-joblib
```

The first command verifies packaged checksums and portable inference. The second additionally checks portable output against the author-environment snapshot. The parity record checks the corresponding formal query outputs against the submitted display values.
""",
        encoding="utf-8",
    )
    write_pack_manifests(root, asset=asset_name, logical_role=logical_role)
    zip_path = output_root / f"{asset_name}.zip"
    return {"asset": asset_name, "zip": str(zip_path), "sha256": write_deterministic_zip(root, zip_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build individual frozen PVSE-R assets for paper Tables 5 and 6")
    parser.add_argument("--main-noisy-root", required=True)
    parser.add_argument("--budget-root", required=True)
    parser.add_argument("--budget-supplement-root", required=True)
    parser.add_argument("--budget-perclass-train", required=True)
    parser.add_argument("--budget-total3-train", required=True)
    parser.add_argument("--submitted-table5", required=True)
    parser.add_argument("--submitted-table6", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    main_root = Path(args.main_noisy_root).resolve()
    budget_root = Path(args.budget_root).resolve()
    supplement_root = Path(args.budget_supplement_root).resolve()
    table5_path = Path(args.submitted_table5).resolve()
    table6_path = Path(args.submitted_table6).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    table5 = _table5(table5_path)
    built: list[dict[str, Any]] = []

    for severity, per_class in ((20, 1), (40, 2), (60, 3)):
        level = f"corrupt_per_class_{per_class}"
        train_path = main_root / f"support_rows_{level}_train64.csv"
        selection_path = main_root / f"selected_model_{level}.json"
        query_path = main_root / f"query_outputs_{level}.csv"
        selected = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
        config = replace(PAPER_RELIABILITY_CONFIGS[severity], random_state=FIT_RANDOM_STATE)
        if selected["model"] != ("logreg" if severity == 40 else "l1_logreg"):
            raise ValueError(f"unexpected selected model for severity {severity}")
        bundle = _fit(_read_frame(train_path, expected_rows=1_250, level=level), config)
        metrics = _query_metrics(query_path, baseline_method="Noisy A0", method="V418F global+patch soft weighted same-protocol")
        _assert_table5(metrics, table5[severity])
        built.append(
            _build_asset(
                asset_name=MAIN_ASSETS[severity],
                logical_role=f"noisy.resnet12.table5.severity_{severity}.soft",
                bundle=bundle,
                policy={"beta": config.beta, "w_min": config.w_min},
                calibration={"split": "train64 and validation", "train_episodes": 50, "validation_episodes": 20, "severity_percent": severity, "test_selection": False},
                paper_scope=("Table 5 noisy severity", "Table 7 patch ablation"),
                parity={**metrics, "submitted_support_f1": table5[severity]["support_f1"]},
                sources=(
                    _source_row("training_rows", "estimator fitting", train_path, 1_250, fit=True),
                    _source_row("selected_configuration", "validation selection", selection_path, 1, selection=True),
                    _source_row("formal_query_outputs", "formal parity", query_path, 375_000),
                    _source_row("submitted_table5", "display-value reference", table5_path, 3),
                ),
                output_root=output_root,
                force=args.force,
            )
        )

    budget_parity = _budget_metrics(budget_root, supplement_root)
    _assert_table6(table6_path, budget_parity)
    budget_specs = (
        ("perclass1", Path(args.budget_perclass_train).resolve(), "corrupt_per_class_1", 20, {"beta": 1.5, "w_min": 0.0, "threshold": 0.7087526842951775, "comparison": ">=", "max_delete": 3}, {key: value for key, value in budget_parity.items() if key.startswith("perclass") or key == "total3_soft_transfer"}),
        ("total3", Path(args.budget_total3_train).resolve(), "corrupt_total_3", 12, {"threshold": 0.9420585632324219, "comparison": ">=", "max_delete": 3}, {key: value for key, value in budget_parity.items() if key.startswith("total3_") and key != "total3_soft_transfer"}),
    )
    for key, train_path, level, severity, policy, parity in budget_specs:
        config = replace(PAPER_RELIABILITY_CONFIGS[20], severity_percent=severity, beta=1.5, w_min=0.0, random_state=FIT_RANDOM_STATE)
        bundle = _fit(_read_frame(train_path, expected_rows=1_250, level=level), config)
        built.append(
            _build_asset(
                asset_name=BUDGET_ASSETS[key],
                logical_role=f"noisy.resnet12.table6.{key}",
                bundle=bundle,
                policy=policy,
                calibration={"split": "train64 and validation", "train_episodes": 50, "validation_episodes": 20, "protocol": level, "test_selection": False},
                paper_scope=("Table 6 budget ablation",),
                parity=parity,
                sources=(
                    _source_row("training_rows", "estimator fitting", train_path, 1_250, fit=True),
                    _source_row("submitted_table6", "display-value reference", table6_path, 7),
                ),
                output_root=output_root,
                force=args.force,
            )
        )
    print(json.dumps({"passed": True, "assets": built}, indent=2))


if __name__ == "__main__":
    main()
