from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np

from pvse.core.prototypes import prototype_predict
from pvse.eval.bootstrap import paired_episode_bootstrap
from pvse.eval.metrics import paired_classification_metrics, support_recovery_metrics
from pvse.experiments.miniimagenet import (
    EpisodeProtocol,
    RuntimeOptions,
    collect_reliability_rows,
    iter_episode_features,
)
from pvse.models.resnet12 import RFSResNet12Backbone
from pvse.noisy.corruption import apply_main_corruption
from pvse.noisy.features import RELIABILITY_FEATURES, reliability_feature_rows
from pvse.noisy.policies import (
    fixed_topk_keep,
    per_class_topr_keep,
    query_adaptive_budget_predict,
    threshold_cap_keep,
)
from pvse.noisy.reliability import (
    PAPER_RELIABILITY_CONFIGS,
    ReliabilityBundle,
    fit_reliability_estimator,
    reliability_weights,
    save_reliability_bundle,
)
from pvse.utils.io import write_csv, write_json, write_jsonl
from pvse.utils.seeding import stable_text_code


PERCLASS1_THRESHOLD_CAP3 = 0.7087526842951775
TOTAL3_THRESHOLD_CAP3 = 0.9420585632324219
PERCLASS1_QUERY_MARGIN_EDIT_COST = 0.01
TOTAL3_QUERY_MAXPROB_EDIT_COST = 0.02


@dataclass(frozen=True)
class BudgetRunConfig:
    data_root: str
    output_dir: str
    seed: int = 3920
    train_seed_offset: int = 222
    val_seed_offset: int = 111
    test_seed_offset: int = 0
    corruption_seed: int = 41820
    train_episodes: int = 50
    val_episodes: int = 20
    test_episodes: int = 1000
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 5200
    protocol: EpisodeProtocol = EpisodeProtocol()
    runtime: RuntimeOptions = RuntimeOptions()


def _rng(config: BudgetRunConfig, *, seed_offset: int, level: str) -> np.random.Generator:
    return np.random.default_rng(
        int(config.corruption_seed) + int(seed_offset) + stable_text_code(level)
    )


def _fit_bundle(
    backbone: RFSResNet12Backbone,
    config: BudgetRunConfig,
    *,
    level: str,
    corrupt_per_class: int | None = None,
    corrupt_total: int | None = None,
) -> tuple[ReliabilityBundle, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows, manifests, corruptions = collect_reliability_rows(
        backbone,
        data_root=config.data_root,
        split="train64",
        episodes=int(config.train_episodes),
        seed=int(config.seed),
        seed_offset=int(config.train_seed_offset),
        corruption_seed=int(config.corruption_seed),
        level_name=level,
        corrupt_per_class=corrupt_per_class,
        corrupt_total=corrupt_total,
        protocol=config.protocol,
        runtime=config.runtime,
    )
    # Both Table 6 reliability models use the paper L1/logistic configuration.
    # ``corrupt_total_3`` changes exactly 3/25 supports, hence 12% aggregate
    # contamination; the protocol label remains the authoritative identifier.
    model_config = replace(
        PAPER_RELIABILITY_CONFIGS[20],
        severity_percent=(20 if corrupt_per_class == 1 else 12),
        beta=1.5,
        w_min=0.0,
    )
    bundle = fit_reliability_estimator(
        rows,
        config=model_config,
        features=RELIABILITY_FEATURES,
    )
    return bundle, rows, manifests, corruptions


def _evaluate_protocol(
    backbone: RFSResNet12Backbone,
    config: BudgetRunConfig,
    *,
    level: str,
    protocol_label: str,
    own_bundle: ReliabilityBundle,
    transfer_bundle: ReliabilityBundle | None,
    corrupt_per_class: int | None = None,
    corrupt_total: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = _rng(config, seed_offset=config.test_seed_offset, level=level)
    query_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    corruption_rows: list[dict[str, Any]] = []

    for episode in iter_episode_features(
        backbone,
        data_root=config.data_root,
        split="test",
        episodes=int(config.test_episodes),
        seed=int(config.seed) + int(config.test_seed_offset),
        protocol=config.protocol,
        runtime=config.runtime,
    ):
        corrupted = apply_main_corruption(
            episode.support_features,
            episode.support_labels,
            rng=rng,
            corrupt_per_class=corrupt_per_class,
            corrupt_total=corrupt_total,
            support_maps=episode.support_maps,
        )
        assert corrupted.maps is not None
        metadata = {
            "split": "test",
            "corruption_level": level,
            "episode": int(episode.episode),
        }
        rows = reliability_feature_rows(
            corrupted.features,
            episode.support_labels,
            corrupted.maps,
            corrupt_mask=corrupted.corrupt_mask,
            metadata=metadata,
            way=int(config.protocol.way),
        )
        p_own = own_bundle.predict_corruption_probability(rows)
        p_transfer = (
            transfer_bundle.predict_corruption_probability(rows)
            if transfer_bundle is not None
            else p_own
        )

        baseline = np.asarray(
            prototype_predict(
                corrupted.features,
                episode.support_labels,
                episode.query_features,
                way=int(config.protocol.way),
                logit_scale=float(config.protocol.logit_scale),
            ).predictions,
            dtype=np.int64,
        )

        if corrupt_per_class == 1:
            methods: dict[str, np.ndarray] = {
                "Soft weighting": reliability_weights(p_own, beta=1.5, w_min=0.0),
                "Budget<=3 hard delete": threshold_cap_keep(
                    p_own,
                    threshold=PERCLASS1_THRESHOLD_CAP3,
                    max_delete=3,
                ),
                "Fixed top-3 delete": fixed_topk_keep(p_own, 3),
                "Per-class delete r = 1": per_class_topr_keep(
                    p_own,
                    episode.support_labels,
                    r=1,
                    way=int(config.protocol.way),
                ),
            }
            query_policy_name = "Query-margin budget<=3"
            query_decisions = query_adaptive_budget_predict(
                corrupted.features,
                episode.support_labels,
                episode.query_features,
                p_own,
                edit_cost=PERCLASS1_QUERY_MARGIN_EDIT_COST,
                score_mode="margin",
                max_delete=3,
                way=int(config.protocol.way),
                logit_scale=float(config.protocol.logit_scale),
            )
        else:
            methods = {
                "Soft transfer": reliability_weights(p_transfer, beta=1.5, w_min=0.0),
                "Budget<=3 hard delete": threshold_cap_keep(
                    p_own,
                    threshold=TOTAL3_THRESHOLD_CAP3,
                    max_delete=3,
                ),
                "Fixed top-3 delete": fixed_topk_keep(p_own, 3),
            }
            query_policy_name = "Query-maxprob budget<=3"
            query_decisions = query_adaptive_budget_predict(
                corrupted.features,
                episode.support_labels,
                episode.query_features,
                p_own,
                edit_cost=TOTAL3_QUERY_MAXPROB_EDIT_COST,
                score_mode="maxprob",
                max_delete=3,
                way=int(config.protocol.way),
                logit_scale=float(config.protocol.logit_scale),
            )

        predictions: dict[str, np.ndarray] = {}
        for method, keep in methods.items():
            predictions[method] = np.asarray(
                prototype_predict(
                    corrupted.features,
                    episode.support_labels,
                    episode.query_features,
                    weights=keep,
                    way=int(config.protocol.way),
                    logit_scale=float(config.protocol.logit_scale),
                ).predictions,
                dtype=np.int64,
            )
        predictions[query_policy_name] = np.asarray(
            [decision.prediction for decision in query_decisions], dtype=np.int64
        )

        hard_selected: dict[str, np.ndarray] = {
            name: np.asarray(keep <= 1e-6, dtype=bool)
            for name, keep in methods.items()
            if "Soft" not in name
        }
        for row, p1, pt in zip(rows, p_own, p_transfer):
            support_rows.append(
                {
                    **row,
                    "p_corrupt_protocol_model": float(p1),
                    "p_corrupt_transfer_model": float(pt),
                }
            )
        for method, selected in hard_selected.items():
            rec = support_recovery_metrics(selected, corrupted.corrupt_mask)
            support_rows.append(
                {
                    **metadata,
                    "support_index": -1,
                    "summary_row": 1,
                    "method": method,
                    **rec.to_dict(),
                }
            )

        for qi, y in enumerate(episode.query_labels.astype(int).tolist()):
            row: dict[str, Any] = {
                "split": "test",
                "protocol": protocol_label,
                "corruption_level": level,
                "episode": int(episode.episode),
                "query_index": int(qi),
                "true_label": int(y),
                "Noisy A0": int(baseline[qi]),
            }
            for name, pred in predictions.items():
                row[name] = int(pred[qi])
            row["query_selected_k"] = int(query_decisions[qi].selected_k)
            query_rows.append(row)

        for record in corrupted.records:
            corruption_rows.append({**metadata, **asdict(record)})
        manifests.append(episode.manifest)
    return query_rows, support_rows, manifests, corruption_rows


def _aggregate(
    query_rows: list[dict[str, Any]],
    *,
    protocol_label: str,
    method_names: list[str],
    config: BudgetRunConfig,
) -> list[dict[str, Any]]:
    episode_ids = np.asarray([r["episode"] for r in query_rows], dtype=np.int64)
    truth = np.asarray([r["true_label"] for r in query_rows], dtype=np.int64)
    baseline = np.asarray([r["Noisy A0"] for r in query_rows], dtype=np.int64)
    out: list[dict[str, Any]] = []
    for method in method_names:
        pred = np.asarray([r[method] for r in query_rows], dtype=np.int64)
        metrics = paired_classification_metrics(truth, baseline, pred).to_dict()
        ci = paired_episode_bootstrap(
            episode_ids,
            baseline == truth,
            pred == truth,
            samples=int(config.bootstrap_samples),
            seed=int(config.bootstrap_seed),
        ).to_dict()
        out.append(
            {
                "Protocol": protocol_label,
                "Method": method,
                **metrics,
                "episode_bootstrap": ci,
            }
        )
    return out


def run_budget_experiment(
    backbone: RFSResNet12Backbone,
    config: BudgetRunConfig,
) -> dict[str, Any]:
    """Run the two Table-6 corruption settings and their submitted policies."""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", asdict(config))

    per_bundle, per_train, per_train_manifests, per_train_corruptions = _fit_bundle(
        backbone,
        config,
        level="corrupt_per_class_1",
        corrupt_per_class=1,
    )
    total_bundle, total_train, total_train_manifests, total_train_corruptions = _fit_bundle(
        backbone,
        config,
        level="corrupt_total_3",
        corrupt_total=3,
    )
    save_reliability_bundle(per_bundle, out / "reliability_perclass1.joblib")
    save_reliability_bundle(total_bundle, out / "reliability_total3.joblib")
    write_csv(out / "support_rows_train_perclass1.csv", per_train)
    write_csv(out / "support_rows_train_total3.csv", total_train)
    write_jsonl(out / "train_manifests_perclass1.jsonl", per_train_manifests)
    write_jsonl(out / "train_manifests_total3.jsonl", total_train_manifests)
    write_csv(out / "corruption_manifest_train_perclass1.csv", per_train_corruptions)
    write_csv(out / "corruption_manifest_train_total3.csv", total_train_corruptions)

    # The submitted thresholds/edit costs are frozen constants selected on the
    # validation protocol. Preserve those validation feature rows and manifests
    # for auditability without reselecting a policy on test data.
    per_val, per_val_manifests, per_val_corruptions = collect_reliability_rows(
        backbone,
        data_root=config.data_root,
        split="val",
        episodes=int(config.val_episodes),
        seed=int(config.seed),
        seed_offset=int(config.val_seed_offset),
        corruption_seed=int(config.corruption_seed),
        level_name="corrupt_per_class_1",
        corrupt_per_class=1,
        protocol=config.protocol,
        runtime=config.runtime,
    )
    total_val, total_val_manifests, total_val_corruptions = collect_reliability_rows(
        backbone,
        data_root=config.data_root,
        split="val",
        episodes=int(config.val_episodes),
        seed=int(config.seed),
        seed_offset=int(config.val_seed_offset),
        corruption_seed=int(config.corruption_seed),
        level_name="corrupt_total_3",
        corrupt_per_class=None,
        corrupt_total=3,
        protocol=config.protocol,
        runtime=config.runtime,
    )
    write_csv(out / "support_rows_val_perclass1.csv", per_val)
    write_csv(out / "support_rows_val_total3.csv", total_val)
    write_jsonl(out / "val_manifests_perclass1.jsonl", per_val_manifests)
    write_jsonl(out / "val_manifests_total3.jsonl", total_val_manifests)
    write_csv(out / "corruption_manifest_val_perclass1.csv", per_val_corruptions)
    write_csv(out / "corruption_manifest_val_total3.csv", total_val_corruptions)

    per_q, per_s, per_m, per_c = _evaluate_protocol(
        backbone,
        config,
        level="corrupt_per_class_1",
        protocol_label="per-class-1",
        own_bundle=per_bundle,
        transfer_bundle=None,
        corrupt_per_class=1,
    )
    total_q, total_s, total_m, total_c = _evaluate_protocol(
        backbone,
        config,
        level="corrupt_total_3",
        protocol_label="corrupt-total-3",
        own_bundle=total_bundle,
        transfer_bundle=per_bundle,
        corrupt_total=3,
    )
    write_csv(out / "query_outputs_perclass1.csv", per_q)
    write_csv(out / "query_outputs_total3.csv", total_q)
    write_csv(out / "support_outputs_perclass1.csv", per_s)
    write_csv(out / "support_outputs_total3.csv", total_s)
    write_jsonl(out / "test_manifests_perclass1.jsonl", per_m)
    write_jsonl(out / "test_manifests_total3.jsonl", total_m)
    write_csv(out / "corruption_manifest_test_perclass1.csv", per_c)
    write_csv(out / "corruption_manifest_test_total3.csv", total_c)

    per_methods = [
        "Soft weighting",
        "Budget<=3 hard delete",
        "Fixed top-3 delete",
        "Per-class delete r = 1",
        "Query-margin budget<=3",
    ]
    total_methods = [
        "Soft transfer",
        "Budget<=3 hard delete",
        "Fixed top-3 delete",
        "Query-maxprob budget<=3",
    ]
    results = _aggregate(
        per_q,
        protocol_label="per-class-1",
        method_names=per_methods,
        config=config,
    ) + _aggregate(
        total_q,
        protocol_label="corrupt-total-3",
        method_names=total_methods,
        config=config,
    )
    write_json(out / "budget_results.json", results)
    flat_rows = []
    for row in results:
        flat_rows.append(
            {
                "Protocol": row["Protocol"],
                "Method": row["Method"],
                "baseline_accuracy": row["baseline_accuracy"],
                "method_accuracy": row["method_accuracy"],
                "gain_pp": row["gain_pp"],
                "help": row["help"],
                "hurt": row["hurt"],
                "net": row["net"],
                "ci_lower_pp": row["episode_bootstrap"]["lower_pp"],
                "ci_upper_pp": row["episode_bootstrap"]["upper_pp"],
            }
        )
    write_csv(out / "table6_budget_ablation_generated.csv", flat_rows)
    summary = {
        "experiment": "budget_ablation",
        "formal_scope": "CMT Table 6 and Supplement Tables S8-S9",
        "fixed_paper_policies": {
            "perclass1_threshold_cap3": PERCLASS1_THRESHOLD_CAP3,
            "total3_threshold_cap3": TOTAL3_THRESHOLD_CAP3,
            "perclass1_query_margin_edit_cost": PERCLASS1_QUERY_MARGIN_EDIT_COST,
            "total3_query_maxprob_edit_cost": TOTAL3_QUERY_MAXPROB_EDIT_COST,
        },
        "results": results,
    }
    write_json(out / "summary.json", summary)
    return summary
