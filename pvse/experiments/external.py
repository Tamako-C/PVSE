from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pvse.clean.features import clean_feature_rows
from pvse.clean.pvse_c import apply_clean_episode
from pvse.clean.verifier import CleanVerifierBundle
from pvse.data.external import (
    MAIN_EXTERNAL_DATASETS,
    PAPER_EXTERNAL_SETTINGS,
    ExternalSetting,
    common_class_maps,
    sample_external_episode,
)
from pvse.eval.bootstrap import paired_episode_bootstrap
from pvse.eval.metrics import paired_classification_metrics
from pvse.experiments.common import extract_path_episode
from pvse.experiments.miniimagenet import EpisodeProtocol, RuntimeOptions
from pvse.models.resnet12 import RFSResNet12Backbone
from pvse.utils.io import write_csv, write_json, write_jsonl
from pvse.utils.seeding import stable_md5_seed


@dataclass(frozen=True)
class ExternalRunConfig:
    index_csv: str
    output_dir: str
    seed: int = 3920
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 5200
    kmax: int = 3
    edit_cost: float = 0.0
    max_episodes_per_setting: int | None = None
    protocol: EpisodeProtocol = EpisodeProtocol()
    runtime: RuntimeOptions = RuntimeOptions()


def setting_id(setting: ExternalSetting) -> str:
    return "__".join(
        (
            setting.dataset,
            setting.domain_setting,
            setting.support_domain,
            setting.query_domain,
        )
    ).replace(" ", "_")


def _decision_prediction(decision: Any, field: str) -> int:
    if field == "switch":
        return int(decision.switch_prediction)
    hd = getattr(decision, field)
    return int(hd.final_prediction) if hd is not None else int(decision.a0_prediction)


def _aggregate_dataset(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    sub = [r for r in rows if str(r["dataset"]) == dataset]
    if not sub:
        raise ValueError(f"no rows for dataset {dataset}")
    episode_ids = np.asarray([r["episode_uid"] for r in sub], dtype=object)
    truth = np.asarray([r["true_label"] for r in sub], dtype=np.int64)
    baseline = np.asarray([r["a0_pred"] for r in sub], dtype=np.int64)
    methods = {
        "PVSE-C-Switch": np.asarray([r["switch_pred"] for r in sub], dtype=np.int64),
        "PVSE-C-HD A0-only": np.asarray([r["hd_a0_only_pred"] for r in sub], dtype=np.int64),
        "PVSE-C-HD FULL": np.asarray([r["hd_full_pred"] for r in sub], dtype=np.int64),
    }
    method_rows: dict[str, Any] = {}
    for name, pred in methods.items():
        method_rows[name] = {
            **paired_classification_metrics(truth, baseline, pred).to_dict(),
            "episode_bootstrap": paired_episode_bootstrap(
                episode_ids,
                baseline == truth,
                pred == truth,
                samples=int(bootstrap_samples),
                seed=int(bootstrap_seed),
            ).to_dict(),
        }
    return {
        "Dataset": dataset,
        "episodes": len(set(episode_ids.tolist())),
        "queries": len(sub),
        "A0 accuracy": float((baseline == truth).mean()),
        "methods": method_rows,
    }


def run_external_clean_experiment(
    backbone: RFSResNet12Backbone,
    verifier: CleanVerifierBundle,
    config: ExternalRunConfig,
) -> dict[str, Any]:
    """Run the frozen miniImageNet PVSE-C verifier on the six CMT datasets."""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", asdict(config))
    index = pd.read_csv(config.index_csv)
    required = {"dataset", "domain", "class_name", "image_path"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"external index is missing columns: {sorted(missing)}")

    query_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    setting_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for setting in PAPER_EXTERNAL_SETTINGS:
        smap, qmap = common_class_maps(
            index,
            setting.dataset,
            setting.support_domain,
            setting.query_domain,
        )
        if len(smap) < int(config.protocol.way):
            skipped.append(
                {
                    **asdict(setting),
                    "reason": f"only {len(smap)} common eligible classes",
                }
            )
            continue
        rng = np.random.default_rng(
            int(config.seed)
            + stable_md5_seed(
                setting.dataset,
                setting.support_domain,
                setting.query_domain,
                setting.domain_setting,
            )
        )
        requested = int(setting.episodes)
        episodes = (
            min(requested, int(config.max_episodes_per_setting))
            if config.max_episodes_per_setting is not None
            else requested
        )
        sid = setting_id(setting)
        for episode_index in range(episodes):
            manifest = sample_external_episode(
                smap,
                qmap,
                rng=rng,
                setting=setting,
                episode=episode_index,
                way=int(config.protocol.way),
                shot=int(config.protocol.shot),
                query=int(config.protocol.query),
            )
            episode = extract_path_episode(
                manifest.support_paths,
                np.asarray(manifest.support_labels, dtype=np.int64),
                manifest.query_paths,
                np.asarray(manifest.query_labels, dtype=np.int64),
                backbone,
                episode=episode_index,
                split="external",
                manifest=manifest.to_dict(),
                device=config.runtime.device,
                image_size=int(config.protocol.image_size),
                batch_size=int(config.runtime.batch_size),
                amp=bool(config.runtime.amp),
                channels_last=bool(config.runtime.channels_last),
            )
            metadata = {
                "split": "external",
                "dataset": setting.dataset,
                "domain_setting": setting.domain_setting,
                "support_domain": setting.support_domain,
                "query_domain": setting.query_domain,
                "setting_id": sid,
                "episode": episode_index,
            }
            feature_rows = clean_feature_rows(
                episode.support_features,
                episode.support_labels,
                episode.query_features,
                episode.support_maps,
                episode.query_maps,
                true_labels=episode.query_labels,
                metadata=metadata,
                way=int(config.protocol.way),
                logit_scale=float(config.protocol.logit_scale),
                proposal_score_mode="probability_difference",
            )
            decisions = apply_clean_episode(
                episode.support_features,
                episode.support_labels,
                episode.query_features,
                feature_rows,
                verifier,
                run_a0_only=True,
                run_full=True,
                kmax=int(config.kmax),
                edit_cost=float(config.edit_cost),
                way=int(config.protocol.way),
                logit_scale=float(config.protocol.logit_scale),
            )
            scores = verifier.score(feature_rows)
            for qi, (row, decision, y) in enumerate(
                zip(feature_rows, decisions, episode.query_labels.astype(int).tolist())
            ):
                query_rows.append(
                    {
                        **metadata,
                        "episode_uid": f"{sid}:{episode_index}",
                        "query_index": qi,
                        "true_label": int(y),
                        "a0_pred": int(row["a0_pred"]),
                        "proposal_class": int(row["proposal_class"]),
                        "verifier_score": float(scores[qi]),
                        "accepted": int(decision.accepted),
                        "switch_pred": _decision_prediction(decision, "switch"),
                        "hd_a0_only_pred": _decision_prediction(decision, "hard_delete_a0_only"),
                        "hd_full_pred": _decision_prediction(decision, "hard_delete_full"),
                    }
                )
            manifests.append(manifest.to_dict())
        setting_rows.append(
            {
                **asdict(setting),
                "setting_id": sid,
                "episodes_executed": episodes,
            }
        )

    write_csv(out / "external_clean_query_outputs.csv", query_rows)
    write_jsonl(out / "external_episode_manifests.jsonl", manifests)
    write_csv(out / "external_settings_executed.csv", setting_rows)
    write_csv(out / "external_settings_skipped.csv", skipped)
    datasets = [d for d in MAIN_EXTERNAL_DATASETS if any(r["dataset"] == d for r in query_rows)]
    results = [
        _aggregate_dataset(
            query_rows,
            dataset=dataset,
            bootstrap_samples=int(config.bootstrap_samples),
            bootstrap_seed=int(config.bootstrap_seed),
        )
        for dataset in datasets
    ]
    flat: list[dict[str, Any]] = []
    for row in results:
        base = {
            "Dataset": row["Dataset"],
            "Episodes": row["episodes"],
            "Queries": row["queries"],
            "A0 accuracy": row["A0 accuracy"],
        }
        for method, values in row["methods"].items():
            flat.append(
                {
                    **base,
                    "Method": method,
                    "method_accuracy": values["method_accuracy"],
                    "gain_pp": values["gain_pp"],
                    "help": values["help"],
                    "hurt": values["hurt"],
                    "net": values["net"],
                    "ci_lower_pp": values["episode_bootstrap"]["lower_pp"],
                    "ci_upper_pp": values["episode_bootstrap"]["upper_pp"],
                }
            )
    write_csv(out / "table9_external_clean_generated.csv", flat)
    summary = {
        "experiment": "external_clean_fixed_transfer",
        "formal_scope": "CMT Table 9",
        "external_tuning": False,
        "proposal_score_serialization": "p(proposal)-p(A0), as specified by the paper transfer protocol",
        "results": results,
        "skipped_settings": skipped,
    }
    write_json(out / "external_clean_results.json", summary)
    return summary
