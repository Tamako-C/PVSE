from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
import torch

from pvse.clean.features import clean_feature_rows
from pvse.clean.pvse_c import apply_clean_episode
from pvse.clean.verifier import (
    CleanVerifierBundle,
    CleanVerifierConfig,
    fit_clean_verifier,
    save_clean_verifier,
)
from pvse.core.prototypes import DEFAULT_LOGIT_SCALE, prototype_predict
from pvse.data.miniimagenet import MiniImageNetEpisodeSampler, MiniImageNetSplit
from pvse.eval.bootstrap import paired_episode_bootstrap
from pvse.eval.metrics import (
    mean_episode_support_recovery,
    paired_classification_metrics,
)
from pvse.experiments.common import EpisodeFeatures, extract_loaded_episode
from pvse.models.resnet12 import RFSResNet12Backbone
from pvse.noisy.corruption import CorruptedSupport, apply_main_corruption
from pvse.noisy.features import GLOBAL_RELIABILITY_FEATURES, RELIABILITY_FEATURES, reliability_feature_rows
from pvse.noisy.reliability import (
    PAPER_RELIABILITY_CONFIGS,
    ReliabilityBundle,
    ReliabilityConfig,
    fit_reliability_estimator,
    save_reliability_bundle,
)
from pvse.noisy.weighted_proto import weighted_proto_predict
from pvse.utils.io import write_csv, write_json, write_jsonl
from pvse.utils.seeding import stable_text_code


@dataclass(frozen=True)
class EpisodeProtocol:
    way: int = 5
    shot: int = 5
    query: int = 15
    image_size: int = 84
    logit_scale: float = DEFAULT_LOGIT_SCALE


@dataclass(frozen=True)
class RuntimeOptions:
    device: str = "cuda"
    batch_size: int = 96
    amp: bool = False
    channels_last: bool = False


@dataclass(frozen=True)
class SplitSpec:
    split: MiniImageNetSplit
    episodes: int
    seed: int
    seed_offset: int

    @property
    def effective_seed(self) -> int:
        return int(self.seed) + int(self.seed_offset)


@dataclass(frozen=True)
class CleanRunConfig:
    data_root: str
    output_dir: str
    seed: int = 3920
    val_seed_offset: int = 111
    test_seed_offset: int = 0
    val_episodes: int = 300
    test_episodes: int = 1000
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 5200
    kmax: int = 3
    edit_cost: float = 0.0
    run_a0_only: bool = True
    run_full: bool = True
    protocol: EpisodeProtocol = EpisodeProtocol()
    runtime: RuntimeOptions = RuntimeOptions()
    verifier: CleanVerifierConfig = CleanVerifierConfig()


@dataclass(frozen=True)
class NoisyRunConfig:
    data_root: str
    output_dir: str
    severity_percent: int
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
    feature_set: Literal["global_plus_patch", "global_only"] = "global_plus_patch"
    protocol: EpisodeProtocol = EpisodeProtocol()
    runtime: RuntimeOptions = RuntimeOptions()

    @property
    def corrupt_per_class(self) -> int:
        if int(self.severity_percent) not in {20, 40, 60}:
            raise ValueError("main noisy severity must be 20, 40, or 60")
        return int(self.severity_percent) // 20

    @property
    def level_name(self) -> str:
        return f"corrupt_per_class_{self.corrupt_per_class}"


@dataclass(frozen=True)
class NoisyEpisode:
    clean: EpisodeFeatures
    corrupted: CorruptedSupport


def iter_episode_features(
    backbone: RFSResNet12Backbone,
    *,
    data_root: str | Path,
    split: MiniImageNetSplit,
    episodes: int,
    seed: int,
    protocol: EpisodeProtocol = EpisodeProtocol(),
    runtime: RuntimeOptions = RuntimeOptions(),
) -> Iterator[EpisodeFeatures]:
    sampler = MiniImageNetEpisodeSampler(
        data_root,
        split,
        seed=int(seed),
        image_size=int(protocol.image_size),
    )
    for _ in range(int(episodes)):
        loaded = sampler.sample(
            way=int(protocol.way),
            shot=int(protocol.shot),
            query=int(protocol.query),
        )
        yield extract_loaded_episode(
            loaded,
            backbone,
            device=runtime.device,
            batch_size=int(runtime.batch_size),
            amp=bool(runtime.amp),
            channels_last=bool(runtime.channels_last),
        )


def collect_clean_calibration_rows(
    backbone: RFSResNet12Backbone,
    *,
    data_root: str | Path,
    episodes: int = 300,
    seed: int = 3920,
    seed_offset: int = 111,
    protocol: EpisodeProtocol = EpisodeProtocol(),
    runtime: RuntimeOptions = RuntimeOptions(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for episode in iter_episode_features(
        backbone,
        data_root=data_root,
        split="val",
        episodes=int(episodes),
        seed=int(seed) + int(seed_offset),
        protocol=protocol,
        runtime=runtime,
    ):
        metadata = {
            "split": "val",
            "episode": int(episode.episode),
            "seed": int(seed),
            "seed_offset": int(seed_offset),
        }
        rows.extend(
            clean_feature_rows(
                episode.support_features,
                episode.support_labels,
                episode.query_features,
                episode.support_maps,
                episode.query_maps,
                true_labels=episode.query_labels,
                metadata=metadata,
                way=int(protocol.way),
                logit_scale=float(protocol.logit_scale),
                proposal_score_mode="probability",
            )
        )
        manifests.append(episode.manifest)
    return rows, manifests


def _clean_method_predictions(decisions: Sequence, method: str) -> np.ndarray:
    values: list[int] = []
    for decision in decisions:
        if method == "switch":
            values.append(int(decision.switch_prediction))
        elif method == "hd_a0_only":
            values.append(
                int(decision.hard_delete_a0_only.final_prediction)
                if decision.hard_delete_a0_only is not None
                else int(decision.a0_prediction)
            )
        elif method == "hd_full":
            values.append(
                int(decision.hard_delete_full.final_prediction)
                if decision.hard_delete_full is not None
                else int(decision.a0_prediction)
            )
        else:
            raise ValueError(method)
    return np.asarray(values, dtype=np.int64)


def run_clean_experiment(
    backbone: RFSResNet12Backbone,
    config: CleanRunConfig,
) -> dict[str, Any]:
    """Run the paper-scoped miniImageNet clean calibration and test pipeline.

    This is a real image-to-result path. It does not substitute synthetic
    verifier probabilities or oracle labels at test time. The caller must load
    the frozen paper checkpoint into ``backbone`` before invoking this function.
    """
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", asdict(config))

    calibration_rows, val_manifests = collect_clean_calibration_rows(
        backbone,
        data_root=config.data_root,
        episodes=int(config.val_episodes),
        seed=int(config.seed),
        seed_offset=int(config.val_seed_offset),
        protocol=config.protocol,
        runtime=config.runtime,
    )
    write_csv(out / "clean_calibration_rows.csv", calibration_rows)
    write_jsonl(out / "val_episode_manifests.jsonl", val_manifests)
    verifier = fit_clean_verifier(calibration_rows, config=config.verifier)
    save_clean_verifier(verifier, out / "clean_verifier.joblib")

    query_rows: list[dict[str, Any]] = []
    test_manifests: list[dict[str, Any]] = []
    for episode in iter_episode_features(
        backbone,
        data_root=config.data_root,
        split="test",
        episodes=int(config.test_episodes),
        seed=int(config.seed) + int(config.test_seed_offset),
        protocol=config.protocol,
        runtime=config.runtime,
    ):
        metadata = {
            "split": "test",
            "episode": int(episode.episode),
            "seed": int(config.seed),
            "seed_offset": int(config.test_seed_offset),
        }
        rows = clean_feature_rows(
            episode.support_features,
            episode.support_labels,
            episode.query_features,
            episode.support_maps,
            episode.query_maps,
            true_labels=episode.query_labels,
            metadata=metadata,
            way=int(config.protocol.way),
            logit_scale=float(config.protocol.logit_scale),
            proposal_score_mode="probability",
        )
        decisions = apply_clean_episode(
            episode.support_features,
            episode.support_labels,
            episode.query_features,
            rows,
            verifier,
            run_a0_only=bool(config.run_a0_only),
            run_full=bool(config.run_full),
            kmax=int(config.kmax),
            edit_cost=float(config.edit_cost),
            way=int(config.protocol.way),
            logit_scale=float(config.protocol.logit_scale),
        )
        a0 = np.asarray([int(row["a0_pred"]) for row in rows], dtype=np.int64)
        proposal = np.asarray([int(row["proposal_class"]) for row in rows], dtype=np.int64)
        score = verifier.score(rows)
        accepted = score > float(verifier.config.threshold)
        switch = _clean_method_predictions(decisions, "switch")
        hd_a0 = _clean_method_predictions(decisions, "hd_a0_only") if config.run_a0_only else a0.copy()
        hd_full = _clean_method_predictions(decisions, "hd_full") if config.run_full else a0.copy()
        for qi, y in enumerate(episode.query_labels.astype(int).tolist()):
            decision = decisions[qi]
            query_rows.append(
                {
                    "split": "test",
                    "episode": int(episode.episode),
                    "query_index": int(qi),
                    "true_label": int(y),
                    "a0_pred": int(a0[qi]),
                    "proposal_class": int(proposal[qi]),
                    "verifier_score": float(score[qi]),
                    "accepted": int(accepted[qi]),
                    "switch_pred": int(switch[qi]),
                    "hd_a0_only_pred": int(hd_a0[qi]),
                    "hd_full_pred": int(hd_full[qi]),
                    "hd_a0_only_deleted": (
                        list(decision.hard_delete_a0_only.selected_action)
                        if decision.hard_delete_a0_only is not None
                        else []
                    ),
                    "hd_full_deleted": (
                        list(decision.hard_delete_full.selected_action)
                        if decision.hard_delete_full is not None
                        else []
                    ),
                }
            )
        test_manifests.append(episode.manifest)

    write_jsonl(out / "test_episode_manifests.jsonl", test_manifests)
    write_csv(out / "clean_query_outputs.csv", query_rows)
    episode_ids = np.asarray([row["episode"] for row in query_rows], dtype=np.int64)
    truth = np.asarray([row["true_label"] for row in query_rows], dtype=np.int64)
    baseline = np.asarray([row["a0_pred"] for row in query_rows], dtype=np.int64)
    methods = {
        "A0 prototype": baseline,
        "PVSE-C-Switch reference": np.asarray([row["switch_pred"] for row in query_rows], dtype=np.int64),
        "PVSE-C-HD A0-only": np.asarray([row["hd_a0_only_pred"] for row in query_rows], dtype=np.int64),
        "PVSE-C-HD FULL": np.asarray([row["hd_full_pred"] for row in query_rows], dtype=np.int64),
    }
    results: list[dict[str, Any]] = []
    baseline_correct = baseline == truth
    for method, predictions in methods.items():
        metrics = paired_classification_metrics(truth, baseline, predictions).to_dict()
        if method == "A0 prototype":
            ci = {
                "estimate_pp": 0.0,
                "lower_pp": 0.0,
                "upper_pp": 0.0,
                "samples": int(config.bootstrap_samples),
                "seed": int(config.bootstrap_seed),
                "resampling_unit": "episode",
            }
        else:
            ci = paired_episode_bootstrap(
                episode_ids,
                baseline_correct,
                predictions == truth,
                samples=int(config.bootstrap_samples),
                seed=int(config.bootstrap_seed),
            ).to_dict()
        results.append({"method": method, **metrics, "episode_bootstrap": ci})
    summary = {
        "experiment": "clean_test1000",
        "formal_scope": "CMT Table 4",
        "verifier_features": list(verifier.features),
        "verifier_config": asdict(verifier.config),
        "results": results,
    }
    write_json(out / "clean_results.json", summary)
    return summary


def _noisy_rng(config: NoisyRunConfig, seed_offset: int, level_name: str | None = None) -> np.random.Generator:
    level = config.level_name if level_name is None else str(level_name)
    return np.random.default_rng(
        int(config.corruption_seed) + int(seed_offset) + stable_text_code(level)
    )


def _corrupt_episode(
    episode: EpisodeFeatures,
    *,
    rng: np.random.Generator,
    corrupt_per_class: int | None = None,
    corrupt_total: int | None = None,
) -> NoisyEpisode:
    corrupted = apply_main_corruption(
        episode.support_features,
        episode.support_labels,
        rng=rng,
        corrupt_per_class=corrupt_per_class,
        corrupt_total=corrupt_total,
        support_maps=episode.support_maps,
    )
    assert corrupted.maps is not None
    return NoisyEpisode(clean=episode, corrupted=corrupted)


def collect_reliability_rows(
    backbone: RFSResNet12Backbone,
    *,
    data_root: str | Path,
    split: MiniImageNetSplit,
    episodes: int,
    seed: int,
    seed_offset: int,
    corruption_seed: int,
    level_name: str,
    corrupt_per_class: int | None,
    corrupt_total: int | None = None,
    protocol: EpisodeProtocol = EpisodeProtocol(),
    runtime: RuntimeOptions = RuntimeOptions(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(
        int(corruption_seed) + int(seed_offset) + stable_text_code(str(level_name))
    )
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    corruption_records: list[dict[str, Any]] = []
    for episode in iter_episode_features(
        backbone,
        data_root=data_root,
        split=split,
        episodes=int(episodes),
        seed=int(seed) + int(seed_offset),
        protocol=protocol,
        runtime=runtime,
    ):
        noisy = _corrupt_episode(
            episode,
            rng=rng,
            corrupt_per_class=corrupt_per_class,
            corrupt_total=corrupt_total,
        )
        assert noisy.corrupted.maps is not None
        metadata = {
            "split": str(split),
            "corruption_level": str(level_name),
            "episode": int(episode.episode),
            "seed": int(seed),
            "seed_offset": int(seed_offset),
            "corruption_seed": int(corruption_seed),
        }
        rows.extend(
            reliability_feature_rows(
                noisy.corrupted.features,
                episode.support_labels,
                noisy.corrupted.maps,
                corrupt_mask=noisy.corrupted.corrupt_mask,
                metadata=metadata,
                way=int(protocol.way),
            )
        )
        manifests.append(episode.manifest)
        for record in noisy.corrupted.records:
            corruption_records.append(
                {
                    **metadata,
                    **asdict(record),
                    "operation": "copy/replacement",
                    "target_observed_label_retained": True,
                    "source_support_retained": True,
                }
            )
    return rows, manifests, corruption_records


def _reliability_features_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "global_plus_patch":
        return RELIABILITY_FEATURES
    if mode == "global_only":
        return GLOBAL_RELIABILITY_FEATURES
    raise ValueError(f"unsupported feature_set: {mode}")


def _paper_reliability_config_for_run(severity_percent: int, feature_set: str) -> ReliabilityConfig:
    """Bind the immutable paper hyperparameters to the executed feature branch.

    The numerical model is unchanged; this ensures serialized run metadata
    records ``global_only`` for the Table-7 ablation instead of inheriting the
    default ``global_plus_patch`` label from the shared paper configuration.
    """
    return replace(
        PAPER_RELIABILITY_CONFIGS[int(severity_percent)],
        feature_set=str(feature_set),
    )


def run_noisy_experiment(
    backbone: RFSResNet12Backbone,
    config: NoisyRunConfig,
) -> dict[str, Any]:
    """Run one main noisy severity from images through PVSE-R-Soft."""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", asdict(config))
    reliability_config = _paper_reliability_config_for_run(
        int(config.severity_percent), config.feature_set
    )
    features = _reliability_features_for_mode(config.feature_set)

    train_rows, train_manifests, train_corruptions = collect_reliability_rows(
        backbone,
        data_root=config.data_root,
        split="train64",
        episodes=int(config.train_episodes),
        seed=int(config.seed),
        seed_offset=int(config.train_seed_offset),
        corruption_seed=int(config.corruption_seed),
        level_name=config.level_name,
        corrupt_per_class=int(config.corrupt_per_class),
        protocol=config.protocol,
        runtime=config.runtime,
    )
    write_csv(out / "support_rows_train.csv", train_rows)
    write_jsonl(out / "train_episode_manifests.jsonl", train_manifests)
    write_csv(out / "corruption_manifest_train.csv", train_corruptions)
    bundle = fit_reliability_estimator(
        train_rows,
        config=reliability_config,
        features=features,
    )
    save_reliability_bundle(bundle, out / "reliability_bundle.joblib")

    val_rows, val_manifests, val_corruptions = collect_reliability_rows(
        backbone,
        data_root=config.data_root,
        split="val",
        episodes=int(config.val_episodes),
        seed=int(config.seed),
        seed_offset=int(config.val_seed_offset),
        corruption_seed=int(config.corruption_seed),
        level_name=config.level_name,
        corrupt_per_class=int(config.corrupt_per_class),
        protocol=config.protocol,
        runtime=config.runtime,
    )
    write_csv(out / "support_rows_val.csv", val_rows)
    write_jsonl(out / "val_episode_manifests.jsonl", val_manifests)
    write_csv(out / "corruption_manifest_val.csv", val_corruptions)

    test_rng = _noisy_rng(config, int(config.test_seed_offset))
    query_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    test_manifests: list[dict[str, Any]] = []
    corruption_records: list[dict[str, Any]] = []
    selected_masks: list[np.ndarray] = []
    true_masks: list[np.ndarray] = []
    for episode in iter_episode_features(
        backbone,
        data_root=config.data_root,
        split="test",
        episodes=int(config.test_episodes),
        seed=int(config.seed) + int(config.test_seed_offset),
        protocol=config.protocol,
        runtime=config.runtime,
    ):
        noisy = _corrupt_episode(
            episode,
            rng=test_rng,
            corrupt_per_class=int(config.corrupt_per_class),
        )
        assert noisy.corrupted.maps is not None
        metadata = {
            "split": "test",
            "corruption_level": config.level_name,
            "episode": int(episode.episode),
        }
        rows = reliability_feature_rows(
            noisy.corrupted.features,
            episode.support_labels,
            noisy.corrupted.maps,
            corrupt_mask=noisy.corrupted.corrupt_mask,
            metadata=metadata,
            way=int(config.protocol.way),
        )
        p_corrupt = bundle.predict_corruption_probability(rows)
        weights = bundle.support_weights(rows)
        baseline = prototype_predict(
            noisy.corrupted.features,
            episode.support_labels,
            episode.query_features,
            way=int(config.protocol.way),
            logit_scale=float(config.protocol.logit_scale),
        ).predictions
        method = weighted_proto_predict(
            noisy.corrupted.features,
            episode.support_labels,
            episode.query_features,
            weights,
            way=int(config.protocol.way),
            logit_scale=float(config.protocol.logit_scale),
        ).predictions
        baseline = np.asarray(baseline, dtype=np.int64)
        method = np.asarray(method, dtype=np.int64)
        for row, prob, weight in zip(rows, p_corrupt, weights):
            support_rows.append({**row, "p_corrupt": float(prob), "support_weight": float(weight)})
        top_k = int(noisy.corrupted.corrupt_mask.sum())
        order = np.argsort(-p_corrupt, kind="mergesort")
        selected = np.zeros(len(p_corrupt), dtype=bool)
        selected[order[:top_k]] = True
        selected_masks.append(selected)
        true_masks.append(noisy.corrupted.corrupt_mask.copy())
        for qi, y in enumerate(episode.query_labels.astype(int).tolist()):
            query_rows.append(
                {
                    "split": "test",
                    "corruption_level": config.level_name,
                    "episode": int(episode.episode),
                    "query_index": int(qi),
                    "true_label": int(y),
                    "noisy_a0_pred": int(baseline[qi]),
                    "pvse_r_soft_pred": int(method[qi]),
                    "noisy_a0_correct": int(baseline[qi] == y),
                    "pvse_r_soft_correct": int(method[qi] == y),
                }
            )
        for record in noisy.corrupted.records:
            corruption_records.append({**metadata, **asdict(record)})
        test_manifests.append(episode.manifest)

    write_csv(out / "support_rows_test.csv", support_rows)
    write_csv(out / "corruption_manifest_test.csv", corruption_records)
    write_jsonl(out / "test_episode_manifests.jsonl", test_manifests)
    write_csv(out / "noisy_query_outputs.csv", query_rows)

    episode_ids = np.asarray([row["episode"] for row in query_rows], dtype=np.int64)
    truth = np.asarray([row["true_label"] for row in query_rows], dtype=np.int64)
    baseline = np.asarray([row["noisy_a0_pred"] for row in query_rows], dtype=np.int64)
    method = np.asarray([row["pvse_r_soft_pred"] for row in query_rows], dtype=np.int64)
    metrics = paired_classification_metrics(truth, baseline, method).to_dict()
    ci = paired_episode_bootstrap(
        episode_ids,
        baseline == truth,
        method == truth,
        samples=int(config.bootstrap_samples),
        seed=int(config.bootstrap_seed),
    ).to_dict()
    support_metrics = mean_episode_support_recovery(selected_masks, true_masks)
    summary = {
        "experiment": "noisy_severity",
        "formal_scope": "CMT Table 5" if config.feature_set == "global_plus_patch" else "CMT Table 7 global-only ablation",
        "severity_percent": int(config.severity_percent),
        "corruption_level": config.level_name,
        "corruption_protocol": {
            "source_scope": "within_episode_other_observed_class",
            "operation": "copy/replacement",
            "target_label_retained": True,
            "source_retained": True,
            "targets_sampled_before_sources": True,
        },
        "reliability_config": asdict(reliability_config),
        "features": list(features),
        "metrics": metrics,
        "episode_bootstrap": ci,
        **support_metrics,
    }
    write_json(out / "noisy_results.json", summary)
    return summary
