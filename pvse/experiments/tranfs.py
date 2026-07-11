from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from pvse.core.prototypes import DEFAULT_LOGIT_SCALE, l2_normalize, softmax_np
from pvse.eval.bootstrap import paired_episode_bootstrap
from pvse.eval.metrics import paired_classification_metrics, support_recovery_metrics
from pvse.experiments.miniimagenet import EpisodeProtocol, RuntimeOptions, iter_episode_features
from pvse.models.resnet12 import RFSResNet12Backbone
from pvse.noisy.features import RELIABILITY_FEATURES, reliability_feature_rows
from pvse.noisy.reliability import (
    ReliabilityBundle,
    ReliabilityConfig,
    fit_reliability_estimator,
    reliability_weights,
    save_reliability_bundle,
)
from pvse.protocols.tranfs import TraNFSCorruption, apply_tranfs_swap, normalize_protocol_name
from pvse.utils.io import write_csv, write_json, write_jsonl
from pvse.utils.seeding import stable_text_code


@dataclass(frozen=True)
class TraNFSSelectedPolicy:
    soft_model: Literal["logreg", "l1_logreg"]
    beta: float
    w_min: float
    hd_model: Literal["logreg", "l1_logreg"]
    hd_threshold: float


PAPER_TRANFS_POLICIES: dict[tuple[str, int], TraNFSSelectedPolicy] = {
    ("sym_swap", 20): TraNFSSelectedPolicy("logreg", 2.0, 0.0, "l1_logreg", 0.7539909601211549),
    ("sym_swap", 40): TraNFSSelectedPolicy("l1_logreg", 1.0, 0.5, "logreg", 1.553866604808718e-05),
    ("sym_swap", 60): TraNFSSelectedPolicy("l1_logreg", 1.0, 0.0, "logreg", 0.9997226923704148),
    ("pair_swap", 20): TraNFSSelectedPolicy("l1_logreg", 1.5, 0.0, "l1_logreg", 9.673499152995646e-05),
    ("pair_swap", 40): TraNFSSelectedPolicy("l1_logreg", 1.0, 0.0, "l1_logreg", 4.336991423770087e-06),
    ("pair_swap", 60): TraNFSSelectedPolicy("l1_logreg", 1.0, 0.0, "logreg", 2.2522890503751114e-06),
}


@dataclass(frozen=True)
class TraNFSRunConfig:
    data_root: str
    output_dir: str
    seed: int = 3920
    corruption_seed: int = 41820
    train_seed_offset: int = 516220
    val_seed_offset: int = 516110
    test_seed_offset: int = 516000
    train_episodes: int = 50
    val_episodes: int = 20
    test_episodes: int = 1000
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 5200
    protocols: tuple[str, ...] = ("sym_swap", "pair_swap")
    severities: tuple[int, ...] = (20, 40, 60)
    protocol: EpisodeProtocol = EpisodeProtocol()
    runtime: RuntimeOptions = RuntimeOptions()


@dataclass(frozen=True)
class TraNFSEpisode:
    episode: int
    clean_support: np.ndarray
    noisy: TraNFSCorruption
    query_features: np.ndarray
    query_labels: np.ndarray
    manifest: dict[str, Any]


def _model_config(kind: str, severity: int) -> ReliabilityConfig:
    if kind == "l1_logreg":
        return ReliabilityConfig(
            severity_percent=int(severity),
            solver="liblinear",
            penalty="l1",
            C=0.5,
            beta=1.0,
            w_min=0.0,
        )
    if kind == "logreg":
        return ReliabilityConfig(
            severity_percent=int(severity),
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            beta=1.0,
            w_min=0.0,
        )
    raise ValueError(f"unsupported model kind: {kind}")


def _pool_protocol(config: TraNFSRunConfig) -> EpisodeProtocol:
    return EpisodeProtocol(
        way=config.protocol.way,
        shot=2 * config.protocol.shot,
        query=config.protocol.query,
        image_size=config.protocol.image_size,
        logit_scale=config.protocol.logit_scale,
    )


def _clean_pool_indices(way: int, shot: int) -> np.ndarray:
    return np.concatenate(
        [np.arange(c * 2 * shot, c * 2 * shot + shot) for c in range(int(way))]
    ).astype(np.int64)


def iter_tranfs_episodes(
    backbone: RFSResNet12Backbone,
    config: TraNFSRunConfig,
    *,
    split: Literal["train64", "val", "test"],
    episodes: int,
    seed_offset: int,
    protocol_name: str,
    severity: int,
) -> Iterator[TraNFSEpisode]:
    protocol_name = normalize_protocol_name(protocol_name)
    level = f"{protocol_name}_{int(severity)}"
    rng = np.random.default_rng(
        int(config.corruption_seed) + int(seed_offset) + stable_text_code(level)
    )
    clean_indices = _clean_pool_indices(config.protocol.way, config.protocol.shot)
    for episode in iter_episode_features(
        backbone,
        data_root=config.data_root,
        split=split,
        episodes=int(episodes),
        seed=int(config.seed) + int(seed_offset),
        protocol=_pool_protocol(config),
        runtime=config.runtime,
    ):
        noisy = apply_tranfs_swap(
            episode.support_features,
            episode.support_maps,
            protocol=protocol_name,
            severity=int(severity),
            rng=rng,
            ways=int(config.protocol.way),
            shot=int(config.protocol.shot),
        )
        yield TraNFSEpisode(
            episode=int(episode.episode),
            clean_support=episode.support_features[clean_indices].copy(),
            noisy=noisy,
            query_features=episode.query_features,
            query_labels=episode.query_labels,
            manifest=episode.manifest,
        )


def collect_tranfs_rows(
    backbone: RFSResNet12Backbone,
    config: TraNFSRunConfig,
    *,
    split: Literal["train64", "val"],
    episodes: int,
    seed_offset: int,
    protocol_name: str,
    severity: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    corruptions: list[dict[str, Any]] = []
    level = f"{normalize_protocol_name(protocol_name)}_{int(severity)}"
    for ep in iter_tranfs_episodes(
        backbone,
        config,
        split=split,
        episodes=episodes,
        seed_offset=seed_offset,
        protocol_name=protocol_name,
        severity=severity,
    ):
        metadata = {
            "split": split,
            "corruption_level": level,
            "protocol": normalize_protocol_name(protocol_name),
            "severity": int(severity),
            "episode": ep.episode,
        }
        rows.extend(
            reliability_feature_rows(
                ep.noisy.support_features,
                ep.noisy.support_labels,
                ep.noisy.support_maps,
                corrupt_mask=ep.noisy.corrupt_mask,
                metadata=metadata,
                way=int(config.protocol.way),
            )
        )
        manifests.append(ep.manifest)
        corruptions.extend({**metadata, **record} for record in ep.noisy.metadata)
    return rows, manifests, corruptions


def _tranfs_predict(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    *,
    weights: np.ndarray,
    way: int,
    logit_scale: float,
) -> np.ndarray:
    """Protocol-specific weighted prototype prediction for Table 8.

    A zero-weight class is assigned an unavailable logit rather than invoking
    the main-experiment fallback. The selected paper configurations do not
    normally reach this edge case, but retaining it preserves source parity.
    """
    support = l2_normalize(np.asarray(support_features, dtype=np.float32), axis=1)
    labels = np.asarray(support_labels, dtype=np.int64)
    queries = l2_normalize(np.asarray(query_features, dtype=np.float32), axis=1)
    w = np.asarray(weights, dtype=np.float32)
    protos = np.zeros((int(way), support.shape[1]), dtype=np.float32)
    available = np.zeros(int(way), dtype=bool)
    for c in range(int(way)):
        idx = np.flatnonzero(labels == c)
        wc = w[idx]
        if len(idx) and float(wc.sum()) > 1e-8:
            proto = (support[idx] * wc[:, None]).sum(axis=0) / float(wc.sum())
            protos[c] = l2_normalize(proto[None, :], axis=1)[0]
            available[c] = True
    logits = float(logit_scale) * (queries @ protos.T)
    logits[:, ~available] = -1e9
    if not available.any():
        logits[:] = 0.0
    return np.argmax(softmax_np(logits, axis=1), axis=1).astype(np.int64)


def _strict_hd_keep(probabilities: np.ndarray, threshold: float, max_delete: int = 3) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float32)
    candidates = np.flatnonzero(p > float(threshold))
    if len(candidates) > int(max_delete):
        candidates = candidates[
            np.argsort(-p[candidates], kind="mergesort")[: int(max_delete)]
        ]
    keep = np.ones(len(p), dtype=np.float32)
    keep[candidates] = 0.0
    return keep


def _evaluate_level(
    backbone: RFSResNet12Backbone,
    config: TraNFSRunConfig,
    *,
    protocol_name: str,
    severity: int,
    soft_bundle: ReliabilityBundle,
    hd_bundle: ReliabilityBundle,
    policy: TraNFSSelectedPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol_name = normalize_protocol_name(protocol_name)
    level = f"{protocol_name}_{severity}"
    query_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    corruption_rows: list[dict[str, Any]] = []
    for ep in iter_tranfs_episodes(
        backbone,
        config,
        split="test",
        episodes=int(config.test_episodes),
        seed_offset=int(config.test_seed_offset),
        protocol_name=protocol_name,
        severity=severity,
    ):
        metadata = {
            "split": "test",
            "protocol": protocol_name,
            "severity": severity,
            "corruption_level": level,
            "episode": ep.episode,
        }
        rows = reliability_feature_rows(
            ep.noisy.support_features,
            ep.noisy.support_labels,
            ep.noisy.support_maps,
            corrupt_mask=ep.noisy.corrupt_mask,
            metadata=metadata,
            way=int(config.protocol.way),
        )
        p_soft = soft_bundle.predict_corruption_probability(rows)
        p_hd = hd_bundle.predict_corruption_probability(rows)
        soft_weights = reliability_weights(
            p_soft,
            beta=float(policy.beta),
            w_min=float(policy.w_min),
        )
        hd_weights = _strict_hd_keep(p_hd, policy.hd_threshold, 3)
        oracle_weights = (~ep.noisy.corrupt_mask).astype(np.float32)
        ones = np.ones(len(ep.noisy.support_labels), dtype=np.float32)
        baseline = _tranfs_predict(
            ep.noisy.support_features,
            ep.noisy.support_labels,
            ep.query_features,
            weights=ones,
            way=config.protocol.way,
            logit_scale=config.protocol.logit_scale,
        )
        soft_pred = _tranfs_predict(
            ep.noisy.support_features,
            ep.noisy.support_labels,
            ep.query_features,
            weights=soft_weights,
            way=config.protocol.way,
            logit_scale=config.protocol.logit_scale,
        )
        hd_pred = _tranfs_predict(
            ep.noisy.support_features,
            ep.noisy.support_labels,
            ep.query_features,
            weights=hd_weights,
            way=config.protocol.way,
            logit_scale=config.protocol.logit_scale,
        )
        oracle_pred = _tranfs_predict(
            ep.noisy.support_features,
            ep.noisy.support_labels,
            ep.query_features,
            weights=oracle_weights,
            way=config.protocol.way,
            logit_scale=config.protocol.logit_scale,
        )
        hd_recovery = support_recovery_metrics(hd_weights <= 1e-6, ep.noisy.corrupt_mask)
        for row, ps, ph, ws, wh in zip(rows, p_soft, p_hd, soft_weights, hd_weights):
            support_rows.append(
                {
                    **row,
                    "p_corrupt_soft_model": float(ps),
                    "p_corrupt_hd_model": float(ph),
                    "soft_weight": float(ws),
                    "hd_keep": float(wh),
                }
            )
        support_rows.append(
            {
                **metadata,
                "support_index": -1,
                "summary_row": 1,
                **hd_recovery.to_dict(),
            }
        )
        for qi, y in enumerate(ep.query_labels.astype(int).tolist()):
            query_rows.append(
                {
                    **metadata,
                    "query_index": qi,
                    "true_label": y,
                    "noisy_a0_pred": int(baseline[qi]),
                    "pvse_r_soft_pred": int(soft_pred[qi]),
                    "pvse_r_hd_pred": int(hd_pred[qi]),
                    "oracle_pred": int(oracle_pred[qi]),
                }
            )
        manifests.append(ep.manifest)
        corruption_rows.extend({**metadata, **record} for record in ep.noisy.metadata)
    return query_rows, support_rows, manifests, corruption_rows


def _aggregate_level(
    rows: list[dict[str, Any]],
    *,
    config: TraNFSRunConfig,
) -> dict[str, Any]:
    episode_ids = np.asarray([r["episode"] for r in rows], dtype=np.int64)
    truth = np.asarray([r["true_label"] for r in rows], dtype=np.int64)
    baseline = np.asarray([r["noisy_a0_pred"] for r in rows], dtype=np.int64)
    methods = {
        "PVSE-R-Soft": np.asarray([r["pvse_r_soft_pred"] for r in rows], dtype=np.int64),
        "PVSE-R-HD": np.asarray([r["pvse_r_hd_pred"] for r in rows], dtype=np.int64),
        "Oracle": np.asarray([r["oracle_pred"] for r in rows], dtype=np.int64),
    }
    out: dict[str, Any] = {
        "Protocol": rows[0]["protocol"],
        "Noise": int(rows[0]["severity"]),
        "Noisy A0 accuracy": float((baseline == truth).mean()),
        "methods": {},
    }
    for method, pred in methods.items():
        out["methods"][method] = {
            **paired_classification_metrics(truth, baseline, pred).to_dict(),
            "episode_bootstrap": paired_episode_bootstrap(
                episode_ids,
                baseline == truth,
                pred == truth,
                samples=int(config.bootstrap_samples),
                seed=int(config.bootstrap_seed),
            ).to_dict(),
        }
    return out


def run_tranfs_experiment(
    backbone: RFSResNet12Backbone,
    config: TraNFSRunConfig,
) -> dict[str, Any]:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", asdict(config))
    all_results: list[dict[str, Any]] = []

    for raw_protocol in config.protocols:
        protocol_name = normalize_protocol_name(raw_protocol)
        for severity in config.severities:
            key = (protocol_name, int(severity))
            if key not in PAPER_TRANFS_POLICIES:
                raise ValueError(f"no submitted policy for {key}")
            policy = PAPER_TRANFS_POLICIES[key]
            level_dir = out / f"{protocol_name}_{severity}"
            level_dir.mkdir(parents=True, exist_ok=True)
            train_rows, train_manifests, train_corruptions = collect_tranfs_rows(
                backbone,
                config,
                split="train64",
                episodes=int(config.train_episodes),
                seed_offset=int(config.train_seed_offset),
                protocol_name=protocol_name,
                severity=int(severity),
            )
            val_rows, val_manifests, val_corruptions = collect_tranfs_rows(
                backbone,
                config,
                split="val",
                episodes=int(config.val_episodes),
                seed_offset=int(config.val_seed_offset),
                protocol_name=protocol_name,
                severity=int(severity),
            )
            soft_bundle = fit_reliability_estimator(
                train_rows,
                config=_model_config(policy.soft_model, severity),
                features=RELIABILITY_FEATURES,
            )
            hd_bundle = (
                soft_bundle
                if policy.hd_model == policy.soft_model
                else fit_reliability_estimator(
                    train_rows,
                    config=_model_config(policy.hd_model, severity),
                    features=RELIABILITY_FEATURES,
                )
            )
            save_reliability_bundle(soft_bundle, level_dir / "soft_bundle.joblib")
            if hd_bundle is soft_bundle:
                write_json(level_dir / "hd_bundle_alias.json", {"alias": "soft_bundle.joblib"})
            else:
                save_reliability_bundle(hd_bundle, level_dir / "hd_bundle.joblib")
            write_csv(level_dir / "support_rows_train.csv", train_rows)
            write_csv(level_dir / "support_rows_val.csv", val_rows)
            write_jsonl(level_dir / "train_manifests.jsonl", train_manifests)
            write_jsonl(level_dir / "val_manifests.jsonl", val_manifests)
            write_csv(level_dir / "corruption_manifest_train.csv", train_corruptions)
            write_csv(level_dir / "corruption_manifest_val.csv", val_corruptions)
            write_json(level_dir / "selected_policy.json", asdict(policy))

            query_rows, support_rows, test_manifests, test_corruptions = _evaluate_level(
                backbone,
                config,
                protocol_name=protocol_name,
                severity=int(severity),
                soft_bundle=soft_bundle,
                hd_bundle=hd_bundle,
                policy=policy,
            )
            write_csv(level_dir / "query_outputs.csv", query_rows)
            write_csv(level_dir / "support_outputs.csv", support_rows)
            write_jsonl(level_dir / "test_manifests.jsonl", test_manifests)
            write_csv(level_dir / "corruption_manifest_test.csv", test_corruptions)
            result = _aggregate_level(query_rows, config=config)
            result["selected_policy"] = asdict(policy)
            write_json(level_dir / "results.json", result)
            all_results.append(result)

    flat: list[dict[str, Any]] = []
    for result in all_results:
        for method, values in result["methods"].items():
            flat.append(
                {
                    "Protocol": result["Protocol"],
                    "Noise": result["Noise"],
                    "Noisy A0 accuracy": result["Noisy A0 accuracy"],
                    "Method": method,
                    "method_accuracy": values["method_accuracy"],
                    "gain_pp": values["gain_pp"],
                    "ci_lower_pp": values["episode_bootstrap"]["lower_pp"],
                    "ci_upper_pp": values["episode_bootstrap"]["upper_pp"],
                    "net": values["net"],
                }
            )
    write_csv(out / "table8_tranfs_style_generated.csv", flat)
    summary = {
        "experiment": "tranfs_style_protocol",
        "formal_scope": "CMT Table 8",
        "not_official_tranfs_model_reproduction": True,
        "description": "PVSE under the official TraNFS support replacement protocols",
        "results": all_results,
    }
    write_json(out / "summary.json", summary)
    return summary
