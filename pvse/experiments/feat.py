from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from pvse.eval.bootstrap import paired_episode_bootstrap
from pvse.eval.metrics import (
    mean_episode_support_recovery,
    paired_classification_metrics,
)
from pvse.plugins.feat import (
    FEAT_CLEAN_FEATURES,
    FEAT_CLEAN_LAMBDA_HURT,
    FEAT_CLEAN_THRESHOLD,
    FeatAdapter,
    FeatCleanGatePolicy,
    FeatProtocol,
    build_feat_transform,
    corrupt_feat_support_images,
    feat_clean_calibration_grid,
    feat_clean_feature_dict,
    feat_clean_hard_delete,
    feat_support_reliability_rows,
    fit_feat_clean_gate,
    fit_select_feat_reliability,
    normalize_feat_patches,
    retained_support_by_class,
    save_feat_clean_gate,
    save_feat_reliability,
    support_by_class,
)
from pvse.utils.io import write_csv, write_json, write_jsonl


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class FeatRunConfig:
    data_root: str
    feat_root: str
    checkpoint: str
    output_dir: str
    mode: Literal["clean", "noisy", "both"] = "both"
    device: str = "cuda"
    batch_size: int = 100
    clean_val_episodes: int = 300
    clean_test_episodes: int = 600
    clean_val_seed: int = 517103
    clean_test_seed: int = 517003
    noisy_train_episodes: int = 50
    noisy_val_episodes: int = 50
    noisy_test_episodes: int = 1000
    noisy_train_seed: int = 519100
    noisy_val_seed: int = 519200
    noisy_test_seed: int = 517003
    severities: tuple[int, ...] = (1, 2, 3)
    max_delete: int = 3
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 5200
    protocol: FeatProtocol = FeatProtocol()


@dataclass(frozen=True)
class FeatEpisodeManifest:
    episode: int
    split: str
    class_names: tuple[str, ...]
    support_paths: tuple[str, ...]
    support_labels: tuple[int, ...]
    query_paths: tuple[str, ...]
    query_labels: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": int(self.episode),
            "split": self.split,
            "class_names": list(self.class_names),
            "support_paths": list(self.support_paths),
            "support_labels": list(self.support_labels),
            "query_paths": list(self.query_paths),
            "query_labels": list(self.query_labels),
        }


class FeatEpisodeSampler:
    """Python-``random`` episode sampler for the reported FEAT protocol."""

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "val", "test"],
        *,
        seed: int,
        protocol: FeatProtocol = FeatProtocol(),
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.split = split
        self.root = self.data_root / split / split
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"miniImageNet FEAT split not found: {self.root}; expected train/train, val/val, test/test"
            )
        self.protocol = protocol
        self.rng = random.Random(int(seed))
        self.index: dict[str, tuple[Path, ...]] = {}
        for class_dir in sorted((p for p in self.root.iterdir() if p.is_dir()), key=lambda p: p.name):
            images = tuple(
                path
                for path in sorted(class_dir.iterdir(), key=lambda p: p.name)
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if len(images) >= int(protocol.shot) + int(protocol.query):
                self.index[class_dir.name] = images
        if len(self.index) < int(protocol.way):
            raise RuntimeError(f"only {len(self.index)} eligible FEAT classes under {self.root}")
        self._episode = 0

    def sample(self) -> FeatEpisodeManifest:
        classes = self.rng.sample(list(self.index.keys()), int(self.protocol.way))
        support_paths: list[str] = []
        support_labels: list[int] = []
        query_paths: list[str] = []
        query_labels: list[int] = []
        for local_label, class_name in enumerate(classes):
            picks = self.rng.sample(
                list(self.index[class_name]),
                int(self.protocol.shot) + int(self.protocol.query),
            )
            support_paths.extend(str(path.relative_to(self.data_root)) for path in picks[: self.protocol.shot])
            query_paths.extend(str(path.relative_to(self.data_root)) for path in picks[self.protocol.shot :])
            support_labels.extend([local_label] * int(self.protocol.shot))
            query_labels.extend([local_label] * int(self.protocol.query))
        manifest = FeatEpisodeManifest(
            episode=int(self._episode),
            split=self.split,
            class_names=tuple(classes),
            support_paths=tuple(support_paths),
            support_labels=tuple(support_labels),
            query_paths=tuple(query_paths),
            query_labels=tuple(query_labels),
        )
        self._episode += 1
        return manifest

    def resolve(self, paths: Sequence[str]) -> list[Path]:
        return [self.data_root / path for path in paths]


def _load_images(paths: Sequence[Path], transform: Any) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
    if not tensors:
        raise ValueError("empty image list")
    return torch.stack(tensors, dim=0)


def _encode_batches(
    adapter: FeatAdapter,
    images: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    maps: list[torch.Tensor] = []
    for start in range(0, len(images), int(batch_size)):
        encoded = adapter.encode(images[start : start + int(batch_size)])
        embeddings.append(encoded.embeddings)
        maps.append(encoded.spatial_maps)
    return torch.cat(embeddings, dim=0), torch.cat(maps, dim=0)


def _clean_calibration_rows(
    adapter: FeatAdapter,
    config: FeatRunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sampler = FeatEpisodeSampler(
        config.data_root,
        "val",
        seed=int(config.clean_val_seed),
        protocol=config.protocol,
    )
    transform = build_feat_transform(config.protocol)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for _ in range(int(config.clean_val_episodes)):
        manifest = sampler.sample()
        support_paths = sampler.resolve(manifest.support_paths)
        query_paths = sampler.resolve(manifest.query_paths)
        images = _load_images([*support_paths, *query_paths], transform)
        embeddings, maps = _encode_batches(adapter, images, batch_size=config.batch_size)
        n_support = len(manifest.support_labels)
        by_class = support_by_class(manifest.support_labels, way=config.protocol.way)
        query_indices = list(range(n_support, len(images)))
        logits = adapter.native_logits(embeddings, by_class, query_indices)
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        support_patches = normalize_feat_patches(maps[:n_support])
        query_patches = normalize_feat_patches(maps[n_support:])
        for query_index, true_label in enumerate(manifest.query_labels):
            order = np.argsort(-probabilities[query_index], kind="mergesort")
            a0 = int(order[0])
            proposal = int(order[1])
            feature_row = feat_clean_feature_dict(
                query_patches[query_index],
                support_patches,
                manifest.support_labels,
                a0_prediction=a0,
                proposal=proposal,
                probabilities=probabilities[query_index],
                way=config.protocol.way,
            )
            a0_correct = int(a0 == int(true_label))
            rows.append(
                {
                    "split": "val",
                    "episode": int(manifest.episode),
                    "query_index": int(query_index),
                    "true_label": int(true_label),
                    "a0_pred": a0,
                    "proposal_class": proposal,
                    "a0_correct": a0_correct,
                    "help_label": int(not a0_correct and proposal == int(true_label)),
                    "hurt_label": int(a0_correct and proposal != int(true_label)),
                    **feature_row,
                }
            )
        manifests.append(manifest.to_dict())
    return rows, manifests


def _aggregate_query_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_key: str,
    method_key: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    episodes = np.asarray([int(row["episode"]) for row in rows], dtype=np.int64)
    truth = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
    baseline = np.asarray([int(row[baseline_key]) for row in rows], dtype=np.int64)
    method = np.asarray([int(row[method_key]) for row in rows], dtype=np.int64)
    paired = paired_classification_metrics(truth, baseline, method)
    bootstrap = paired_episode_bootstrap(
        episodes,
        baseline == truth,
        method == truth,
        samples=int(bootstrap_samples),
        seed=int(bootstrap_seed),
    )
    return {**paired.to_dict(), "episode_bootstrap": bootstrap.to_dict()}


def run_feat_clean(
    adapter: FeatAdapter,
    config: FeatRunConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration_rows, val_manifests = _clean_calibration_rows(adapter, config)
    write_csv(out / "clean_val_rows.csv", calibration_rows)
    write_jsonl(out / "clean_val_episode_manifests.jsonl", val_manifests)
    gate = fit_feat_clean_gate(
        calibration_rows,
        policy=FeatCleanGatePolicy(
            lambda_hurt=FEAT_CLEAN_LAMBDA_HURT,
            threshold=FEAT_CLEAN_THRESHOLD,
        ),
    )
    save_feat_clean_gate(gate, out / "clean_gate.joblib")
    grid = feat_clean_calibration_grid(gate, calibration_rows)
    write_csv(out / "clean_val_grid.csv", grid)

    sampler = FeatEpisodeSampler(
        config.data_root,
        "test",
        seed=int(config.clean_test_seed),
        protocol=config.protocol,
    )
    transform = build_feat_transform(config.protocol)
    query_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for _ in range(int(config.clean_test_episodes)):
        manifest = sampler.sample()
        support_paths = sampler.resolve(manifest.support_paths)
        query_paths = sampler.resolve(manifest.query_paths)
        images = _load_images([*support_paths, *query_paths], transform)
        embeddings, maps = _encode_batches(adapter, images, batch_size=config.batch_size)
        n_support = len(manifest.support_labels)
        by_class = support_by_class(manifest.support_labels, way=config.protocol.way)
        query_embedding_indices = list(range(n_support, len(images)))
        logits = adapter.native_logits(embeddings, by_class, query_embedding_indices)
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        support_patches = normalize_feat_patches(maps[:n_support])
        query_patches = normalize_feat_patches(maps[n_support:])
        episode_features: list[dict[str, Any]] = []
        base_records: list[tuple[int, int, int]] = []
        for query_index, true_label in enumerate(manifest.query_labels):
            order = np.argsort(-probabilities[query_index], kind="mergesort")
            a0 = int(order[0])
            proposal = int(order[1])
            features = feat_clean_feature_dict(
                query_patches[query_index],
                support_patches,
                manifest.support_labels,
                a0_prediction=a0,
                proposal=proposal,
                probabilities=probabilities[query_index],
                way=config.protocol.way,
            )
            episode_features.append(features)
            base_records.append((int(true_label), a0, proposal))
        p_help, p_hurt = gate.probabilities(episode_features)
        scores = gate.score(episode_features)
        accepted = gate.accepted(episode_features)
        for query_index, ((true_label, a0, proposal), features) in enumerate(
            zip(base_records, episode_features)
        ):
            switch_prediction = proposal if bool(accepted[query_index]) else a0
            decision = feat_clean_hard_delete(
                adapter,
                embeddings,
                manifest.support_labels,
                query_index=n_support + query_index,
                a0_prediction=a0,
                proposal=proposal,
                accepted=bool(accepted[query_index]),
                max_delete=int(config.max_delete),
            )
            query_rows.append(
                {
                    "split": "test",
                    "episode": int(manifest.episode),
                    "query_index": int(query_index),
                    "true_label": true_label,
                    "a0_pred": a0,
                    "proposal_class": proposal,
                    "switch_pred": int(switch_prediction),
                    "pvse_hd_pred": int(decision.prediction),
                    "gate_accepted": int(accepted[query_index]),
                    "hd_applied": int(decision.applied),
                    "deleted_k": len(decision.deleted_indices),
                    "deleted_indices": list(decision.deleted_indices),
                    "hd_proposal_margin": decision.proposal_margin,
                    "p_help": float(p_help[query_index]),
                    "p_hurt": float(p_hurt[query_index]),
                    "gate_score": float(scores[query_index]),
                    **features,
                }
            )
        manifests.append(manifest.to_dict())
    write_jsonl(out / "clean_query_outputs.jsonl", query_rows)
    write_jsonl(out / "clean_test_episode_manifests.jsonl", manifests)
    summary = {
        "protocol": "frozen_FEAT_clean_plugin",
        "test_episodes": int(config.clean_test_episodes),
        "query_count": len(query_rows),
        "gate_policy": asdict(gate.policy),
        "feature_columns": list(FEAT_CLEAN_FEATURES),
        "calibration_grid_best": grid[0] if grid else None,
        "FEAT native": _aggregate_query_predictions(
            query_rows,
            baseline_key="a0_pred",
            method_key="a0_pred",
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
        ),
        "FEAT + PVSE-C-Switch": _aggregate_query_predictions(
            query_rows,
            baseline_key="a0_pred",
            method_key="switch_pred",
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
        ),
        "FEAT + PVSE-C-HD": _aggregate_query_predictions(
            query_rows,
            baseline_key="a0_pred",
            method_key="pvse_hd_pred",
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
        ),
        "apply_count": int(sum(int(row["hd_applied"]) for row in query_rows)),
        "mean_deleted_supports_per_query": float(
            np.mean([int(row["deleted_k"]) for row in query_rows])
        ),
    }
    write_json(out / "clean_summary.json", summary)
    return summary


def _collect_noisy_support_rows(
    adapter: FeatAdapter,
    config: FeatRunConfig,
    *,
    split: Literal["train", "val"],
    episodes: int,
    seed: int,
    severity: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # The reported FEAT protocol adds 1000*severity inside each train/val stream.
    sampler = FeatEpisodeSampler(
        config.data_root,
        split,
        seed=int(seed) + 1000 * int(severity),
        protocol=config.protocol,
    )
    # The same Random object drives episode sampling and image-copy corruption.
    corruption_rng = sampler.rng
    transform = build_feat_transform(config.protocol)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    corruptions: list[dict[str, Any]] = []
    for _ in range(int(episodes)):
        manifest = sampler.sample()
        support = _load_images(sampler.resolve(manifest.support_paths), transform)
        noisy, corrupt_mask, metadata = corrupt_feat_support_images(
            support,
            manifest.support_labels,
            corrupt_per_class=int(severity),
            rng=corruption_rng,
            way=config.protocol.way,
        )
        embeddings, maps = _encode_batches(adapter, noisy, batch_size=config.batch_size)
        prefix = {
            "split": split,
            "episode": int(manifest.episode),
            "severity_r_per_class": int(severity),
        }
        rows.extend(
            feat_support_reliability_rows(
                embeddings,
                maps,
                manifest.support_labels,
                corrupt_mask=corrupt_mask,
                metadata=prefix,
                way=config.protocol.way,
            )
        )
        corruptions.extend({**prefix, **record} for record in metadata)
        manifests.append(manifest.to_dict())
    return rows, manifests, corruptions


def _run_feat_noisy_level(
    adapter: FeatAdapter,
    config: FeatRunConfig,
    *,
    severity: int,
    output_dir: Path,
) -> dict[str, Any]:
    level_dir = output_dir / f"r{int(severity)}"
    level_dir.mkdir(parents=True, exist_ok=True)
    train_rows, train_manifests, train_corruptions = _collect_noisy_support_rows(
        adapter,
        config,
        split="train",
        episodes=config.noisy_train_episodes,
        seed=config.noisy_train_seed,
        severity=severity,
    )
    val_rows, val_manifests, val_corruptions = _collect_noisy_support_rows(
        adapter,
        config,
        split="val",
        episodes=config.noisy_val_episodes,
        seed=config.noisy_val_seed,
        severity=severity,
    )
    write_csv(level_dir / "support_rows_train.csv", train_rows)
    write_csv(level_dir / "support_rows_val.csv", val_rows)
    write_jsonl(level_dir / "train_episode_manifests.jsonl", train_manifests)
    write_jsonl(level_dir / "val_episode_manifests.jsonl", val_manifests)
    write_csv(level_dir / "corruption_manifest_train.csv", train_corruptions)
    write_csv(level_dir / "corruption_manifest_val.csv", val_corruptions)

    patch_bundle, patch_grid = fit_select_feat_reliability(
        train_rows,
        val_rows,
        feature_sets=("global_plus_patch",),
    )
    global_bundle, global_grid = fit_select_feat_reliability(
        train_rows,
        val_rows,
        feature_sets=("global_only",),
    )
    save_feat_reliability(patch_bundle, level_dir / "reliability_global_plus_patch.joblib")
    save_feat_reliability(global_bundle, level_dir / "reliability_global_only.joblib")
    write_csv(level_dir / "selection_grid_global_plus_patch.csv", patch_grid)
    write_csv(level_dir / "selection_grid_global_only.csv", global_grid)

    sampler = FeatEpisodeSampler(
        config.data_root,
        "test",
        seed=int(config.noisy_test_seed) + 1000 * int(severity),
        protocol=config.protocol,
    )
    corruption_rng = sampler.rng
    transform = build_feat_transform(config.protocol)
    query_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    corruption_rows: list[dict[str, Any]] = []
    patch_masks: list[np.ndarray] = []
    global_masks: list[np.ndarray] = []
    true_masks: list[np.ndarray] = []
    for _ in range(int(config.noisy_test_episodes)):
        manifest = sampler.sample()
        support = _load_images(sampler.resolve(manifest.support_paths), transform)
        queries = _load_images(sampler.resolve(manifest.query_paths), transform)
        noisy_support, corrupt_mask, metadata = corrupt_feat_support_images(
            support,
            manifest.support_labels,
            corrupt_per_class=int(severity),
            rng=corruption_rng,
            way=config.protocol.way,
        )
        images = torch.cat([noisy_support, queries], dim=0)
        embeddings, maps = _encode_batches(adapter, images, batch_size=config.batch_size)
        n_support = len(manifest.support_labels)
        prefix = {
            "split": "test",
            "episode": int(manifest.episode),
            "severity_r_per_class": int(severity),
        }
        feature_rows = feat_support_reliability_rows(
            embeddings[:n_support],
            maps[:n_support],
            manifest.support_labels,
            corrupt_mask=corrupt_mask,
            metadata=prefix,
            way=config.protocol.way,
        )
        p_patch = patch_bundle.probabilities(feature_rows)
        p_global = global_bundle.probabilities(feature_rows)
        deleted_patch = patch_bundle.deleted_mask(feature_rows, max_delete=config.max_delete)
        deleted_global = global_bundle.deleted_mask(feature_rows, max_delete=config.max_delete)
        by_class = retained_support_by_class(
            manifest.support_labels,
            way=config.protocol.way,
        )
        patch_by_class = retained_support_by_class(
            manifest.support_labels,
            deleted_patch,
            way=config.protocol.way,
        )
        global_by_class = retained_support_by_class(
            manifest.support_labels,
            deleted_global,
            way=config.protocol.way,
        )
        query_indices = list(range(n_support, len(images)))
        baseline_logits = adapter.native_logits(embeddings, by_class, query_indices)
        patch_logits = adapter.native_logits(embeddings, patch_by_class, query_indices)
        global_logits = adapter.native_logits(embeddings, global_by_class, query_indices)
        baseline = baseline_logits.argmax(dim=1).detach().cpu().numpy()
        patch_prediction = patch_logits.argmax(dim=1).detach().cpu().numpy()
        global_prediction = global_logits.argmax(dim=1).detach().cpu().numpy()
        for feature_row, pp, pg, dp, dg in zip(
            feature_rows, p_patch, p_global, deleted_patch, deleted_global
        ):
            support_rows.append(
                {
                    **feature_row,
                    "p_corrupt_global_plus_patch": float(pp),
                    "p_corrupt_global_only": float(pg),
                    "deleted_global_plus_patch": int(dp),
                    "deleted_global_only": int(dg),
                }
            )
        for query_index, true_label in enumerate(manifest.query_labels):
            query_rows.append(
                {
                    **prefix,
                    "query_index": int(query_index),
                    "true_label": int(true_label),
                    "noisy_a0_pred": int(baseline[query_index]),
                    "pvse_hd_pred": int(patch_prediction[query_index]),
                    "global_only_pred": int(global_prediction[query_index]),
                    "deleted_global_plus_patch": int(deleted_patch.sum()),
                    "deleted_global_only": int(deleted_global.sum()),
                }
            )
        patch_masks.append(deleted_patch)
        global_masks.append(deleted_global)
        true_masks.append(corrupt_mask)
        manifests.append(manifest.to_dict())
        corruption_rows.extend({**prefix, **record} for record in metadata)
    write_jsonl(level_dir / "query_outputs.jsonl", query_rows)
    write_csv(level_dir / "support_outputs.csv", support_rows)
    write_jsonl(level_dir / "test_episode_manifests.jsonl", manifests)
    write_csv(level_dir / "corruption_manifest_test.csv", corruption_rows)

    patch_metrics = _aggregate_query_predictions(
        query_rows,
        baseline_key="noisy_a0_pred",
        method_key="pvse_hd_pred",
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    global_metrics = _aggregate_query_predictions(
        query_rows,
        baseline_key="noisy_a0_pred",
        method_key="global_only_pred",
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    baseline_metrics = _aggregate_query_predictions(
        query_rows,
        baseline_key="noisy_a0_pred",
        method_key="noisy_a0_pred",
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    summary = {
        "severity_r_per_class": int(severity),
        "noise_percent": int(severity) * 20,
        "protocol": "frozen_FEAT_with_PVSE_image_copy_replacement",
        "test_episodes": int(config.noisy_test_episodes),
        "query_count": len(query_rows),
        "Noisy FEAT native": baseline_metrics,
        "FEAT + PVSE-R-HD global+patch": {
            **patch_metrics,
            **mean_episode_support_recovery(patch_masks, true_masks),
            "selection": asdict(patch_bundle.selection),
            "mean_deleted_supports": float(np.mean([mask.sum() for mask in patch_masks])),
        },
        "FEAT + PVSE-R-HD global-only": {
            **global_metrics,
            **mean_episode_support_recovery(global_masks, true_masks),
            "selection": asdict(global_bundle.selection),
            "mean_deleted_supports": float(np.mean([mask.sum() for mask in global_masks])),
        },
    }
    write_json(level_dir / "summary.json", summary)
    return summary


def run_feat_noisy(
    adapter: FeatAdapter,
    config: FeatRunConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    levels = [
        _run_feat_noisy_level(
            adapter,
            config,
            severity=int(severity),
            output_dir=out,
        )
        for severity in config.severities
    ]
    summary = {
        "protocol": "frozen_FEAT_noisy_plugin",
        "PVSE-R-Soft": "not run because FEAT exposes no documented support sample-weight semantics",
        "levels": levels,
    }
    write_json(out / "noisy_summary.json", summary)
    return summary


def run_feat_experiment(config: FeatRunConfig) -> dict[str, Any]:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", asdict(config))
    requested_device = str(config.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for FEAT but torch.cuda.is_available() is false")
    adapter = FeatAdapter.load(
        config.feat_root,
        config.checkpoint,
        device=requested_device,
        protocol=config.protocol,
    )
    write_json(
        out / "checkpoint_report.json",
        asdict(adapter.checkpoint_report) if adapter.checkpoint_report is not None else {},
    )
    result: dict[str, Any] = {
        "mode": config.mode,
        "checkpoint_report": (
            asdict(adapter.checkpoint_report) if adapter.checkpoint_report is not None else None
        ),
    }
    if config.mode in {"clean", "both"}:
        result["clean"] = run_feat_clean(adapter, config, out / "clean")
    if config.mode in {"noisy", "both"}:
        result["noisy"] = run_feat_noisy(adapter, config, out / "noisy")
    write_json(out / "summary.json", result)
    return result
