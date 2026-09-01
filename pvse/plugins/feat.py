from __future__ import annotations

import hashlib
import importlib
import itertools
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms


FEAT_MEAN: tuple[float, float, float] = (
    120.39586422 / 255.0,
    115.59361427 / 255.0,
    104.54012653 / 255.0,
)
FEAT_STD: tuple[float, float, float] = (
    70.68188272 / 255.0,
    68.27635443 / 255.0,
    72.54505529 / 255.0,
)

FEAT_CLEAN_FEATURES: tuple[str, ...] = (
    "a0_prob",
    "a0_margin",
    "a0_entropy",
    "proposal_prob",
    "proposal_ratio",
    "patch_score_proposal",
    "patch_score_a0",
    "patch_margin_over_a0",
    "proposal_patch_rank",
    "proposal_equals_best_patch",
    "proposal_support_agreement_count",
    "proposal_patch_entropy",
)

FEAT_NOISY_GLOBAL_FEATURES: tuple[str, ...] = (
    "cos_own_centroid_loo",
    "cos_nearest_other_centroid",
    "gap_own_vs_nearest_other",
    "support_norm",
    "nn_same_class_similarity",
    "nn_other_class_similarity",
    "own_class_compactness_loo",
)
FEAT_NOISY_PATCH_FEATURES: tuple[str, ...] = (
    "own_patch_bidir",
    "nearest_other_patch_bidir",
    "own_minus_other_patch_bidir",
    "own_patch_entropy",
    "nearest_other_patch_entropy",
    "own_support_agreement_count",
    "best_patch_class_is_not_observed",
)
FEAT_NOISY_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "global_only": FEAT_NOISY_GLOBAL_FEATURES,
    "global_plus_patch": FEAT_NOISY_GLOBAL_FEATURES + FEAT_NOISY_PATCH_FEATURES,
}

FEAT_CLEAN_LAMBDA_HURT = 1.5
FEAT_CLEAN_THRESHOLD = 0.5482710944385025
FEAT_THRESHOLD_QUANTILES: tuple[int, ...] = (0, 25, 50, 60, 70, 80, 85, 90, 95, 97, 99)


@dataclass(frozen=True)
class FeatProtocol:
    way: int = 5
    shot: int = 5
    query: int = 15
    resize: int = 92
    crop: int = 84
    use_euclidean: bool = True
    temperature: float = 64.0
    temperature2: float = 32.0


@dataclass(frozen=True)
class FeatCheckpointReport:
    feat_root: str
    checkpoint: str
    checkpoint_sha256: str
    payload_key: str
    strict: bool
    parameter_count: int


@dataclass
class FeatEncoding:
    embeddings: torch.Tensor
    spatial_maps: torch.Tensor


@dataclass(frozen=True)
class FeatCleanGatePolicy:
    lambda_hurt: float = FEAT_CLEAN_LAMBDA_HURT
    threshold: float = FEAT_CLEAN_THRESHOLD
    help_random_state: int = 517
    hurt_random_state: int = 518


@dataclass
class FeatCleanGateBundle:
    help_model: Pipeline
    hurt_model: Pipeline
    features: tuple[str, ...] = FEAT_CLEAN_FEATURES
    policy: FeatCleanGatePolicy = FeatCleanGatePolicy()

    def probabilities(self, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        x = mapping_matrix(rows, self.features)
        return positive_probability(self.help_model, x), positive_probability(self.hurt_model, x)

    def score(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        p_help, p_hurt = self.probabilities(rows)
        return p_help - float(self.policy.lambda_hurt) * p_hurt

    def accepted(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        # The reported clean FEAT gate uses an inclusive threshold.
        return self.score(rows) >= float(self.policy.threshold)


@dataclass(frozen=True)
class FeatHardDeleteDecision:
    prediction: int
    applied: bool
    deleted_indices: tuple[int, ...]
    proposal_margin: float | None


@dataclass(frozen=True)
class FeatReliabilitySelection:
    feature_set: str
    model_type: str
    threshold_quantile: int
    threshold: float
    support_precision: float
    support_recall: float
    support_f1: float
    predicted_corrupt_count: int


@dataclass
class FeatReliabilityBundle:
    model: Pipeline
    features: tuple[str, ...]
    selection: FeatReliabilitySelection

    def probabilities(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return positive_probability(self.model, mapping_matrix(rows, self.features)).astype(np.float32)

    def deleted_mask(self, rows: Sequence[Mapping[str, Any]], *, max_delete: int = 3) -> np.ndarray:
        return feat_threshold_cap_delete(
            self.probabilities(rows),
            threshold=float(self.selection.threshold),
            max_delete=int(max_delete),
        )


@contextmanager
def _prepend_sys_path(path: Path) -> Iterable[None]:
    resolved = str(path.resolve())
    sys.path.insert(0, resolved)
    try:
        yield
    finally:
        try:
            sys.path.remove(resolved)
        except ValueError:
            pass


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_feat_transform(protocol: FeatProtocol = FeatProtocol()):
    return transforms.Compose(
        [
            transforms.Resize(int(protocol.resize)),
            transforms.CenterCrop(int(protocol.crop)),
            transforms.ToTensor(),
            transforms.Normalize(mean=FEAT_MEAN, std=FEAT_STD),
        ]
    )


class FeatAdapter:
    """Strict adapter around the external FEAT implementation used in Table 10.

    Callers provide the FEAT source tree and checkpoint through the documented
    local-path interfaces. This adapter imports the upstream ``FEAT`` class,
    loads the checkpoint ``params`` payload with ``strict=True``, captures the
    encoder layer-4 spatial map, and exposes native set-to-set logits.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device,
        protocol: FeatProtocol = FeatProtocol(),
        checkpoint_report: FeatCheckpointReport | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.protocol = protocol
        self.checkpoint_report = checkpoint_report
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(
        cls,
        feat_root: str | Path,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda",
        protocol: FeatProtocol = FeatProtocol(),
    ) -> "FeatAdapter":
        root = Path(feat_root).resolve()
        ckpt = Path(checkpoint).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"FEAT source root not found: {root}")
        if not ckpt.is_file():
            raise FileNotFoundError(f"FEAT checkpoint not found: {ckpt}")
        with _prepend_sys_path(root):
            module = importlib.import_module("model.models.feat")
            feat_class = getattr(module, "FEAT", None)
            if feat_class is None:
                raise ImportError(f"{root}/model/models/feat.py does not expose FEAT")
        args = SimpleNamespace(
            backbone_class="Res12",
            way=int(protocol.way),
            shot=int(protocol.shot),
            query=int(protocol.query),
            eval_way=int(protocol.way),
            eval_shot=int(protocol.shot),
            eval_query=int(protocol.query),
            use_euclidean=bool(protocol.use_euclidean),
            temperature=float(protocol.temperature),
            temperature2=float(protocol.temperature2),
        )
        target_device = torch.device(device)
        model = feat_class(args).to(target_device)
        try:
            payload = torch.load(str(ckpt), map_location=target_device, weights_only=False)
        except TypeError:  # PyTorch < 2.0 compatibility
            payload = torch.load(str(ckpt), map_location=target_device)
        if not isinstance(payload, Mapping) or "params" not in payload:
            raise ValueError("FEAT checkpoint must be a mapping with a 'params' state_dict")
        state = payload["params"]
        model.load_state_dict(state, strict=True)
        report = FeatCheckpointReport(
            feat_root=str(root),
            checkpoint=str(ckpt),
            checkpoint_sha256=sha256_file(ckpt),
            payload_key="params",
            strict=True,
            parameter_count=int(sum(p.numel() for p in model.parameters())),
        )
        return cls(
            model,
            device=target_device,
            protocol=protocol,
            checkpoint_report=report,
        )

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> FeatEncoding:
        layer4 = getattr(getattr(self.model, "encoder", None), "layer4", None)
        if layer4 is None:
            raise AttributeError("FEAT model must expose model.encoder.layer4")
        box: MutableMapping[str, torch.Tensor] = {}

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            box["spatial"] = output.detach()

        handle = layer4.register_forward_hook(hook)
        try:
            embeddings = self.model.encoder(images.to(self.device, non_blocking=True))
        finally:
            handle.remove()
        if "spatial" not in box:
            raise RuntimeError("FEAT layer4 hook did not receive a spatial map")
        spatial = box["spatial"]
        if embeddings.ndim != 2 or spatial.ndim != 4 or len(embeddings) != len(spatial):
            raise RuntimeError(
                f"unexpected FEAT outputs: embeddings={tuple(embeddings.shape)}, spatial={tuple(spatial.shape)}"
            )
        return FeatEncoding(embeddings=embeddings, spatial_maps=spatial)

    def native_logits(
        self,
        embeddings: torch.Tensor,
        support_by_class: Mapping[int, Sequence[int]],
        query_indices: Sequence[int],
    ) -> torch.Tensor:
        prototypes: list[torch.Tensor] = []
        for class_id in range(int(self.protocol.way)):
            indices = tuple(int(i) for i in support_by_class[class_id])
            if not indices:
                raise ValueError(f"FEAT class {class_id} has no retained support")
            index = torch.as_tensor(indices, device=embeddings.device, dtype=torch.long)
            prototypes.append(embeddings[index].mean(dim=0))
        proto = torch.stack(prototypes, dim=0).unsqueeze(0)
        proto = self.model.slf_attn(proto, proto, proto).squeeze(0)
        query_index = torch.as_tensor(tuple(int(i) for i in query_indices), device=embeddings.device)
        queries = embeddings[query_index]
        if bool(self.protocol.use_euclidean):
            return -((queries.unsqueeze(1) - proto.unsqueeze(0)) ** 2).sum(dim=2) / float(
                self.protocol.temperature
            )
        proto = F.normalize(proto, dim=-1)
        queries = F.normalize(queries, dim=-1)
        return (queries @ proto.t()) / float(self.protocol.temperature)


def support_by_class(labels: Sequence[int], *, way: int = 5) -> dict[int, list[int]]:
    out = {class_id: [] for class_id in range(int(way))}
    for index, label in enumerate(labels):
        out[int(label)].append(int(index))
    return out


def normalize_feat_patches(spatial_maps: torch.Tensor) -> torch.Tensor:
    if spatial_maps.ndim != 4:
        raise ValueError(f"expected [N,C,H,W], got {tuple(spatial_maps.shape)}")
    return F.normalize(spatial_maps.flatten(2).transpose(1, 2).contiguous(), dim=-1)


def _feat_patch_score(
    query_patch: torch.Tensor,
    support_patches: torch.Tensor,
) -> tuple[float, float, int]:
    if query_patch.ndim != 2 or support_patches.ndim != 3:
        raise ValueError("query_patch must be [P,C] and support_patches [S,P,C]")
    if int(support_patches.shape[0]) == 0:
        return 0.0, 1.0, 0
    support_flat = support_patches.reshape(-1, support_patches.shape[-1])
    similarities = query_patch @ support_flat.t()
    query_max = similarities.max(dim=1).values
    support_max = similarities.max(dim=0).values
    bidirectional = float((0.5 * (query_max.mean() + support_max.mean())).detach().cpu())
    probs = torch.softmax(query_max, dim=0)
    entropy = float(
        (
            -(probs * torch.log(probs + 1e-12)).sum()
            / math.log(max(2, int(query_max.numel())))
        )
        .detach()
        .cpu()
    )
    per_support: list[float] = []
    for index in range(int(support_patches.shape[0])):
        sim = query_patch @ support_patches[index].t()
        per_support.append(float(sim.max(dim=1).values.mean().detach().cpu()))
    threshold = max(per_support) - 0.02
    agreement = int(sum(value >= threshold for value in per_support))
    return bidirectional, entropy, agreement


def feat_clean_feature_dict(
    query_patch: torch.Tensor,
    support_patches: torch.Tensor,
    support_labels: Sequence[int],
    *,
    a0_prediction: int,
    proposal: int,
    probabilities: np.ndarray,
    way: int = 5,
) -> dict[str, float | int]:
    labels = np.asarray(support_labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (int(way),):
        raise ValueError(f"probabilities must have shape ({way},)")
    class_scores: list[float] = []
    class_info: dict[int, tuple[float, float, int]] = {}
    for class_id in range(int(way)):
        indices = np.flatnonzero(labels == class_id)
        score, entropy, agreement = _feat_patch_score(query_patch, support_patches[indices])
        class_scores.append(score)
        class_info[class_id] = (score, entropy, agreement)
    a0 = int(a0_prediction)
    prop = int(proposal)
    proposal_score, proposal_entropy, proposal_agreement = class_info[prop]
    a0_score = class_info[a0][0]
    rank = int(1 + sum(score > proposal_score for score in class_scores))
    sorted_probs = np.sort(probs)[::-1]
    a0_prob = float(probs[a0])
    proposal_prob = float(probs[prop])
    return {
        "a0_prob": a0_prob,
        "a0_margin": float(sorted_probs[0] - sorted_probs[1]),
        "a0_entropy": float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(probs))),
        "proposal_prob": proposal_prob,
        "proposal_ratio": float(proposal_prob / (a0_prob + 1e-12)),
        "patch_score_proposal": proposal_score,
        "patch_score_a0": a0_score,
        "patch_margin_over_a0": proposal_score - a0_score,
        "proposal_patch_rank": rank,
        "proposal_equals_best_patch": int(prop == int(np.argmax(class_scores))),
        "proposal_support_agreement_count": proposal_agreement,
        "proposal_patch_entropy": proposal_entropy,
    }


def mapping_matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(features)), dtype=np.float32)
    missing = sorted({feature for feature in features if any(feature not in row for row in rows)})
    if missing:
        raise KeyError(f"missing FEAT features: {missing}")
    # Match the original pandas ``astype(float)`` training path.  Keeping the
    # calibration matrix in float64 preserves the selected threshold exactly.
    values = np.asarray([[row[feature] for feature in features] for row in rows], dtype=np.float64)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def positive_probability(model: Pipeline, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        return np.zeros(len(x), dtype=np.float64)
    return probabilities[:, int(positive[0])].astype(np.float64)


def fit_feat_clean_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: FeatCleanGatePolicy = FeatCleanGatePolicy(),
) -> FeatCleanGateBundle:
    x = mapping_matrix(rows, FEAT_CLEAN_FEATURES)
    y_help = np.asarray([int(row["help_label"]) for row in rows], dtype=np.int64)
    y_hurt = np.asarray([int(row["hurt_label"]) for row in rows], dtype=np.int64)
    if len(np.unique(y_help)) < 2 or len(np.unique(y_hurt)) < 2:
        raise ValueError("FEAT clean calibration needs both classes for help and hurt")
    help_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=int(policy.help_random_state),
        ),
    )
    hurt_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=int(policy.hurt_random_state),
        ),
    )
    help_model.fit(x, y_help)
    hurt_model.fit(x, y_hurt)
    return FeatCleanGateBundle(help_model, hurt_model, FEAT_CLEAN_FEATURES, policy)


def feat_clean_calibration_grid(
    bundle: FeatCleanGateBundle,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    p_help, p_hurt = bundle.probabilities(rows)
    help_label = np.asarray([int(row["help_label"]) for row in rows], dtype=np.int64)
    hurt_label = np.asarray([int(row["hurt_label"]) for row in rows], dtype=np.int64)
    a0_correct = np.asarray([int(row["a0_correct"]) for row in rows], dtype=np.int64)
    proposal = np.asarray([int(row["proposal_class"]) for row in rows], dtype=np.int64)
    truth = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
    safe_total = int(a0_correct.sum())
    rescue_total = int(((a0_correct == 0) & (proposal == truth)).sum())
    output: list[dict[str, Any]] = []
    for lambda_hurt in (0.5, 1.0, 1.5, 2.0, 3.0):
        score = p_help - float(lambda_hurt) * p_hurt
        for quantile in (0, 20, 40, 50, 60, 70, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99):
            threshold = float(np.quantile(score, quantile / 100.0))
            applied = score >= threshold
            if not applied.any():
                continue
            help_count = int((applied & (help_label == 1)).sum())
            hurt_count = int((applied & (hurt_label == 1)).sum())
            output.append(
                {
                    "lambda_hurt": float(lambda_hurt),
                    "quantile": int(quantile),
                    "threshold": threshold,
                    "apply_count": int(applied.sum()),
                    "help": help_count,
                    "hurt": hurt_count,
                    "net": help_count - hurt_count,
                    "precision": help_count / max(1, help_count + hurt_count),
                    "safe_damage": hurt_count / max(1, safe_total),
                    "rescue_hit": help_count / max(1, rescue_total),
                    "apply_rate": float(applied.mean()),
                }
            )
    return sorted(
        output,
        key=lambda row: (
            -int(row["net"]),
            -float(row["precision"]),
            -float(row["rescue_hit"]),
            float(row["safe_damage"]),
        ),
    )


def iter_subsets(items: Sequence[int], max_delete: int = 3) -> Iterable[tuple[int, ...]]:
    yield tuple()
    for size in range(1, int(max_delete) + 1):
        yield from itertools.combinations(tuple(int(i) for i in items), size)


def feat_clean_hard_delete(
    adapter: FeatAdapter,
    embeddings: torch.Tensor,
    support_labels: Sequence[int],
    *,
    query_index: int,
    a0_prediction: int,
    proposal: int,
    accepted: bool,
    max_delete: int = 3,
) -> FeatHardDeleteDecision:
    a0 = int(a0_prediction)
    prop = int(proposal)
    if not accepted:
        return FeatHardDeleteDecision(a0, False, tuple(), None)
    by_class = support_by_class(support_labels, way=adapter.protocol.way)
    a0_supports = tuple(by_class[a0])
    best_score = -float("inf")
    best_subset: tuple[int, ...] | None = None
    with torch.inference_mode():
        for subset in iter_subsets(a0_supports, max_delete=max_delete):
            edited = {class_id: list(indices) for class_id, indices in by_class.items()}
            removed = set(subset)
            edited[a0] = [index for index in edited[a0] if index not in removed]
            if not edited[a0]:
                continue
            logits = adapter.native_logits(embeddings, edited, [int(query_index)]).squeeze(0)
            prediction = int(logits.argmax().detach().cpu())
            score = float((logits[prop] - logits[a0]).detach().cpu())
            if prediction != prop:
                continue
            if best_subset is None or score > best_score or (
                abs(score - best_score) < 1e-12 and len(subset) < len(best_subset)
            ):
                best_score = score
                best_subset = tuple(subset)
    if best_subset is None:
        return FeatHardDeleteDecision(a0, False, tuple(), None)
    return FeatHardDeleteDecision(prop, True, best_subset, best_score)


def corrupt_feat_support_images(
    support_images: torch.Tensor,
    support_labels: Sequence[int],
    *,
    corrupt_per_class: int,
    rng: random.Random,
    way: int = 5,
) -> tuple[torch.Tensor, np.ndarray, list[dict[str, Any]]]:
    """FEAT image-copy protocol used by the reported noisy plug-in experiment.

    Targets and sources are sampled one class at a time with ``random.Random``.
    The observed target label is kept.
    """
    labels = np.asarray(support_labels, dtype=np.int64)
    noisy = support_images.clone()
    corrupt = np.zeros(len(labels), dtype=bool)
    metadata: list[dict[str, Any]] = []
    for class_id in range(int(way)):
        targets = rng.sample(np.flatnonzero(labels == class_id).astype(int).tolist(), int(corrupt_per_class))
        source_pool = np.flatnonzero(labels != class_id).astype(int).tolist()
        for target in targets:
            source = int(rng.choice(source_pool))
            noisy[target] = support_images[source]
            corrupt[target] = True
            metadata.append(
                {
                    "target_support_index": int(target),
                    "observed_label": int(class_id),
                    "source_support_index": source,
                    "source_label": int(labels[source]),
                    "corruption_mode": f"pvse_copy_replacement_per_class_{int(corrupt_per_class)}",
                }
            )
    return noisy, corrupt, metadata


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)


def feat_support_reliability_rows(
    support_embeddings: torch.Tensor | np.ndarray,
    support_maps: torch.Tensor,
    support_labels: Sequence[int],
    *,
    corrupt_mask: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
    way: int = 5,
) -> list[dict[str, Any]]:
    labels = np.asarray(support_labels, dtype=np.int64)
    embeddings = (
        support_embeddings.detach().float().cpu().numpy()
        if isinstance(support_embeddings, torch.Tensor)
        else np.asarray(support_embeddings, dtype=np.float32)
    )
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError("support embeddings and labels differ")
    if support_maps.ndim != 4 or len(support_maps) != len(labels):
        raise ValueError("support maps and labels differ")
    corrupt = np.zeros(len(labels), dtype=bool) if corrupt_mask is None else np.asarray(corrupt_mask, dtype=bool)
    if len(corrupt) != len(labels):
        raise ValueError("corrupt_mask and labels differ")
    normalized = _normalize_rows(embeddings)
    patches = normalize_feat_patches(support_maps.detach())
    prefix = dict(metadata or {})
    rows: list[dict[str, Any]] = []
    for index, observed_label in enumerate(labels.astype(int)):
        same = np.flatnonzero(labels == observed_label)
        same_loo = same[same != index]
        if len(same_loo) == 0:
            raise ValueError("FEAT support reliability requires at least two supports per class")
        own_centroid = normalized[same_loo].mean(axis=0)
        own_centroid /= np.linalg.norm(own_centroid) + 1e-12
        own_similarity = float(normalized[index] @ own_centroid)
        other_classes = [class_id for class_id in range(int(way)) if class_id != observed_label]
        other_similarities: list[float] = []
        for class_id in other_classes:
            class_indices = np.flatnonzero(labels == class_id)
            centroid = normalized[class_indices].mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            other_similarities.append(float(normalized[index] @ centroid))
        nearest_other = max(other_similarities)
        other_indices = np.flatnonzero(labels != observed_label)
        own_patch, own_entropy, own_agreement = _feat_patch_score(patches[index], patches[same_loo])
        other_patch_scores: list[tuple[int, float]] = []
        other_patch_entropies: list[float] = []
        for class_id in other_classes:
            class_indices = np.flatnonzero(labels == class_id)
            score, entropy, _agreement = _feat_patch_score(patches[index], patches[class_indices])
            other_patch_scores.append((class_id, score))
            other_patch_entropies.append(entropy)
        best_other_class, best_other_patch = max(other_patch_scores, key=lambda item: item[1])
        rows.append(
            {
                **prefix,
                "support_index": int(index),
                "observed_label": int(observed_label),
                "corrupted_label": int(corrupt[index]),
                "cos_own_centroid_loo": own_similarity,
                "cos_nearest_other_centroid": nearest_other,
                "gap_own_vs_nearest_other": own_similarity - nearest_other,
                "support_norm": float(np.linalg.norm(embeddings[index])),
                "nn_same_class_similarity": float((normalized[index] @ normalized[same_loo].T).max()),
                "nn_other_class_similarity": float((normalized[index] @ normalized[other_indices].T).max()),
                "own_class_compactness_loo": float(np.mean(normalized[same_loo] @ own_centroid)),
                "own_patch_bidir": own_patch,
                "nearest_other_patch_bidir": best_other_patch,
                "own_minus_other_patch_bidir": own_patch - best_other_patch,
                "own_patch_entropy": own_entropy,
                "nearest_other_patch_entropy": float(np.mean(other_patch_entropies)),
                "own_support_agreement_count": own_agreement,
                "best_patch_class_is_not_observed": int(best_other_patch > own_patch),
                "nearest_other_patch_class": int(best_other_class),
            }
        )
    return rows


def _support_binary_metrics(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    pred = np.asarray(predicted, dtype=bool)
    actual = np.asarray(truth, dtype=bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def fit_select_feat_reliability(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    *,
    feature_sets: Sequence[str] = ("global_plus_patch",),
) -> tuple[FeatReliabilityBundle, list[dict[str, Any]]]:
    if not train_rows or not val_rows:
        raise ValueError("FEAT reliability selection requires train and validation rows")
    grid: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str], Pipeline] = {}
    y_train = np.asarray([int(row["corrupted_label"]) for row in train_rows], dtype=np.int64)
    y_val = np.asarray([int(row["corrupted_label"]) for row in val_rows], dtype=np.int64)
    if len(np.unique(y_train)) < 2:
        raise ValueError("FEAT reliability training labels contain fewer than two classes")
    for feature_set in feature_sets:
        if feature_set not in FEAT_NOISY_FEATURE_SETS:
            raise ValueError(f"unknown FEAT feature set: {feature_set}")
        features = FEAT_NOISY_FEATURE_SETS[feature_set]
        x_train = mapping_matrix(train_rows, features)
        x_val = mapping_matrix(val_rows, features)
        for model_type, penalty in (("logreg", "l2"), ("l1_logreg", "l1")):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    penalty=penalty,
                    class_weight="balanced",
                    random_state=519,
                ),
            )
            model.fit(x_train, y_train)
            fitted[(feature_set, model_type)] = model
            probabilities = positive_probability(model, x_val)
            for quantile in FEAT_THRESHOLD_QUANTILES:
                threshold = float(np.quantile(probabilities, quantile / 100.0))
                predicted = probabilities >= threshold
                precision, recall, f1 = _support_binary_metrics(predicted, y_val)
                grid.append(
                    {
                        "feature_set": feature_set,
                        "model_type": model_type,
                        "threshold_quantile": int(quantile),
                        "threshold": threshold,
                        "support_precision": precision,
                        "support_recall": recall,
                        "support_F1": f1,
                        "predicted_corrupt_count": int(predicted.sum()),
                    }
                )
    grid_sorted = sorted(
        grid,
        key=lambda row: (
            -float(row["support_F1"]),
            -float(row["support_precision"]),
            -float(row["support_recall"]),
        ),
    )
    best = grid_sorted[0]
    selection = FeatReliabilitySelection(
        feature_set=str(best["feature_set"]),
        model_type=str(best["model_type"]),
        threshold_quantile=int(best["threshold_quantile"]),
        threshold=float(best["threshold"]),
        support_precision=float(best["support_precision"]),
        support_recall=float(best["support_recall"]),
        support_f1=float(best["support_F1"]),
        predicted_corrupt_count=int(best["predicted_corrupt_count"]),
    )
    return FeatReliabilityBundle(
        model=fitted[(selection.feature_set, selection.model_type)],
        features=FEAT_NOISY_FEATURE_SETS[selection.feature_set],
        selection=selection,
    ), grid_sorted


def feat_threshold_cap_delete(
    probabilities: np.ndarray,
    *,
    threshold: float,
    max_delete: int = 3,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    candidates = np.flatnonzero(values >= float(threshold))
    if len(candidates) > int(max_delete):
        order = np.argsort(-values[candidates], kind="mergesort")[: int(max_delete)]
        candidates = candidates[order]
    deleted = np.zeros(len(values), dtype=bool)
    deleted[candidates] = True
    return deleted


def retained_support_by_class(
    labels: Sequence[int],
    deleted: np.ndarray | None = None,
    *,
    way: int = 5,
) -> dict[int, list[int]]:
    labels_array = np.asarray(labels, dtype=np.int64)
    removed = np.zeros(len(labels_array), dtype=bool) if deleted is None else np.asarray(deleted, dtype=bool)
    out = {class_id: [] for class_id in range(int(way))}
    for index, label in enumerate(labels_array.astype(int)):
        if not removed[index]:
            out[label].append(index)
    for class_id in range(int(way)):
        if not out[class_id]:
            # k<=3 cannot empty a 5-shot class; keep the defensive fallback.
            out[class_id] = np.flatnonzero(labels_array == class_id).astype(int).tolist()
    return out


def save_feat_clean_gate(bundle: FeatCleanGateBundle, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "format": "pvse_feat_clean_gate_v1",
            "features": list(bundle.features),
            "policy": asdict(bundle.policy),
            "help_model": bundle.help_model,
            "hurt_model": bundle.hurt_model,
        },
        target,
    )


def load_feat_clean_gate(path: str | Path) -> FeatCleanGateBundle:
    payload = joblib.load(Path(path))
    if not isinstance(payload, Mapping) or payload.get("format") != "pvse_feat_clean_gate_v1":
        raise ValueError("unrecognized FEAT clean gate bundle")
    return FeatCleanGateBundle(
        help_model=payload["help_model"],
        hurt_model=payload["hurt_model"],
        features=tuple(payload["features"]),
        policy=FeatCleanGatePolicy(**payload["policy"]),
    )


def save_feat_reliability(bundle: FeatReliabilityBundle, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "format": "pvse_feat_reliability_v1",
            "features": list(bundle.features),
            "selection": asdict(bundle.selection),
            "model": bundle.model,
        },
        target,
    )


def load_feat_reliability(path: str | Path) -> FeatReliabilityBundle:
    payload = joblib.load(Path(path))
    if not isinstance(payload, Mapping) or payload.get("format") != "pvse_feat_reliability_v1":
        raise ValueError("unrecognized FEAT reliability bundle")
    return FeatReliabilityBundle(
        model=payload["model"],
        features=tuple(payload["features"]),
        selection=FeatReliabilitySelection(**payload["selection"]),
    )
