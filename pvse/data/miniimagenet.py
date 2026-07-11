from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from PIL import Image

from pvse.data.transforms import build_eval_transform

MiniImageNetSplit = Literal["train64", "val", "test"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def split_root(data_root: str | Path, split: MiniImageNetSplit) -> Path:
    root = Path(data_root)
    relative = {
        "train64": Path("train") / "train",
        "val": Path("val") / "val",
        "test": Path("test") / "test",
    }[split]
    path = root / relative
    if not path.is_dir():
        raise FileNotFoundError(
            f"miniImageNet split directory not found: {path}; expected train/train, val/val, test/test"
        )
    return path


def _image_paths(class_dir: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(class_dir.iterdir(), key=lambda p: p.name)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


@dataclass(frozen=True)
class EpisodeManifest:
    episode: int
    split: str
    class_ids: tuple[int, ...]
    class_names: tuple[str, ...]
    support_paths: tuple[str, ...]
    support_labels: tuple[int, ...]
    query_paths: tuple[str, ...]
    query_labels: tuple[int, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in (
            "class_ids",
            "class_names",
            "support_paths",
            "support_labels",
            "query_paths",
            "query_labels",
        ):
            data[key] = list(data[key])
        return data


@dataclass(frozen=True)
class LoadedEpisode:
    support_images: torch.Tensor
    support_labels: np.ndarray
    query_images: torch.Tensor
    query_labels: np.ndarray
    manifest: EpisodeManifest


class MiniImageNetEpisodeSampler:
    """Deterministic sampler matching the final miniImageNet scripts.

    Class directories and images are sorted lexicographically. Sampling uses a
    dedicated ``RandomState`` so that the sequence matches the paper protocol's
    ``np.random.seed`` + ``np.random.choice`` behavior without mutating global RNG.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: MiniImageNetSplit,
        *,
        seed: int,
        image_size: int = 84,
    ) -> None:
        self.root = split_root(data_root, split)
        self.split = split
        self.classes = tuple(sorted((p.name for p in self.root.iterdir() if p.is_dir())))
        self.class_paths = tuple(self.root / name for name in self.classes)
        self.images_by_class = tuple(_image_paths(path) for path in self.class_paths)
        if not self.classes:
            raise RuntimeError(f"no miniImageNet classes found under {self.root}")
        empty = [name for name, images in zip(self.classes, self.images_by_class) if not images]
        if empty:
            raise RuntimeError(f"classes without images: {empty[:5]}")
        self.rng = np.random.RandomState(int(seed))
        self.transform = build_eval_transform(int(image_size))
        self._episode = 0

    def sample_manifest(self, *, way: int = 5, shot: int = 5, query: int = 15) -> EpisodeManifest:
        if int(way) > len(self.classes):
            raise ValueError("way exceeds number of classes")
        class_ids = self.rng.choice(len(self.classes), int(way), replace=False).astype(int)
        support_paths: list[str] = []
        support_labels: list[int] = []
        query_paths: list[str] = []
        query_labels: list[int] = []
        for mapped_label, class_id in enumerate(class_ids.tolist()):
            pool = self.images_by_class[class_id]
            if len(pool) < int(shot) + int(query):
                raise ValueError(
                    f"class {self.classes[class_id]} has {len(pool)} images; "
                    f"requires {int(shot) + int(query)}"
                )
            chosen = self.rng.choice(len(pool), int(shot) + int(query), replace=False).astype(int)
            selected = [pool[i] for i in chosen.tolist()]
            support_paths.extend(str(p) for p in selected[: int(shot)])
            query_paths.extend(str(p) for p in selected[int(shot) :])
            support_labels.extend([mapped_label] * int(shot))
            query_labels.extend([mapped_label] * int(query))
        manifest = EpisodeManifest(
            episode=int(self._episode),
            split=str(self.split),
            class_ids=tuple(int(i) for i in class_ids),
            class_names=tuple(self.classes[int(i)] for i in class_ids),
            support_paths=tuple(support_paths),
            support_labels=tuple(support_labels),
            query_paths=tuple(query_paths),
            query_labels=tuple(query_labels),
        )
        self._episode += 1
        return manifest

    def load_manifest(self, manifest: EpisodeManifest) -> LoadedEpisode:
        def load(paths: Iterable[str]) -> torch.Tensor:
            tensors: list[torch.Tensor] = []
            for path in paths:
                with Image.open(path) as image:
                    tensors.append(self.transform(image.convert("RGB")))
            if not tensors:
                raise ValueError("empty image list")
            return torch.stack(tensors, dim=0)

        return LoadedEpisode(
            support_images=load(manifest.support_paths),
            support_labels=np.asarray(manifest.support_labels, dtype=np.int64),
            query_images=load(manifest.query_paths),
            query_labels=np.asarray(manifest.query_labels, dtype=np.int64),
            manifest=manifest,
        )

    def sample(self, *, way: int = 5, shot: int = 5, query: int = 15) -> LoadedEpisode:
        return self.load_manifest(self.sample_manifest(way=way, shot=shot, query=query))
