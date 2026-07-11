from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from pvse.data.miniimagenet import LoadedEpisode
from pvse.data.transforms import build_eval_transform
from pvse.models.extract import extract_features
from pvse.models.resnet12 import RFSResNet12Backbone


@dataclass(frozen=True)
class EpisodeFeatures:
    episode: int
    split: str
    support_features: np.ndarray
    support_maps: torch.Tensor
    support_labels: np.ndarray
    query_features: np.ndarray
    query_maps: torch.Tensor
    query_labels: np.ndarray
    manifest: dict


def extract_loaded_episode(
    loaded: LoadedEpisode,
    backbone: RFSResNet12Backbone,
    *,
    device: str | torch.device,
    batch_size: int = 96,
    amp: bool = False,
    channels_last: bool = False,
) -> EpisodeFeatures:
    n_support = len(loaded.support_labels)
    images = torch.cat([loaded.support_images, loaded.query_images], dim=0)
    extracted = extract_features(
        backbone,
        images,
        device=device,
        batch_size=int(batch_size),
        amp=bool(amp),
        channels_last=bool(channels_last),
    )
    return EpisodeFeatures(
        episode=int(loaded.manifest.episode),
        split=str(loaded.manifest.split),
        support_features=extracted.global_features[:n_support].astype(np.float32, copy=False),
        support_maps=extracted.spatial_features[:n_support].float(),
        support_labels=np.asarray(loaded.support_labels, dtype=np.int64),
        query_features=extracted.global_features[n_support:].astype(np.float32, copy=False),
        query_maps=extracted.spatial_features[n_support:].float(),
        query_labels=np.asarray(loaded.query_labels, dtype=np.int64),
        manifest=loaded.manifest.to_dict(),
    )


def load_image_paths(paths: Sequence[str | Path], *, image_size: int = 84) -> torch.Tensor:
    transform = build_eval_transform(int(image_size))
    tensors = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
    if not tensors:
        raise ValueError("no image paths supplied")
    return torch.stack(tensors, dim=0)


def extract_path_episode(
    support_paths: Sequence[str | Path],
    support_labels: np.ndarray,
    query_paths: Sequence[str | Path],
    query_labels: np.ndarray,
    backbone: RFSResNet12Backbone,
    *,
    episode: int,
    split: str,
    manifest: dict,
    device: str | torch.device,
    image_size: int = 84,
    batch_size: int = 96,
    amp: bool = False,
    channels_last: bool = False,
) -> EpisodeFeatures:
    s_labels = np.asarray(support_labels, dtype=np.int64)
    q_labels = np.asarray(query_labels, dtype=np.int64)
    images = torch.cat(
        [
            load_image_paths(support_paths, image_size=int(image_size)),
            load_image_paths(query_paths, image_size=int(image_size)),
        ],
        dim=0,
    )
    extracted = extract_features(
        backbone,
        images,
        device=device,
        batch_size=int(batch_size),
        amp=bool(amp),
        channels_last=bool(channels_last),
    )
    n_support = len(s_labels)
    if n_support != len(support_paths) or len(q_labels) != len(query_paths):
        raise ValueError("path and label counts differ")
    return EpisodeFeatures(
        episode=int(episode),
        split=str(split),
        support_features=extracted.global_features[:n_support].astype(np.float32, copy=False),
        support_maps=extracted.spatial_features[:n_support].float(),
        support_labels=s_labels,
        query_features=extracted.global_features[n_support:].astype(np.float32, copy=False),
        query_maps=extracted.spatial_features[n_support:].float(),
        query_labels=q_labels,
        manifest=dict(manifest),
    )
