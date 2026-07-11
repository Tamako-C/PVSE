from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from pvse.models.resnet12 import RFSResNet12Backbone


@dataclass(frozen=True)
class ExtractedFeatures:
    global_features: np.ndarray
    spatial_features: torch.Tensor


def extract_features(
    backbone: RFSResNet12Backbone,
    images: torch.Tensor,
    *,
    device: str | torch.device,
    batch_size: int = 128,
    amp: bool = False,
    channels_last: bool = False,
) -> ExtractedFeatures:
    """Extract both global embeddings and layer-4 spatial maps."""
    if images.ndim != 4:
        raise ValueError(f"images must have shape [N,3,H,W], got {tuple(images.shape)}")
    target = torch.device(device)
    backbone = backbone.to(target).eval()
    global_batches: list[torch.Tensor] = []
    map_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(images), int(batch_size)):
            batch = images[start : start + int(batch_size)].to(target, non_blocking=True)
            if channels_last and target.type == "cuda":
                batch = batch.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=target.type,
                enabled=bool(amp) and target.type == "cuda",
            ):
                output = backbone.forward_with_spatial(batch)
            global_batches.append(output.global_features.float().cpu())
            map_batches.append(output.spatial_features.float().cpu())
    if not global_batches:
        raise ValueError("images is empty")
    return ExtractedFeatures(
        global_features=torch.cat(global_batches, dim=0).numpy().astype(np.float32),
        spatial_features=torch.cat(map_batches, dim=0),
    )
