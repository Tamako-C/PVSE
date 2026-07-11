from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from pvse.models.checkpoint import CheckpointLoadReport, load_backbone_checkpoint
from pvse.models.resnet12 import RFSResNet12Backbone


def resolve_device(requested: str) -> torch.device:
    value = str(requested)
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(value)


def load_paper_backbone(
    checkpoint: str | Path,
    *,
    device: str,
    expected_sha256: str = "",
) -> tuple[RFSResNet12Backbone, CheckpointLoadReport]:
    target = resolve_device(device)
    model = RFSResNet12Backbone(drop_rate=0.1, dropblock_size=5)
    report = load_backbone_checkpoint(
        model,
        checkpoint,
        map_location="cpu",
        require_complete=True,
        expected_sha256=expected_sha256 or None,
    )
    model.eval().to(target)
    return model, report


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
