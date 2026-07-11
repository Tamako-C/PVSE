from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CheckpointLoadReport:
    checkpoint: str
    sha256: str
    payload_key: str
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatch_keys: tuple[str, ...]
    target_key_count: int
    loaded_key_count: int
    loaded_parameter_fraction: float

    @property
    def complete(self) -> bool:
        return not self.missing_keys and not self.shape_mismatch_keys

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _state_from_payload(payload: Any) -> tuple[Mapping[str, torch.Tensor], str]:
    if not isinstance(payload, Mapping):
        if not hasattr(payload, "keys"):
            raise TypeError("checkpoint payload is not a state dictionary")
        return payload, "raw"
    for key in ("backbone", "params", "state_dict", "model_state", "model"):
        value = payload.get(key)
        if isinstance(value, Mapping) and value:
            if key in {"model_state", "model"} and any(str(k).startswith("backbone.") for k in value):
                return {
                    str(k).replace("backbone.", "", 1): v
                    for k, v in value.items()
                    if str(k).startswith("backbone.")
                }, key
            return value, key
    if payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload, "raw"
    raise ValueError("could not locate a backbone state dictionary in checkpoint")


def _candidate_mappings(state: Mapping[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    raw = {str(k): v for k, v in state.items() if torch.is_tensor(v)}
    candidates: list[dict[str, torch.Tensor]] = [raw]
    if any(k.startswith("module.") for k in raw):
        candidates.append({k.replace("module.", "", 1): v for k, v in raw.items()})
    expanded = list(candidates)
    for candidate in expanded:
        if any(k.startswith("backbone.") for k in candidate):
            candidates.append(
                {k.replace("backbone.", "", 1): v for k, v in candidate.items() if k.startswith("backbone.")}
            )
        if any(k.startswith("encoder.") for k in candidate):
            mapped: dict[str, torch.Tensor] = {}
            for key, tensor in candidate.items():
                new_key = key.replace("encoder.", "", 1)
                for layer in ("layer1", "layer2", "layer3", "layer4"):
                    new_key = new_key.replace(f"{layer}.0.", f"{layer}.")
                new_key = new_key.replace("downsample.", "shortcut.")
                mapped[new_key] = tensor
            candidates.append(mapped)
    return candidates


def load_backbone_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    require_complete: bool = True,
    expected_sha256: str | None = None,
) -> CheckpointLoadReport:
    """Load a backbone checkpoint with an auditable compatibility report.

    The loader never silently treats a partial ``strict=False`` load as success.
    ``require_complete=True`` is the recommended paper-reproduction setting.
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = file_sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, observed {digest}"
        )
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch<2.0 compatibility
        payload = torch.load(path, map_location=map_location)
    state, payload_key = _state_from_payload(payload)
    target = model.state_dict()

    best: dict[str, torch.Tensor] = {}
    best_shape_mismatch: tuple[str, ...] = ()
    best_unexpected: tuple[str, ...] = ()
    for candidate in _candidate_mappings(state):
        compatible = {
            k: v for k, v in candidate.items() if k in target and tuple(v.shape) == tuple(target[k].shape)
        }
        shape_mismatch = tuple(
            sorted(k for k, v in candidate.items() if k in target and tuple(v.shape) != tuple(target[k].shape))
        )
        unexpected = tuple(sorted(k for k in candidate if k not in target))
        if len(compatible) > len(best):
            best = compatible
            best_shape_mismatch = shape_mismatch
            best_unexpected = unexpected
    if not best:
        raise RuntimeError(f"checkpoint has no compatible backbone tensors: {path}")

    missing = tuple(sorted(k for k in target if k not in best))
    model.load_state_dict(best, strict=False)
    target_parameter_count = sum(int(v.numel()) for v in target.values())
    loaded_parameter_count = sum(int(v.numel()) for v in best.values())
    report = CheckpointLoadReport(
        checkpoint=str(path),
        sha256=digest,
        payload_key=payload_key,
        loaded_keys=tuple(sorted(best)),
        missing_keys=missing,
        unexpected_keys=best_unexpected,
        shape_mismatch_keys=best_shape_mismatch,
        target_key_count=len(target),
        loaded_key_count=len(best),
        loaded_parameter_fraction=(loaded_parameter_count / max(1, target_parameter_count)),
    )
    if require_complete and not report.complete:
        raise RuntimeError(
            "incomplete backbone checkpoint load: "
            f"missing={len(report.missing_keys)}, shape_mismatch={len(report.shape_mismatch_keys)}"
        )
    return report
