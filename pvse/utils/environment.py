from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any

import torch


_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "Pillow",
    "PyYAML",
    "torch",
    "torchvision",
)


def environment_report() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
