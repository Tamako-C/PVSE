from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def stable_text_code(text: str) -> int:
    """Deterministic code used to namespace corruption RNG streams."""
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(str(text))))


def stable_md5_seed(*parts: object) -> int:
    """Deterministic seed function used by the external-transfer protocol."""
    digest = hashlib.md5("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def set_process_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
