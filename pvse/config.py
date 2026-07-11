from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import os
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


class ConfigError(ValueError):
    """Raised when a paper-run YAML configuration is invalid."""


T = TypeVar("T")


def _set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ConfigError("override key must not be empty")
    cursor = target
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"override path crosses a non-mapping key: {dotted_key}")
        cursor = child
    cursor[parts[-1]] = value


def _expand_strings(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_strings(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_strings(item) for key, item in value.items()}
    return value


def load_paper_config(
    path: str | Path,
    *,
    overrides: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Load a versioned YAML config and apply ``key=value`` overrides.

    Relative values under ``paths`` are resolved against the YAML file's
    directory. Environment variables are expanded before validation. This keeps
    checked-in configs portable while ensuring every run emits an explicit,
    effective configuration.
    """
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("paper config must contain a top-level mapping")
    data: dict[str, Any] = deepcopy(payload)
    for expression in overrides:
        if "=" not in expression:
            raise ConfigError(f"override must use key=value syntax: {expression!r}")
        key, raw = expression.split("=", 1)
        _set_nested(data, key.strip(), yaml.safe_load(raw))
    data = _expand_strings(data)
    if int(data.get("schema_version", -1)) != 1:
        raise ConfigError("unsupported or missing schema_version; expected 1")
    experiment = data.get("experiment")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ConfigError("experiment must be a non-empty string")
    paths = data.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ConfigError("paths must be a mapping")
    for key, value in list(paths.items()):
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise ConfigError(f"paths.{key} must be a string or null")
        if "${" in value:
            # Keep the unresolved marker for a precise required-path error.
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            paths[key] = str((config_path.parent / candidate).resolve())
        else:
            paths[key] = str(candidate.resolve())
    data["config_file"] = str(config_path)
    return data


def required_path(config: Mapping[str, Any], key: str) -> str:
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ConfigError("paths mapping is missing")
    value = paths.get(key)
    if not isinstance(value, str) or not value or "${" in value:
        raise ConfigError(
            f"paths.{key} is required; set it in YAML, through an environment variable, "
            f"or with --set paths.{key}=/absolute/path"
        )
    return value


def optional_path(config: Mapping[str, Any], key: str) -> str | None:
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        return None
    value = paths.get(key)
    if value in (None, ""):
        return None
    if not isinstance(value, str) or "${" in value:
        raise ConfigError(f"paths.{key} contains an unresolved value: {value!r}")
    return value


def dataclass_kwargs(cls: type[T], values: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    """Validate a mapping against dataclass fields before construction."""
    names = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ConfigError(f"unknown {context} keys: {unknown}")
    return dict(values)
