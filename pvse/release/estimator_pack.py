from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare_output(path: str | Path, *, expected_name: str, force: bool) -> Path:
    output = Path(path).resolve()
    if output.name != expected_name:
        raise ValueError(f"output directory must be named {expected_name}")
    if output.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def pipeline_arrays(model: Any) -> dict[str, np.ndarray]:
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    return {
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "scaler_var": np.asarray(scaler.var_, dtype=np.float64),
        "coef": np.asarray(classifier.coef_, dtype=np.float64),
        "intercept": np.asarray(classifier.intercept_, dtype=np.float64),
        "classes": np.asarray(classifier.classes_, dtype=np.int64),
    }


def write_binary_estimator(
    directory: str | Path,
    *,
    estimator_id: str,
    logical_role: str,
    model: Any,
    features: Sequence[str],
    model_metadata: Mapping[str, Any],
    policy: Mapping[str, Any],
    calibration: Mapping[str, Any],
    paper_scope: Sequence[str],
    joblib_file: str,
    arrays_stem: str | None = None,
    file_stem: str | None = None,
) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / f"{arrays_stem or estimator_id}.npz"
    arrays = pipeline_arrays(model)
    np.savez_compressed(
        arrays_path,
        **{f"model_{name}": value for name, value in arrays.items()},
    )
    metadata_path = destination / f"{file_stem or estimator_id}.json"
    write_json(
        metadata_path,
        {
            "schema": "pvse.portable.binary_reliability",
            "schema_version": 1,
            "estimator_id": estimator_id,
            "logical_role": logical_role,
            "paper_scope": list(paper_scope),
            "arrays_file": arrays_path.name,
            "arrays_sha256": sha256_file(arrays_path),
            "features": [str(feature) for feature in features],
            "model": dict(model_metadata),
            "policy": dict(policy),
            "calibration": dict(calibration),
            "joblib_file": joblib_file,
        },
    )
    return metadata_path


def write_two_head_gate(
    directory: str | Path,
    *,
    estimator_id: str,
    logical_role: str,
    help_model: Any,
    hurt_model: Any,
    features: Sequence[str],
    model_metadata: Mapping[str, Any],
    policy: Mapping[str, Any],
    calibration: Mapping[str, Any],
    paper_scope: Sequence[str],
    joblib_file: str,
    file_stem: str | None = None,
) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / f"{file_stem or estimator_id}.npz"
    help_arrays = pipeline_arrays(help_model)
    hurt_arrays = pipeline_arrays(hurt_model)
    np.savez_compressed(
        arrays_path,
        **{f"help_{name}": value for name, value in help_arrays.items()},
        **{f"hurt_{name}": value for name, value in hurt_arrays.items()},
    )
    metadata_path = destination / f"{file_stem or estimator_id}.json"
    write_json(
        metadata_path,
        {
            "schema": "pvse.portable.two_head_gate",
            "schema_version": 1,
            "estimator_id": estimator_id,
            "logical_role": logical_role,
            "paper_scope": list(paper_scope),
            "arrays_file": arrays_path.name,
            "arrays_sha256": sha256_file(arrays_path),
            "features": [str(feature) for feature in features],
            "model": dict(model_metadata),
            "policy": dict(policy),
            "calibration": dict(calibration),
            "joblib_file": joblib_file,
        },
    )
    return metadata_path


def write_source_manifest(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_id",
        "role",
        "sha256",
        "rows_used",
        "used_for_fit",
        "used_for_model_selection",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_pack_manifests(
    root: str | Path,
    *,
    asset: str,
    logical_role: str,
) -> None:
    directory = Path(root)
    excluded = {"MANIFEST.json", "MANIFEST.csv", "SHA256SUMS.txt"}
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(
        directory / "MANIFEST.json",
        {
            "schema": "pvse.estimator_release_manifest",
            "schema_version": 1,
            "asset": asset,
            "logical_role": logical_role,
            "files": rows,
        },
    )
    with (directory / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = path.relative_to(directory).as_posix()
            if relative == "SHA256SUMS.txt":
                continue
            handle.write(f"{sha256_file(path)}  {relative}\n")


def write_deterministic_zip(root: str | Path, destination: str | Path) -> str:
    directory = Path(root)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.zip")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.comment = b""
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = path.relative_to(directory).as_posix()
            info = zipfile.ZipInfo(
                f"{directory.name}/{relative}",
                date_time=(2026, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(temporary, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("generated ZIP failed its integrity check")
        if archive.comment:
            raise RuntimeError("generated ZIP contains an unexpected comment")
    temporary.replace(output)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n",
        encoding="utf-8",
    )
    return digest
