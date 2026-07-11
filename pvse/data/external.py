from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}
MAIN_EXTERNAL_DATASETS = (
    "CUB",
    "Caltech101",
    "DTD",
    "FGVC_Aircraft",
    "OfficeHome",
    "PACS",
)


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _find_dir(root: Path, names: Iterable[str]) -> Path | None:
    wanted = {str(name).lower() for name in names}
    if root.is_dir() and root.name.lower() in wanted:
        return root
    return next((p for p in root.rglob("*") if p.is_dir() and p.name.lower() in wanted), None)


def _class_rows(dataset: str, domain: str, folder: Path, *, ignore: set[str] | None = None) -> list[dict]:
    ignored = ignore or set()
    class_dirs = sorted(
        (p for p in folder.iterdir() if p.is_dir() and p.name not in ignored),
        key=lambda p: p.name.lower(),
    )
    rows: list[dict] = []
    for class_id, class_dir in enumerate(class_dirs):
        images = sorted((p for p in class_dir.rglob("*") if _is_image(p)), key=lambda p: str(p).lower())
        for image in images:
            rows.append(
                {
                    "dataset": dataset,
                    "domain": domain,
                    "class_name": class_dir.name,
                    "class_id": int(class_id),
                    "image_path": str(image),
                    "original_split_if_any": "",
                }
            )
    return rows


def parse_pacs(root: str | Path) -> tuple[list[dict], dict]:
    root = Path(root)
    base = None
    for parts in (
        ("PACS_Dataset", "pacs_data", "pacs_data"),
        ("pacs_data", "pacs_data"),
        ("PACS_Dataset", "dct2_images", "dct2_images"),
        ("dct2_images", "dct2_images"),
    ):
        candidate = root.joinpath(*parts)
        if candidate.exists():
            base = candidate
            break
    base = base or _find_dir(root, ["pacs_data", "dct2_images"])
    if base is None:
        return [], {"status": "root_not_found"}
    rows: list[dict] = []
    for domain in ("art_painting", "cartoon", "photo", "sketch"):
        path = base / domain
        if path.is_dir():
            rows.extend(_class_rows("PACS", domain, path))
    return rows, {"detected_root": str(base)}


def parse_officehome(root: str | Path) -> tuple[list[dict], dict]:
    root = Path(root)
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        children = {child.name.lower().replace(" ", "_") for child in path.iterdir() if child.is_dir()}
        if {"art", "clipart", "product"}.issubset(children) and (
            "real_world" in children or "realworld" in children
        ):
            candidates.append(path)
    base = sorted(candidates, key=lambda p: len(str(p)))[0] if candidates else root
    variants = {
        "Art": ("Art", "art"),
        "Clipart": ("Clipart", "clipart"),
        "Product": ("Product", "product"),
        "RealWorld": ("Real World", "Real_World", "real_world", "realworld", "RealWorld"),
    }
    rows: list[dict] = []
    detected: dict[str, str] = {}
    for domain, names in variants.items():
        directory = next((base / name for name in names if (base / name).is_dir()), None)
        if directory is not None:
            detected[domain] = str(directory)
            rows.extend(_class_rows("OfficeHome", domain, directory))
    return rows, {"detected_root": str(base), "detected_domains": detected}


def parse_cub(root: str | Path) -> tuple[list[dict], dict]:
    base = _find_dir(Path(root), ["CUB_200_2011"])
    if base is None:
        return [], {"status": "root_not_found"}
    image_root = base / "images"
    if not image_root.is_dir():
        return [], {"status": "images_not_found", "detected_root": str(base)}
    return _class_rows("CUB", "all", image_root), {
        "detected_root": str(base),
        "label_source": "folder_names",
    }


def parse_dtd(root: str | Path) -> tuple[list[dict], dict]:
    image_root = next(
        (p for p in Path(root).rglob("images") if p.is_dir() and any(c.is_dir() for c in p.iterdir())),
        None,
    )
    if image_root is None:
        return [], {"status": "images_not_found"}
    return _class_rows("DTD", "all", image_root), {"detected_root": str(image_root)}


def parse_aircraft(root: str | Path) -> tuple[list[dict], dict]:
    base = _find_dir(Path(root), ["fgvc-aircraft-2013b"])
    if base is None:
        return [], {"status": "root_not_found"}
    data = base / "data"
    image_root = data / "images"
    if not image_root.is_dir():
        return [], {"status": "images_not_found", "detected_root": str(base)}
    class_to_id: dict[str, int] = {}
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        label_file = data / f"images_variant_{split}.txt"
        if not label_file.is_file():
            continue
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_id, class_name = parts[0], " ".join(parts[1:])
            class_to_id.setdefault(class_name, len(class_to_id))
            image = image_root / f"{image_id}.jpg"
            if image.is_file():
                rows.append(
                    {
                        "dataset": "FGVC_Aircraft",
                        "domain": "all",
                        "class_name": class_name,
                        "class_id": int(class_to_id[class_name]),
                        "image_path": str(image),
                        "original_split_if_any": split,
                    }
                )
    return rows, {
        "detected_root": str(base),
        "label_source": "images_variant_split_files",
    }


def parse_caltech101(root: str | Path) -> tuple[list[dict], dict]:
    base = _find_dir(Path(root), ["101_ObjectCategories"])
    if base is None:
        return [], {"status": "root_not_found"}
    ignored = {"BACKGROUND_Google", "background", "BACKGROUND", "Faces_easy"}
    return _class_rows("Caltech101", "all", base, ignore=ignored), {
        "detected_root": str(base),
        "ignored": sorted(ignored),
    }


PARSERS = {
    "PACS": parse_pacs,
    "OfficeHome": parse_officehome,
    "CUB": parse_cub,
    "DTD": parse_dtd,
    "FGVC_Aircraft": parse_aircraft,
    "Caltech101": parse_caltech101,
}


def build_external_index(dataset_roots: Mapping[str, str | Path]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    notes: dict[str, dict] = {}
    for dataset in MAIN_EXTERNAL_DATASETS:
        if dataset not in dataset_roots:
            notes[dataset] = {"status": "root_not_supplied"}
            continue
        parsed, note = PARSERS[dataset](dataset_roots[dataset])
        notes[dataset] = note
        rows.extend(parsed)
    index = pd.DataFrame(
        rows,
        columns=(
            "dataset",
            "domain",
            "class_name",
            "class_id",
            "image_path",
            "original_split_if_any",
        ),
    )
    return index, notes


def eligible_class_map(
    index: pd.DataFrame,
    dataset: str,
    domain: str,
    *,
    min_images: int = 20,
) -> dict[str, list[str]]:
    subset = index[
        (index["dataset"].astype(str) == str(dataset))
        & (index["domain"].astype(str) == str(domain))
    ]
    out: dict[str, list[str]] = {}
    for class_name, group in subset.groupby("class_name"):
        paths = group["image_path"].astype(str).tolist()
        if len(paths) >= int(min_images):
            out[str(class_name)] = paths
    return out


def common_class_maps(
    index: pd.DataFrame,
    dataset: str,
    support_domain: str,
    query_domain: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    support = eligible_class_map(index, dataset, support_domain)
    query = eligible_class_map(index, dataset, query_domain)
    common = sorted(set(support) & set(query))
    return ({name: support[name] for name in common}, {name: query[name] for name in common})


@dataclass(frozen=True)
class ExternalSetting:
    dataset: str
    domain_setting: str
    support_domain: str
    query_domain: str
    episodes: int


PAPER_EXTERNAL_SETTINGS: tuple[ExternalSetting, ...] = (
    ExternalSetting("CUB", "same_domain", "all", "all", 300),
    ExternalSetting("Caltech101", "same_domain", "all", "all", 300),
    ExternalSetting("DTD", "same_domain", "all", "all", 300),
    ExternalSetting("FGVC_Aircraft", "same_domain", "all", "all", 300),
    ExternalSetting("OfficeHome", "same_domain", "Art", "Art", 300),
    ExternalSetting("OfficeHome", "same_domain", "Clipart", "Clipart", 300),
    ExternalSetting("OfficeHome", "same_domain", "Product", "Product", 300),
    ExternalSetting("OfficeHome", "same_domain", "RealWorld", "RealWorld", 300),
    ExternalSetting("PACS", "same_domain", "art_painting", "art_painting", 300),
    ExternalSetting("PACS", "same_domain", "cartoon", "cartoon", 300),
    ExternalSetting("PACS", "same_domain", "photo", "photo", 300),
    ExternalSetting("PACS", "same_domain", "sketch", "sketch", 300),
    ExternalSetting("PACS", "cross_domain", "photo", "art_painting", 100),
    ExternalSetting("PACS", "cross_domain", "photo", "cartoon", 100),
    ExternalSetting("PACS", "cross_domain", "photo", "sketch", 100),
    ExternalSetting("PACS", "cross_domain", "art_painting", "sketch", 100),
    ExternalSetting("PACS", "cross_domain", "cartoon", "sketch", 100),
    ExternalSetting("OfficeHome", "cross_domain", "RealWorld", "Art", 100),
    ExternalSetting("OfficeHome", "cross_domain", "RealWorld", "Clipart", 100),
    ExternalSetting("OfficeHome", "cross_domain", "Product", "Clipart", 100),
    ExternalSetting("OfficeHome", "cross_domain", "Art", "RealWorld", 100),
    ExternalSetting("OfficeHome", "cross_domain", "Clipart", "RealWorld", 100),
)


@dataclass(frozen=True)
class ExternalEpisodeManifest:
    setting: ExternalSetting
    episode: int
    class_names: tuple[str, ...]
    support_paths: tuple[str, ...]
    support_labels: tuple[int, ...]
    query_paths: tuple[str, ...]
    query_labels: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "setting": asdict(self.setting),
            "episode": int(self.episode),
            "class_names": list(self.class_names),
            "support_paths": list(self.support_paths),
            "support_labels": list(self.support_labels),
            "query_paths": list(self.query_paths),
            "query_labels": list(self.query_labels),
        }


def sample_external_episode(
    support_class_map: Mapping[str, list[str]],
    query_class_map: Mapping[str, list[str]],
    *,
    rng: np.random.Generator,
    setting: ExternalSetting,
    episode: int,
    way: int = 5,
    shot: int = 5,
    query: int = 15,
) -> ExternalEpisodeManifest:
    classes = sorted(set(support_class_map) & set(query_class_map))
    if len(classes) < int(way):
        raise ValueError(f"only {len(classes)} common eligible classes, need {way}")
    chosen = rng.choice(classes, size=int(way), replace=False).tolist()
    support_paths: list[str] = []
    query_paths: list[str] = []
    support_labels: list[int] = []
    query_labels: list[int] = []
    for label, class_name in enumerate(chosen):
        support_pool = support_class_map[class_name]
        query_pool = query_class_map[class_name]
        if support_pool is query_pool or support_pool == query_pool:
            selected = rng.choice(
                support_pool, size=int(shot) + int(query), replace=False
            ).tolist()
            support_selected = selected[: int(shot)]
            query_selected = selected[int(shot) :]
        else:
            support_selected = rng.choice(support_pool, size=int(shot), replace=False).tolist()
            query_selected = rng.choice(query_pool, size=int(query), replace=False).tolist()
        support_paths.extend(str(path) for path in support_selected)
        query_paths.extend(str(path) for path in query_selected)
        support_labels.extend([label] * int(shot))
        query_labels.extend([label] * int(query))
    return ExternalEpisodeManifest(
        setting=setting,
        episode=int(episode),
        class_names=tuple(str(name) for name in chosen),
        support_paths=tuple(support_paths),
        support_labels=tuple(support_labels),
        query_paths=tuple(query_paths),
        query_labels=tuple(query_labels),
    )
