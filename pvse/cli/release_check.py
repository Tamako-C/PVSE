from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable

import yaml

from pvse import __version__
from pvse.artifacts.validate import validate_submitted_results
from pvse.config import load_paper_config


def _pyproject_version(path: Path) -> str:
    """Read project.version on Python 3.10+ without adding a TOML dependency."""
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        import re as _re

        text = path.read_text(encoding="utf-8")
        project_match = _re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
        if not project_match:
            raise AssertionError("[project] table missing from pyproject.toml")
        version_match = _re.search(
            r"(?m)^\s*version\s*=\s*[\"']([^\"']+)[\"']\s*$",
            project_match.group(1),
        )
        if not version_match:
            raise AssertionError("project.version missing from pyproject.toml")
        return version_match.group(1)
    with path.open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


@dataclass(frozen=True)
class ReleaseCheck:
    name: str
    passed: bool
    severity: str
    detail: str


_REQUIRED_FILES = (
    "README.md",
    "pyproject.toml",
    ".gitattributes",
    "THIRD_PARTY_NOTICES.md",
    "LICENSE",
    "docs/EXPERIMENTS.md",
    "docs/REPRODUCTION_STATUS.md",
    "docs/DATA_AND_CHECKPOINTS.md",
    "docs/PROVENANCE.md",
    "docs/RESULT_TRACEABILITY.md",
    "docs/FEAT_ADAPTER.md",
    "docs/REAL_DATA_SMOKE.md",
    "provenance/provenance_manifest.json",
    "provenance/paper_experiments.csv",
    "provenance/real_data_smoke_certificate.json",
)
_FORBIDDEN_BINARY_SUFFIXES = {".pth", ".pt", ".ckpt", ".pkl", ".joblib"}
_PRIVATE_PATH_PATTERNS = (
    re.compile("/" + "mnt/" + "data/"),
    re.compile("/" + "home/" + r"(?!USER|user|name|path|your)"),
    re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit repository integrity and report unresolved public-release blockers. "
            "The default exit code reflects structural integrity; use --require-public-ready "
            "to make license/asset blockers fatal."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--require-public-ready", action="store_true")
    return parser


def _attempt(name: str, severity: str, fn: Callable[[], str]) -> ReleaseCheck:
    try:
        return ReleaseCheck(name=name, passed=True, severity=severity, detail=str(fn()))
    except Exception as exc:
        return ReleaseCheck(
            name=name,
            passed=False,
            severity=severity,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _required_files(root: Path) -> str:
    missing = [rel for rel in _REQUIRED_FILES if not (root / rel).is_file()]
    if missing:
        raise AssertionError(f"missing required files: {missing}")
    license_markers = [root / "LICENSE", root / "LICENSE.txt", root / "LICENSE.md", root / "LICENSE_PENDING.md"]
    if not any(path.is_file() for path in license_markers):
        raise AssertionError("neither a software license nor LICENSE_PENDING.md is present")
    return f"{len(_REQUIRED_FILES)} required files plus license-status marker present"


def _version_consistency(root: Path) -> str:
    version = _pyproject_version(root / "pyproject.toml")
    if version != __version__:
        raise AssertionError(f"pyproject={version!r}, pvse.__version__={__version__!r}")
    return version


def _submitted_artifacts(root: Path) -> str:
    _, summary = validate_submitted_results(root / "artifacts/submitted/supporting_data")
    if not summary["passed"]:
        failed = [row["name"] for row in summary["checks"] if not row["passed"]]
        raise AssertionError(f"failed submitted checks: {failed}")
    return f"{summary['passed_count']}/{summary['check_count']} arithmetic/integrity checks"


def _gitattributes(root: Path) -> str:
    text = (root / ".gitattributes").read_text(encoding="utf-8")
    if "artifacts/submitted/** -text" not in text:
        raise AssertionError("submitted artifact tree is not protected from EOL normalization")
    return "submitted artifact tree marked -text"


def _paper_configs(root: Path) -> str:
    configs = sorted((root / "configs/paper").glob("*.yaml"))
    if len(configs) != 14:
        raise AssertionError(f"expected 14 paper configs, found {len(configs)}")
    experiments: list[str] = []
    for path in configs:
        config = load_paper_config(path)
        experiments.append(str(config["experiment"]))
    return f"{len(configs)} configs parsed; experiments={sorted(set(experiments))}"


def _provenance(root: Path) -> str:
    path = root / "provenance/provenance_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 2:
        raise AssertionError("unsupported paper-to-code manifest schema")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != 10:
        raise AssertionError("expected 10 paper-item provenance records")
    missing: list[str] = []
    for item in items:
        for key in ("configs", "modules", "reference_artifacts"):
            for rel in item.get(key, []):
                if not (root / rel).exists():
                    missing.append(str(rel))
    if missing:
        raise AssertionError(f"missing provenance targets: {sorted(set(missing))}")
    return "10 paper-item records; all checked-in targets exist"


def _forbidden_files(root: Path) -> str:
    bad: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in {".git", ".venv", "build", "dist", "outputs", "__pycache__", ".pytest_cache"} for part in rel.parts):
            continue
        if path.suffix.lower() in _FORBIDDEN_BINARY_SUFFIXES:
            bad.append(str(rel))
    if bad:
        raise AssertionError(f"model binary files are tracked in the source repository: {sorted(set(bad))}")
    return "source repository contains no model binary files"


def _text_scan(root: Path) -> str:
    findings: list[str] = []
    roots = [root / "pvse", root / "configs", root / "scripts", root / "provenance"]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".csv", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in (*_PRIVATE_PATH_PATTERNS, *_SECRET_PATTERNS):
                if pattern.search(text):
                    findings.append(f"{path.relative_to(root)}:{pattern.pattern}")
    if findings:
        raise AssertionError(f"private-path/secret patterns found: {findings[:10]}")
    return "no known private-path or credential patterns in source/config/provenance"


def _license_ready(root: Path) -> tuple[bool, str]:
    path = root / "LICENSE"
    if not path.is_file() or (root / "LICENSE_PENDING.md").exists():
        return False, "standard license file missing or LICENSE_PENDING.md remains"
    text = path.read_text(encoding="utf-8")
    ready = (
        text.startswith("MIT License\n")
        and "Copyright (c) 2026 Qianfeng Yuan" in text
        and "Permission is hereby granted" in text
    )
    return ready, "MIT license installed for Qianfeng Yuan" if ready else "LICENSE is not the approved MIT text"


def _real_smoke_ready(root: Path) -> tuple[bool, str]:
    path = root / "provenance/real_data_smoke_certificate.json"
    if not path.is_file():
        return False, "real-data smoke certificate missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"real-data smoke certificate unreadable: {exc}"
    validator = payload.get("validator") or {}
    post_run = payload.get("post_run_audit") or {}
    ready = (
        payload.get("success") is True
        and payload.get("verification_level") == "reduced_count_real_data_integration"
        and int(validator.get("checks", -1)) == int(validator.get("passed", -2))
        and int(validator.get("failed", -1)) == 0
        and int(validator.get("steps", -1)) == 33
        and post_run.get("scientific_source_unchanged_during_run") is True
        and post_run.get("submitted_reference_unchanged") is True
    )
    if not ready:
        return False, "real-data verification certificate does not record a clean reduced-count pass"
    return True, "reduced-count real-data integration passed across all reported experiment families"


def _checkpoint_hash_ready(root: Path) -> tuple[bool, str]:
    configs = [
        path
        for path in sorted((root / "configs/paper").glob("*.yaml"))
        if yaml.safe_load(path.read_text(encoding="utf-8")).get("experiment")
        not in {"feat", "computational_profile"}
    ]
    missing: list[str] = []
    for path in configs:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        digest = str(payload.get("checkpoint_sha256") or "")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            missing.append(path.name)
    if missing:
        return False, f"exact backbone SHA-256 missing from {len(missing)} configs"
    return True, f"exact backbone SHA-256 recorded in {len(configs)} configs"


def _feat_asset_ready(root: Path) -> tuple[bool, str]:
    payload = json.loads((root / "provenance/provenance_manifest.json").read_text(encoding="utf-8"))
    feat = next((row for row in payload.get("assets", []) if row.get("id") == "feat"), None)
    if not isinstance(feat, dict):
        return False, "FEAT external-asset record missing"
    digest = str(feat.get("sha256") or "")
    revision = str(feat.get("source_revision") or "")
    ready = bool(re.fullmatch(r"[0-9a-fA-F]{64}", digest) and revision and "unknown" not in revision.lower())
    return ready, "FEAT source revision/checkpoint digest recorded" if ready else "FEAT source revision/checkpoint digest not yet recorded"


def run_release_check(root: str | Path) -> dict[str, Any]:
    repository = Path(root).expanduser().resolve()
    checks = [
        _attempt("required_files", "error", lambda: _required_files(repository)),
        _attempt("version_consistency", "error", lambda: _version_consistency(repository)),
        _attempt("submitted_artifacts", "error", lambda: _submitted_artifacts(repository)),
        _attempt("submitted_eol_protection", "error", lambda: _gitattributes(repository)),
        _attempt("paper_configs", "error", lambda: _paper_configs(repository)),
        _attempt("provenance_targets", "error", lambda: _provenance(repository)),
        _attempt("forbidden_files", "error", lambda: _forbidden_files(repository)),
        _attempt("private_path_and_secret_scan", "error", lambda: _text_scan(repository)),
    ]
    structural_passed = all(check.passed for check in checks if check.severity == "error")
    license_ready, license_detail = _license_ready(repository)
    backbone_ready, backbone_detail = _checkpoint_hash_ready(repository)
    feat_ready, feat_detail = _feat_asset_ready(repository)
    smoke_ready, smoke_detail = _real_smoke_ready(repository)
    blockers = [
        {"name": "software_license", "resolved": license_ready, "detail": license_detail},
        {"name": "paper_backbone_identity", "resolved": backbone_ready, "detail": backbone_detail},
        {"name": "feat_asset_identity", "resolved": feat_ready, "detail": feat_detail},
        {"name": "real_data_release_smoke", "resolved": smoke_ready, "detail": smoke_detail},
    ]
    report = {
        "kind": "pvse_release_check_v1",
        "repository": str(repository),
        "version": __version__,
        "structural_passed": structural_passed,
        "public_release_ready": structural_passed and all(row["resolved"] for row in blockers),
        "checks": [asdict(check) for check in checks],
        "public_release_blockers": blockers,
        "verification_level": "reduced_count_real_data_integration",
    }
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run_release_check(args.root)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")
    passed = bool(report["public_release_ready"] if args.require_public_ready else report["structural_passed"])
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
