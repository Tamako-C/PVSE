from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class ArtifactCheck:
    source: str
    relative_path: str
    expected_size: int | None
    observed_size: int | None
    expected_sha256: str
    observed_sha256: str | None
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _check_one(
    root: Path,
    *,
    source: str,
    relative_path: str,
    expected_sha256: str,
    expected_size: int | None,
) -> ArtifactCheck:
    path = root / relative_path
    if not path.is_file():
        return ArtifactCheck(
            source=source,
            relative_path=relative_path,
            expected_size=expected_size,
            observed_size=None,
            expected_sha256=expected_sha256,
            observed_sha256=None,
            status="missing",
        )
    size = path.stat().st_size
    digest = _digest(path)
    status = "pass"
    if expected_size is not None and size != expected_size:
        status = "mismatch"
    if digest.lower() != expected_sha256.lower():
        status = "mismatch"
    return ArtifactCheck(
        source=source,
        relative_path=relative_path,
        expected_size=expected_size,
        observed_size=int(size),
        expected_sha256=expected_sha256,
        observed_sha256=digest,
        status=status,
    )


def verify_manifest(root: str | Path) -> list[ArtifactCheck]:
    root = Path(root)
    manifest = root / "MANIFEST.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    checks: list[ArtifactCheck] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            checks.append(
                _check_one(
                    root,
                    source="MANIFEST.csv",
                    relative_path=row["file"],
                    expected_sha256=row["sha256"],
                    expected_size=int(row["size_bytes"]),
                )
            )
    return checks


def verify_checksums(root: str | Path) -> list[ArtifactCheck]:
    root = Path(root)
    checksum_file = root / "CHECKSUMS.sha256"
    if not checksum_file.is_file():
        raise FileNotFoundError(checksum_file)
    checks: list[ArtifactCheck] = []
    for raw_line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, relative_path = raw_line.split(maxsplit=1)
        checks.append(
            _check_one(
                root,
                source="CHECKSUMS.sha256",
                relative_path=relative_path.strip(),
                expected_sha256=digest.strip(),
                expected_size=None,
            )
        )
    return checks


def verify_submitted_artifacts(root: str | Path) -> tuple[list[ArtifactCheck], dict]:
    checks = verify_manifest(root) + verify_checksums(root)
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    summary = {
        "root": str(Path(root)),
        "total_rows": len(checks),
        "status_counts": counts,
        "passed": bool(checks) and all(check.status == "pass" for check in checks),
    }
    return checks, summary
