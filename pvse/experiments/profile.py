from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pvse.editing.delete_lattice import action_count
from pvse.utils.io import write_csv, write_json


def _optional_apply_rate(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("clean query output is empty")
    if "hd_a0_only_deleted" in rows[0]:
        # JSON-list serialization from the curated clean runner.
        import json

        applied = sum(bool(json.loads(row["hd_a0_only_deleted"])) for row in rows)
        definition = "selected A0-only delete set is non-empty"
    elif "hd_a0_only_pred" in rows[0] and "a0_pred" in rows[0]:
        applied = sum(row["hd_a0_only_pred"] != row["a0_pred"] for row in rows)
        definition = "final A0-only prediction differs from A0 (fallback approximation)"
    else:
        raise ValueError(
            "clean query output needs hd_a0_only_deleted, or hd_a0_only_pred plus a0_pred"
        )
    return {
        "source": str(source),
        "query_count": len(rows),
        "applied_count": int(applied),
        "apply_rate": float(applied / len(rows)),
        "definition": definition,
    }


def build_computational_profile(
    output_dir: str | Path,
    *,
    clean_query_outputs: str | Path | None = None,
) -> dict[str, Any]:
    """Generate the CMT Table-11 structural profile from executable constants.

    Candidate counts are derived from the action enumerator. The reported 5.39%
    apply-rate claim can be audited when formal clean per-query outputs are
    supplied; it is deliberately not copied into a generated result otherwise.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full = action_count(25, 3)
    a0_only = action_count(5, 3)
    rows = [
        {
            "Component": "Full lattice oracle",
            "Candidate count": f"{full}/query",
            "Trigger": "analysis only",
            "Purpose": "Exact delete<=3 upper bound",
        },
        {
            "Component": "PVSE-C-HD A0-only",
            "Candidate count": f"{a0_only}/query",
            "Trigger": "after verifier accepts",
            "Purpose": "Deployed clean hard deletion",
        },
        {
            "Component": "PVSE-C-HD FULL",
            "Candidate count": f"{full}/query",
            "Trigger": "ablation",
            "Purpose": "More complete but more expensive",
        },
        {
            "Component": "PVSE-R-Soft",
            "Candidate count": "25 supports",
            "Trigger": "every noisy episode",
            "Purpose": "Support weighting",
        },
        {
            "Component": "PVSE-R-HD",
            "Candidate count": "25 supports",
            "Trigger": "every noisy episode",
            "Purpose": "Support mask / cleaning",
        },
    ]
    write_csv(out / "table11_computational_profile_generated.csv", rows)
    apply_rate = _optional_apply_rate(clean_query_outputs)
    result = {
        "experiment": "computational_profile",
        "formal_scope": "CMT Table 11 and Supplement Table S15",
        "candidate_counts": {
            "full_delete_lattice": full,
            "a0_only_delete_lattice": a0_only,
            "supports_scored_by_pvse_r": 25,
        },
        "clean_a0_only_apply_rate": apply_rate,
        "apply_rate_reference_not_reconstructed_without_formal_query_outputs": apply_rate is None,
    }
    write_json(out / "computational_profile.json", result)
    return result
