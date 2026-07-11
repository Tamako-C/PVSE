from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import re

import pandas as pd

from pvse.artifacts.verify import verify_submitted_artifacts


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _close(observed: float, expected: float, tolerance: float = 0.0011) -> bool:
    """Compare displayed percentage-point values at the paper's precision."""
    return bool(abs(float(observed) - float(expected)) <= float(tolerance))


def _percent(value: Any) -> float:
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    return float(text)


def _signed_number(value: Any) -> float:
    text = str(value).strip().replace(",", "")
    match = re.match(r"^([+-]?\d+(?:\.\d+)?)", text)
    if match is None:
        raise ValueError(f"no leading numeric value in {value!r}")
    return float(match.group(1))


def _signed_integer(value: Any) -> int:
    return int(round(_signed_number(value)))


def _append_gain_check(
    checks: list[ValidationCheck],
    *,
    name: str,
    baseline_percent: float,
    method_percent: float,
    displayed_gain: Any,
    tolerance: float = 0.0011,
) -> None:
    computed = float(method_percent - baseline_percent)
    displayed = _signed_number(displayed_gain)
    checks.append(
        ValidationCheck(
            name,
            _close(computed, displayed, tolerance),
            f"method-baseline={computed:+.6f}pp; displayed={displayed:+.3f}pp",
        )
    )


def validate_submitted_results(root: str | Path) -> tuple[list[ValidationCheck], dict[str, Any]]:
    """Validate byte integrity and arithmetic invariants in the CMT artifact tree.

    The submitted CSVs contain display-rounded percentages. Checks therefore use
    exact integer identities where available and a documented display tolerance
    for values rounded to three decimals.
    """
    root = Path(root)
    checks: list[ValidationCheck] = []
    _, artifact_summary = verify_submitted_artifacts(root)
    checks.append(
        ValidationCheck(
            "byte_exact_artifacts",
            bool(artifact_summary["passed"]),
            f"{artifact_summary['status_counts']} over {artifact_summary['total_rows']} manifest/checksum rows",
        )
    )

    tables = root / "main_paper_exact" / "tables"

    table1 = pd.read_csv(tables / "table1_action_density.csv")
    for _, row in table1.iterrows():
        safe = _percent(row["Safe"])
        rescue = _percent(row["Rescue"])
        dead = _percent(row["Dead"])
        oracle = _percent(row["Oracle acc."])
        oracle_gain = _signed_number(row["Oracle gain"])
        checks.append(
            ValidationCheck(
                f"table1_partition_{row['Split']}",
                _close(safe + rescue + dead, 100.0, 0.02),
                f"safe+rescue+dead={safe + rescue + dead:.4f}%",
            )
        )
        checks.append(
            ValidationCheck(
                f"table1_oracle_identity_{row['Split']}",
                _close(safe + rescue, oracle, 0.011)
                and _close(rescue, oracle_gain, 0.011),
                f"safe+rescue={safe + rescue:.4f}%; oracle={oracle:.4f}%; gain={oracle_gain:+.4f}pp",
            )
        )

    table4 = pd.read_csv(tables / "table4_clean_test1000.csv")
    baseline_acc = _percent(table4[table4["Method"] == "A0 prototype"].iloc[0]["Acc."])
    expected_clean = {
        "PVSE-C-Switch reference": (1967, 1456, 0.681),
        "PVSE-C-HD A0-only": (1967, 1449, 0.691),
        "PVSE-C-HD FULL": (1967, 1453, 0.685),
    }
    for method, (help_count, hurt_count, shown_gain) in expected_clean.items():
        row = table4[table4["Method"] == method].iloc[0]
        observed_pair = str(row["Help/Hurt"])
        observed_net = _signed_integer(row["Net"])
        gain_from_counts = 100.0 * (help_count - hurt_count) / 75_000
        method_acc = _percent(row["Acc."])
        checks.append(
            ValidationCheck(
                f"table4_counts_{method}",
                observed_pair == f"{help_count}/{hurt_count}"
                and observed_net == help_count - hurt_count
                and _close(gain_from_counts, shown_gain),
                f"help/hurt={observed_pair}; net={observed_net}; net/75000={gain_from_counts:.6f}pp",
            )
        )
        _append_gain_check(
            checks,
            name=f"table4_accuracy_gain_{method}",
            baseline_percent=baseline_acc,
            method_percent=method_acc,
            displayed_gain=row["Gain (episode CI)"],
            tolerance=0.0016,
        )

    table5 = pd.read_csv(tables / "table5_noisy_severity.csv")
    for noise, net, shown_gain in (("20%", 1133, 1.511), ("40%", 2520, 3.360), ("60%", 2666, 3.555)):
        row = table5[table5["Noise"] == noise].iloc[0]
        gain_from_net = 100.0 * net / 75_000
        observed_net = _signed_integer(row["Net"])
        checks.append(
            ValidationCheck(
                f"table5_net_{noise}",
                observed_net == net and _close(gain_from_net, shown_gain),
                f"net={observed_net}; net/75000={gain_from_net:.6f}pp; displayed={row['Gain (episode CI)']}",
            )
        )
        _append_gain_check(
            checks,
            name=f"table5_accuracy_gain_{noise}",
            baseline_percent=_percent(row["Noisy A0"]),
            method_percent=_percent(row["PVSE-R-Soft"]),
            displayed_gain=row["Gain (episode CI)"],
            tolerance=0.0016,
        )

    table6 = pd.read_csv(tables / "table6_budget_ablation.csv")
    expected_table6 = {
        ("per-class-1", "Soft weighting"): 1.491,
        ("per-class-1", "Budget<=3 hard delete"): 0.723,
        ("per-class-1", "Fixed top-3 delete"): 0.932,
        ("per-class-1", "Per-class delete r = 1"): 1.593,
        ("corrupt-total-3", "Soft transfer"): 0.844,
        ("corrupt-total-3", "Budget<=3 hard delete"): 0.736,
        ("corrupt-total-3", "Fixed top-3 delete"): 0.752,
    }
    for key, expected in expected_table6.items():
        protocol, method = key
        hit = table6[(table6["Protocol"] == protocol) & (table6["Method"] == method)]
        observed = _signed_number(hit.iloc[0]["Gain"]) if len(hit) == 1 else float("nan")
        checks.append(
            ValidationCheck(
                f"table6_{protocol}_{method}",
                len(hit) == 1 and _close(observed, expected, 0.0001),
                f"observed={observed:+.3f}pp; expected={expected:+.3f}pp",
            )
        )

    table8 = pd.read_csv(tables / "table8_tranfs_style.csv")
    for _, row in table8.iterrows():
        base = _percent(row["Noisy A0"])
        for label, acc_col, gain_col in (
            ("soft", "PVSE-R-Soft acc.", "PVSE-R-Soft gain (episode CI)"),
            ("hard", "PVSE-R-HD acc.", "PVSE-R-HD gain (episode CI)"),
            ("oracle", "Oracle acc.", "Oracle gain (episode CI)"),
        ):
            _append_gain_check(
                checks,
                name=f"table8_{row['Protocol']}_{row['Noise']}_{label}",
                baseline_percent=base,
                method_percent=_percent(row[acc_col]),
                displayed_gain=row[gain_col],
                tolerance=0.0016,
            )

    table9 = pd.read_csv(tables / "table9_external_clean.csv")
    external_expected = {
        "CUB": (300, 22_500),
        "Caltech101": (300, 22_500),
        "DTD": (300, 22_500),
        "FGVC-Aircraft": (300, 22_500),
        "OfficeHome": (1700, 127_500),
        "PACS": (1700, 127_500),
    }
    for dataset, (episodes, queries) in external_expected.items():
        displayed = str(table9[table9["Dataset"] == dataset].iloc[0]["Episodes / queries"])
        normalized = displayed.replace(",", "").replace(" ", "")
        checks.append(
            ValidationCheck(
                f"table9_{dataset}_episode_query_count",
                normalized == f"{episodes}/{queries}",
                f"submitted={displayed}; expected queries=episodes*5*15={episodes * 75}",
            )
        )

    table10 = pd.read_csv(tables / "table10_feat_plugin.csv")
    for _, row in table10.iterrows():
        _append_gain_check(
            checks,
            name=f"table10_{row['Noise']}",
            baseline_percent=_percent(row["FEAT native"]),
            method_percent=_percent(row["FEAT+PVSE"]),
            displayed_gain=row["Gain (episode CI)"],
            tolerance=0.0016,
        )

    summary = {
        "root": str(root),
        "passed": all(check.passed for check in checks),
        "check_count": len(checks),
        "passed_count": sum(int(check.passed) for check in checks),
        "checks": [check.to_dict() for check in checks],
    }
    return checks, summary
