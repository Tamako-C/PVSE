from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SubmittedTable:
    name: str
    relative_path: str
    dataframe: pd.DataFrame


def load_submitted_tables(root: str | Path) -> dict[str, SubmittedTable]:
    root = Path(root)
    table_dir = root / "main_paper_exact" / "tables"
    out: dict[str, SubmittedTable] = {}
    for path in sorted(table_dir.glob("table*.csv")):
        out[path.stem] = SubmittedTable(
            name=path.stem,
            relative_path=str(path.relative_to(root)),
            dataframe=pd.read_csv(path, encoding="utf-8-sig"),
        )
    return out


def submitted_result_summary(root: str | Path) -> dict[str, Any]:
    tables = load_submitted_tables(root)
    return {
        "table_count": len(tables),
        "tables": {
            name: {
                "relative_path": table.relative_path,
                "rows": int(len(table.dataframe)),
                "columns": [str(c) for c in table.dataframe.columns],
            }
            for name, table in tables.items()
        },
    }
