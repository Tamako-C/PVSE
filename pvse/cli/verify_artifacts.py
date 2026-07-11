from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pvse.artifacts.verify import verify_submitted_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the byte-exact CMT supporting-data artifact tree")
    parser.add_argument(
        "root",
        nargs="?",
        default="artifacts/submitted/supporting_data",
        help="supporting-data root containing MANIFEST.csv and CHECKSUMS.sha256",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checks, summary = verify_submitted_artifacts(args.root)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [check.to_dict() for check in checks]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
