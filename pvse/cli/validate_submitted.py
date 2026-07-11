from __future__ import annotations

import argparse
import json
from pathlib import Path

from pvse.artifacts.validate import validate_submitted_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate byte integrity and arithmetic invariants in the submitted CMT data package."
    )
    parser.add_argument("root", nargs="?", default="artifacts/submitted/supporting_data")
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, summary = validate_submitted_results(args.root)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
