from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pvse.cli.common import print_json
from pvse.data.external import build_external_index
from pvse.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the six-dataset external-transfer image index."
    )
    parser.add_argument(
        "--roots-yaml",
        required=True,
        help="YAML mapping of CUB/Caltech101/DTD/FGVC_Aircraft/OfficeHome/PACS to roots",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--notes-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = yaml.safe_load(Path(args.roots_yaml).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("roots YAML must contain a mapping")
    roots = payload.get("dataset_roots", payload)
    if not isinstance(roots, dict):
        raise ValueError("dataset_roots must be a mapping")
    index, notes = build_external_index(roots)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(output, index=False, encoding="utf-8")
    notes_path = Path(args.notes_json) if args.notes_json else output.with_suffix(".notes.json")
    write_json(notes_path, notes)
    print_json({"rows": len(index), "output_csv": str(output), "notes_json": str(notes_path), "notes": notes})


if __name__ == "__main__":
    main()
