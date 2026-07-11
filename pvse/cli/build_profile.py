from __future__ import annotations

import argparse

from pvse.cli.common import print_json
from pvse.experiments.profile import build_computational_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the structural computational profile (CMT Table 11)."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean-query-outputs", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_computational_profile(
        args.output_dir,
        clean_query_outputs=args.clean_query_outputs or None,
    )
    print_json(result)


if __name__ == "__main__":
    main()
