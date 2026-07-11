from __future__ import annotations

import argparse
from pathlib import Path

from pvse.clean.verifier import load_clean_verifier
from pvse.cli.common import load_paper_backbone, print_json
from pvse.experiments.external import ExternalRunConfig, run_external_clean_experiment
from pvse.experiments.miniimagenet import RuntimeOptions
from pvse.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen PVSE-C external fixed transfer (CMT Table 9)."
    )
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--verifier-bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--max-episodes-per-setting", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    backbone, report = load_paper_backbone(
        args.checkpoint,
        device=args.device,
        expected_sha256=args.checkpoint_sha256,
    )
    verifier = load_clean_verifier(args.verifier_bundle)
    write_json(out / "checkpoint_load_report.json", report.to_dict())
    result = run_external_clean_experiment(
        backbone,
        verifier,
        ExternalRunConfig(
            index_csv=args.index_csv,
            output_dir=str(out),
            bootstrap_samples=args.bootstrap_samples,
            max_episodes_per_setting=args.max_episodes_per_setting,
            runtime=RuntimeOptions(
                device=args.device,
                batch_size=args.batch_size,
                amp=args.amp,
                channels_last=args.channels_last,
            ),
        ),
    )
    print_json(result)


if __name__ == "__main__":
    main()
