from __future__ import annotations

import argparse
from pathlib import Path

from pvse.cli.common import load_paper_backbone, print_json
from pvse.experiments.miniimagenet import CleanRunConfig, RuntimeOptions, run_clean_experiment
from pvse.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real miniImageNet PVSE-C calibration and test pipeline (CMT Table 4)."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--val-episodes", type=int, default=300)
    parser.add_argument("--test-episodes", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--skip-a0-only", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
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
    write_json(out / "checkpoint_load_report.json", report.to_dict())
    config = CleanRunConfig(
        data_root=args.data_root,
        output_dir=str(out),
        val_episodes=int(args.val_episodes),
        test_episodes=int(args.test_episodes),
        bootstrap_samples=int(args.bootstrap_samples),
        run_a0_only=not bool(args.skip_a0_only),
        run_full=not bool(args.skip_full),
        runtime=RuntimeOptions(
            device=args.device,
            batch_size=int(args.batch_size),
            amp=bool(args.amp),
            channels_last=bool(args.channels_last),
        ),
    )
    result = run_clean_experiment(backbone, config)
    print_json(result)


if __name__ == "__main__":
    main()
