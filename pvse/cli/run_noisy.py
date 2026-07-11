from __future__ import annotations

import argparse
from pathlib import Path

from pvse.cli.common import load_paper_backbone, print_json
from pvse.experiments.miniimagenet import NoisyRunConfig, RuntimeOptions, run_noisy_experiment
from pvse.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real miniImageNet PVSE-R-Soft severity (CMT Table 5 or Table 7)."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--severity", type=int, choices=(20, 40, 60), required=True)
    parser.add_argument(
        "--feature-set",
        choices=("global_plus_patch", "global_only"),
        default="global_plus_patch",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--train-episodes", type=int, default=50)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
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
    config = NoisyRunConfig(
        data_root=args.data_root,
        output_dir=str(out),
        severity_percent=int(args.severity),
        train_episodes=int(args.train_episodes),
        val_episodes=int(args.val_episodes),
        test_episodes=int(args.test_episodes),
        bootstrap_samples=int(args.bootstrap_samples),
        feature_set=args.feature_set,
        runtime=RuntimeOptions(
            device=args.device,
            batch_size=int(args.batch_size),
            amp=bool(args.amp),
            channels_last=bool(args.channels_last),
        ),
    )
    result = run_noisy_experiment(backbone, config)
    print_json(result)


if __name__ == "__main__":
    main()
