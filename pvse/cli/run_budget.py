from __future__ import annotations

import argparse
from pathlib import Path

from pvse.cli.common import load_paper_backbone, print_json
from pvse.experiments.budget import BudgetRunConfig, run_budget_experiment
from pvse.experiments.miniimagenet import RuntimeOptions
from pvse.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real Table-6 PVSE-R budget ablation."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--output-dir", required=True)
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
    result = run_budget_experiment(
        backbone,
        BudgetRunConfig(
            data_root=args.data_root,
            output_dir=str(out),
            train_episodes=args.train_episodes,
            val_episodes=args.val_episodes,
            test_episodes=args.test_episodes,
            bootstrap_samples=args.bootstrap_samples,
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
