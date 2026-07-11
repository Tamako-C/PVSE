from __future__ import annotations

import argparse
from dataclasses import replace

from pvse.experiments.feat import FeatRunConfig, run_feat_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real frozen-FEAT plug-in protocols reported in Table 10. "
            "Requires the external FEAT source tree, its checkpoint, and miniImageNet images."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feat-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("clean", "noisy", "both"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--clean-val-episodes", type=int, default=300)
    parser.add_argument("--clean-test-episodes", type=int, default=600)
    parser.add_argument("--noisy-train-episodes", type=int, default=50)
    parser.add_argument("--noisy-val-episodes", type=int, default=50)
    parser.add_argument("--noisy-test-episodes", type=int, default=1000)
    parser.add_argument(
        "--real-data-smoke-episodes",
        type=int,
        default=0,
        help=(
            "Use this many real episodes for every train/val/test stage. This exercises the "
            "actual FEAT/data path; it is not a paper-result reproduction."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = FeatRunConfig(
        data_root=args.data_root,
        feat_root=args.feat_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        mode=args.mode,
        device=args.device,
        batch_size=args.batch_size,
        clean_val_episodes=args.clean_val_episodes,
        clean_test_episodes=args.clean_test_episodes,
        noisy_train_episodes=args.noisy_train_episodes,
        noisy_val_episodes=args.noisy_val_episodes,
        noisy_test_episodes=args.noisy_test_episodes,
    )
    if int(args.real_data_smoke_episodes) > 0:
        count = int(args.real_data_smoke_episodes)
        config = replace(
            config,
            clean_val_episodes=count,
            clean_test_episodes=count,
            noisy_train_episodes=count,
            noisy_val_episodes=count,
            noisy_test_episodes=count,
            bootstrap_samples=min(1000, config.bootstrap_samples),
        )
    run_feat_experiment(config)


if __name__ == "__main__":
    main()
