from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from pvse.clean.verifier import CleanVerifierConfig, load_clean_verifier
from pvse.cli.common import load_paper_backbone, print_json
from pvse.config import ConfigError, dataclass_kwargs, load_paper_config, optional_path, required_path
from pvse.experiments.budget import BudgetRunConfig, run_budget_experiment
from pvse.experiments.external import ExternalRunConfig, run_external_clean_experiment
from pvse.experiments.feat import FeatRunConfig, run_feat_experiment
from pvse.experiments.lattice import LatticeRunConfig, run_lattice_experiment
from pvse.experiments.miniimagenet import (
    CleanRunConfig,
    EpisodeProtocol,
    NoisyRunConfig,
    RuntimeOptions,
    run_clean_experiment,
    run_noisy_experiment,
)
from pvse.experiments.profile import build_computational_profile
from pvse.experiments.tranfs import TraNFSRunConfig, run_tranfs_experiment
from pvse.plugins.feat import FeatProtocol
from pvse.utils.environment import environment_report
from pvse.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one CMT-scoped experiment from a checked-in YAML config."
    )
    parser.add_argument("config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a nested YAML value; may be repeated",
    )
    parser.add_argument("--print-effective", action="store_true")
    return parser


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return dict(value)


def _write_metadata(config: dict[str, Any], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "effective_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_json(out / "environment.json", environment_report())


def _runtime(config: dict[str, Any]) -> RuntimeOptions:
    return RuntimeOptions(**dataclass_kwargs(RuntimeOptions, _mapping(config, "runtime"), context="runtime"))


def _protocol(config: dict[str, Any]) -> EpisodeProtocol:
    return EpisodeProtocol(**dataclass_kwargs(EpisodeProtocol, _mapping(config, "protocol"), context="protocol"))


def _backbone(config: dict[str, Any], runtime: RuntimeOptions):
    checkpoint = required_path(config, "checkpoint")
    expected = str(config.get("checkpoint_sha256") or "")
    return load_paper_backbone(checkpoint, device=runtime.device, expected_sha256=expected)


def run_effective_config(config: dict[str, Any]) -> Any:
    experiment = str(config["experiment"])
    parameters = _mapping(config, "parameters")

    if experiment == "computational_profile":
        output = required_path(config, "output_dir")
        _write_metadata(config, output)
        return build_computational_profile(
            output,
            clean_query_outputs=optional_path(config, "clean_query_outputs"),
        )

    if experiment == "feat":
        output = required_path(config, "output_dir")
        protocol = FeatProtocol(
            **dataclass_kwargs(FeatProtocol, _mapping(config, "protocol"), context="protocol")
        )
        values = {
            "data_root": required_path(config, "data_root"),
            "feat_root": required_path(config, "feat_root"),
            "checkpoint": required_path(config, "checkpoint"),
            "output_dir": output,
            **_mapping(config, "runtime"),
            **parameters,
            "protocol": protocol,
        }
        # YAML lists are made immutable in the dataclass.
        if "severities" in values:
            values["severities"] = tuple(int(v) for v in values["severities"])
        config_obj = FeatRunConfig(
            **dataclass_kwargs(FeatRunConfig, values, context="FEAT parameters")
        )
        _write_metadata(config, output)
        return run_feat_experiment(config_obj)

    runtime = _runtime(config)
    protocol = _protocol(config)
    output = required_path(config, "output_dir")
    backbone, checkpoint_report = _backbone(config, runtime)
    _write_metadata(config, output)
    write_json(Path(output) / "checkpoint_load_report.json", checkpoint_report.to_dict())

    if experiment == "external":
        verifier = load_clean_verifier(required_path(config, "verifier_bundle"))
        values = {
            "index_csv": required_path(config, "index_csv"),
            "output_dir": output,
            "runtime": runtime,
            "protocol": protocol,
            **parameters,
        }
        obj = ExternalRunConfig(
            **dataclass_kwargs(ExternalRunConfig, values, context="external parameters")
        )
        return run_external_clean_experiment(backbone, verifier, obj)

    common = {
        "data_root": required_path(config, "data_root"),
        "output_dir": output,
        "runtime": runtime,
        "protocol": protocol,
    }
    if experiment == "clean":
        values = {**common, **parameters}
        if "verifier" in values:
            raw_verifier = values["verifier"]
            if not isinstance(raw_verifier, dict):
                raise ConfigError("parameters.verifier must be a mapping")
            values["verifier"] = CleanVerifierConfig(
                **dataclass_kwargs(
                    CleanVerifierConfig, raw_verifier, context="clean verifier parameters"
                )
            )
        obj = CleanRunConfig(
            **dataclass_kwargs(CleanRunConfig, values, context="clean parameters")
        )
        return run_clean_experiment(backbone, obj)
    if experiment == "noisy":
        obj = NoisyRunConfig(
            **dataclass_kwargs(NoisyRunConfig, {**common, **parameters}, context="noisy parameters")
        )
        return run_noisy_experiment(backbone, obj)
    if experiment == "lattice":
        obj = LatticeRunConfig(
            **dataclass_kwargs(LatticeRunConfig, {**common, **parameters}, context="lattice parameters")
        )
        return run_lattice_experiment(backbone, obj)
    if experiment == "budget":
        obj = BudgetRunConfig(
            **dataclass_kwargs(BudgetRunConfig, {**common, **parameters}, context="budget parameters")
        )
        return run_budget_experiment(backbone, obj)
    if experiment == "tranfs":
        values = {**common, **parameters}
        for key in ("protocols", "severities"):
            if key in values:
                values[key] = tuple(values[key])
        obj = TraNFSRunConfig(
            **dataclass_kwargs(TraNFSRunConfig, values, context="TraNFS parameters")
        )
        return run_tranfs_experiment(backbone, obj)
    raise ConfigError(f"unsupported experiment: {experiment}")


def main() -> None:
    args = build_parser().parse_args()
    config = load_paper_config(args.config, overrides=args.set)
    if args.print_effective:
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), end="")
        return
    print_json(run_effective_config(config))


if __name__ == "__main__":
    main()
