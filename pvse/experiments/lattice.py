from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pvse.editing.delete_lattice import build_action_bank
from pvse.eval.lattice import evaluate_episode_lattice
from pvse.experiments.miniimagenet import EpisodeProtocol, RuntimeOptions, iter_episode_features
from pvse.models.resnet12 import RFSResNet12Backbone
from pvse.utils.io import write_csv, write_json, write_jsonl


@dataclass(frozen=True)
class LatticeRunConfig:
    data_root: str
    output_dir: str
    split: Literal["val", "test"]
    episodes: int = 1000
    seed: int = 3920
    seed_offset: int = 0
    kmax: int = 3
    protocol: EpisodeProtocol = EpisodeProtocol()
    runtime: RuntimeOptions = RuntimeOptions()


_DENSITY_BINS: tuple[tuple[str, float, float | None], ...] = (
    ("0-1%", 0.0, 0.01),
    ("1-5%", 0.01, 0.05),
    ("5-10%", 0.05, 0.10),
    (">10%", 0.10, None),
)
_MARGIN_BINS: tuple[tuple[str, float | None, float | None], ...] = (
    ("<=0.02", None, 0.02),
    ("0.02-0.05", 0.02, 0.05),
    ("0.05-0.10", 0.05, 0.10),
    ("0.10-0.20", 0.10, 0.20),
    (">0.20", 0.20, None),
)


def _density_label(value: float) -> str:
    for label, lower, upper in _DENSITY_BINS:
        if value > lower and (upper is None or value <= upper):
            return label
    raise ValueError(f"rescue density is outside expected range: {value}")


def _margin_label(value: float) -> str:
    for label, lower, upper in _MARGIN_BINS:
        if (lower is None or value > lower) and (upper is None or value <= upper):
            return label
    raise ValueError(f"margin is outside expected range: {value}")


def summarize_lattice_rows(rows: list[dict[str, Any]], *, split_label: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("no lattice rows supplied")
    total = len(rows)
    counts = {scene: sum(str(r["scene_type"]) == scene for r in rows) for scene in ("safe", "rescue", "dead")}
    oracle = counts["safe"] + counts["rescue"]
    table1 = {
        "Split": split_label,
        "Safe": counts["safe"] / total,
        "Rescue": counts["rescue"] / total,
        "Dead": counts["dead"] / total,
        "Oracle acc.": oracle / total,
        "Oracle gain": counts["rescue"] / total,
        "query_count": total,
    }

    rescue_rows = [r for r in rows if str(r["scene_type"]) == "rescue"]
    density_distribution: list[dict[str, Any]] = []
    delete_composition: list[dict[str, Any]] = []
    for label, _lower, _upper in _DENSITY_BINS:
        group = [r for r in rescue_rows if _density_label(float(r["correct_action_density"])) == label]
        density_distribution.append(
            {
                "Rescue-density bin": label,
                "Share": len(group) / max(1, len(rescue_rows)),
                "n": len(group),
            }
        )
        k1 = sum(int(r["correct_actions_k1"]) for r in group)
        k2 = sum(int(r["correct_actions_k2"]) for r in group)
        k3 = sum(int(r["correct_actions_k3"]) for r in group)
        denom = max(1, k1 + k2 + k3)
        delete_composition.append(
            {
                "Rescue-density bin": label,
                "Delete-1": 100.0 * k1 / denom,
                "Delete-2": 100.0 * k2 / denom,
                "Delete-3": 100.0 * k3 / denom,
                "correct_delete_actions": k1 + k2 + k3,
            }
        )

    margin_composition: list[dict[str, Any]] = []
    for label, _lower, _upper in _MARGIN_BINS:
        group = [r for r in rows if _margin_label(float(r["a0_margin"])) == label]
        denom = max(1, len(group))
        margin_composition.append(
            {
                "A0 margin bin": label,
                "Safe": 100.0 * sum(str(r["scene_type"]) == "safe" for r in group) / denom,
                "Rescue": 100.0 * sum(str(r["scene_type"]) == "rescue" for r in group) / denom,
                "Dead": 100.0 * sum(str(r["scene_type"]) == "dead" for r in group) / denom,
                "n": len(group),
            }
        )
    return {
        "table1": table1,
        "fig2_rescue_density_distribution": density_distribution,
        "fig2_delete_size_composition": delete_composition,
        "fig3_a0_margin_case_composition": margin_composition,
    }


def run_lattice_experiment(
    backbone: RFSResNet12Backbone,
    config: LatticeRunConfig,
) -> dict[str, Any]:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", asdict(config))
    bank = build_action_bank(config.protocol.way * config.protocol.shot, config.kmax)
    query_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for episode in iter_episode_features(
        backbone,
        data_root=config.data_root,
        split=config.split,
        episodes=config.episodes,
        seed=config.seed + config.seed_offset,
        protocol=config.protocol,
        runtime=config.runtime,
    ):
        evaluated = evaluate_episode_lattice(
            episode.support_features,
            episode.support_labels,
            episode.query_features,
            episode.query_labels,
            action_bank=bank,
            kmax=config.kmax,
            way=config.protocol.way,
            logit_scale=config.protocol.logit_scale,
        )
        for row in evaluated.rows():
            query_rows.append(
                {
                    "split": config.split,
                    "episode": episode.episode,
                    **row,
                }
            )
        manifests.append(episode.manifest)
    write_csv(out / "lattice_query_outputs.csv", query_rows)
    write_jsonl(out / "episode_manifests.jsonl", manifests)
    split_label = f"{config.split.capitalize()} {config.episodes}"
    summary = summarize_lattice_rows(query_rows, split_label=split_label)
    write_csv(out / "table1_action_density.csv", [summary["table1"]])
    write_csv(
        out / "fig2_rescue_density_distribution.csv",
        summary["fig2_rescue_density_distribution"],
    )
    write_csv(
        out / "fig2_delete_size_composition.csv",
        summary["fig2_delete_size_composition"],
    )
    write_csv(
        out / "fig3_a0_margin_case_composition.csv",
        summary["fig3_a0_margin_case_composition"],
    )
    result = {
        "experiment": "hard_delete_lattice",
        "formal_scope": "CMT Figure 1, Figures 2-3, and Table 1",
        "split": config.split,
        "episodes": config.episodes,
        "action_count": len(bank),
        "delete_action_count": len(bank) - 1,
        **summary,
    }
    write_json(out / "lattice_results.json", result)
    return result
