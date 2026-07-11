from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import traceback
from typing import Callable

import numpy as np
import torch

from pvse.artifacts.validate import validate_submitted_results
from pvse.clean.features import CLEAN_VERIFIER_FEATURES, clean_feature_rows
from pvse.core.prototypes import prototype_predict
from pvse.editing.delete_lattice import action_count, build_action_bank
from pvse.models.resnet12 import RFSResNet12Backbone
from pvse.noisy.corruption import apply_main_corruption
from pvse.noisy.features import RELIABILITY_FEATURES, reliability_feature_rows
from pvse.noisy.policies import threshold_cap_keep


@dataclass(frozen=True)
class SmokeCheck:
    name: str
    passed: bool
    detail: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic synthetic interface checks. This is a core-code smoke test, "
            "not a reproduction of the paper's numerical results."
        )
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/submitted/supporting_data",
        help="byte-exact CMT supporting-data root",
    )
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser


def _run(name: str, fn: Callable[[], str]) -> SmokeCheck:
    try:
        return SmokeCheck(name=name, passed=True, detail=str(fn()))
    except Exception as exc:  # smoke report must expose all failing checks
        return SmokeCheck(
            name=name,
            passed=False,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}",
        )


def _core_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(20260501)
    support = rng.normal(size=(25, 32)).astype(np.float32)
    labels = np.repeat(np.arange(5), 5).astype(np.int64)
    queries = rng.normal(size=(4, 32)).astype(np.float32)
    support_maps = torch.from_numpy(rng.normal(size=(25, 32, 5, 5)).astype(np.float32))
    query_maps = torch.from_numpy(rng.normal(size=(4, 32, 5, 5)).astype(np.float32))
    return support, labels, queries, support_maps, query_maps


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(1)
    support, labels, queries, support_maps, query_maps = _core_fixture()
    checks: list[SmokeCheck] = []

    def lattice_check() -> str:
        full = len(build_action_bank(25, 3))
        a0_only = action_count(5, 3)
        if full != 2626 or a0_only != 26:
            raise AssertionError(f"unexpected action counts: full={full}, a0_only={a0_only}")
        return f"full={full}, a0_only={a0_only}"

    checks.append(_run("delete_lattice", lattice_check))

    def prototype_check() -> str:
        prediction = prototype_predict(support, labels, queries, way=5, logit_scale=10.0)
        if prediction.logits.shape != (4, 5) or prediction.probabilities.shape != (4, 5):
            raise AssertionError("unexpected prototype output shape")
        if not np.allclose(prediction.probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise AssertionError("probabilities do not sum to one")
        return f"logits={prediction.logits.shape}; probabilities normalized"

    checks.append(_run("prototype_classifier", prototype_check))

    def clean_feature_check() -> str:
        rows = clean_feature_rows(
            support,
            labels,
            queries,
            support_maps,
            query_maps,
            true_labels=np.asarray([0, 1, 2, 3]),
        )
        missing = [name for name in CLEAN_VERIFIER_FEATURES if name not in rows[0]]
        if len(rows) != 4 or len(CLEAN_VERIFIER_FEATURES) != 41 or missing:
            raise AssertionError(f"rows/features mismatch; missing={missing}")
        return f"rows={len(rows)}; verifier_features={len(CLEAN_VERIFIER_FEATURES)}"

    checks.append(_run("clean_feature_schema", clean_feature_check))

    def corruption_and_reliability_check() -> str:
        corrupted = apply_main_corruption(
            support,
            labels,
            rng=np.random.default_rng(41820),
            corrupt_per_class=1,
            support_maps=support_maps,
        )
        if int(corrupted.corrupt_mask.sum()) != 5 or corrupted.maps is None:
            raise AssertionError("unexpected corruption count or missing maps")
        rows = reliability_feature_rows(
            corrupted.features,
            labels,
            corrupted.maps,
            corrupt_mask=corrupted.corrupt_mask,
        )
        missing = [name for name in RELIABILITY_FEATURES if name not in rows[0]]
        if len(rows) != 25 or len(RELIABILITY_FEATURES) != 40 or missing:
            raise AssertionError(f"rows/features mismatch; missing={missing}")
        return f"corrupt={int(corrupted.corrupt_mask.sum())}; rows=25; reliability_features=40"

    checks.append(_run("corruption_and_reliability_schema", corruption_and_reliability_check))

    def adaptive_delete_check() -> str:
        keep = threshold_cap_keep(np.zeros(25, dtype=np.float32), threshold=0.7, max_delete=3)
        if not np.all(keep == 1.0):
            raise AssertionError("threshold-cap deleted supports below threshold")
        return "all-zero probabilities retain all supports"

    checks.append(_run("adaptive_delete_allows_k0", adaptive_delete_check))

    def backbone_check() -> str:
        model = RFSResNet12Backbone().eval()
        with torch.inference_mode():
            out = model.forward_with_spatial(torch.zeros(1, 3, 84, 84))
        if tuple(out.global_features.shape) != (1, 640):
            raise AssertionError(tuple(out.global_features.shape))
        if tuple(out.spatial_features.shape) != (1, 640, 5, 5):
            raise AssertionError(tuple(out.spatial_features.shape))
        return "84x84 -> global[1,640], spatial[1,640,5,5]"

    checks.append(_run("backbone_interface", backbone_check))

    if not args.skip_artifacts:
        def artifact_check() -> str:
            _, summary = validate_submitted_results(args.artifact_root)
            if not summary["passed"]:
                failed = [c["name"] for c in summary["checks"] if not c["passed"]]
                raise AssertionError(f"submitted validation failed: {failed}")
            return f"{summary['passed_count']}/{summary['check_count']} validation checks passed"

        checks.append(_run("submitted_artifacts", artifact_check))

    summary = {
        "kind": "synthetic_core_smoke",
        "paper_reproduction": False,
        "passed": all(check.passed for check in checks),
        "passed_count": sum(int(check.passed) for check in checks),
        "check_count": len(checks),
        "checks": [asdict(check) for check in checks],
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
