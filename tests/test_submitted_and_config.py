from pathlib import Path
import json
import pytest
from pvse.artifacts.verify import verify_submitted_artifacts
from pvse.artifacts.validate import validate_submitted_results
from pvse.config import load_paper_config
from pvse.cli.release_check import run_release_check

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/submitted/supporting_data'
CONFIGS=sorted((ROOT/'configs/paper').glob('*.yaml'))

@pytest.mark.parametrize('path', CONFIGS, ids=lambda p:p.name)
def test_all_paper_configs_parse(path):
    cfg=load_paper_config(path)
    assert cfg['schema_version']==1
    assert cfg['experiment']


def test_exact_config_count():
    assert len(CONFIGS)==14


def test_submitted_manifest_and_checksums():
    rows,summary=verify_submitted_artifacts(ART)
    assert len(rows)==121
    assert summary['passed'] is True


def test_submitted_arithmetic_invariants():
    rows,summary=validate_submitted_results(ART)
    assert summary['passed'] is True
    assert summary['passed_count']==summary['check_count']==52


def test_structural_release_check():
    report=run_release_check(ROOT)
    assert report['structural_passed'] is True
    assert next(x for x in report['public_release_blockers'] if x['name']=='paper_backbone_identity')['resolved'] is True
    assert next(x for x in report['public_release_blockers'] if x['name']=='feat_asset_identity')['resolved'] is True
    assert report['public_release_ready'] is True
    assert next(x for x in report['public_release_blockers'] if x['name']=='software_license')['resolved'] is True
    assert next(x for x in report['public_release_blockers'] if x['name']=='real_data_release_smoke')['resolved'] is True


def test_external_dispatch_does_not_require_miniimagenet_data_root(monkeypatch, tmp_path):
    """Table 9 consumes a prebuilt external index, not paths.data_root."""
    from types import SimpleNamespace
    import pvse.cli.run_config as runner

    sentinel_backbone = object()
    sentinel_verifier = object()
    monkeypatch.setattr(
        runner,
        "_backbone",
        lambda config, runtime: (
            sentinel_backbone,
            SimpleNamespace(to_dict=lambda: {"complete": True}),
        ),
    )
    monkeypatch.setattr(runner, "_write_metadata", lambda config, output: None)
    monkeypatch.setattr(runner, "write_json", lambda path, value: None)
    monkeypatch.setattr(runner, "load_clean_verifier", lambda path: sentinel_verifier)

    observed = {}

    def fake_external(backbone, verifier, config):
        observed.update(backbone=backbone, verifier=verifier, config=config)
        return {"ok": True}

    monkeypatch.setattr(runner, "run_external_clean_experiment", fake_external)
    config = {
        "experiment": "external",
        "paths": {
            "checkpoint": str(tmp_path / "backbone.pth"),
            "index_csv": str(tmp_path / "external.csv"),
            "verifier_bundle": str(tmp_path / "clean.joblib"),
            "output_dir": str(tmp_path / "out"),
        },
        "checkpoint_sha256": "0" * 64,
        "runtime": {},
        "protocol": {},
        "parameters": {"max_episodes_per_setting": 1},
    }
    assert runner.run_effective_config(config) == {"ok": True}
    assert observed["backbone"] is sentinel_backbone
    assert observed["verifier"] is sentinel_verifier
    assert observed["config"].index_csv.endswith("external.csv")


def test_release_check_pyproject_version_fallback_on_python310(monkeypatch):
    """The public package declares Python 3.10 support, where tomllib is absent."""
    import builtins
    from pvse.cli.release_check import _pyproject_version

    original_import = builtins.__import__

    def without_tomllib(name, *args, **kwargs):
        if name == "tomllib":
            raise ModuleNotFoundError("simulated Python 3.10")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tomllib)
    assert _pyproject_version(ROOT / "pyproject.toml") == "0.4.0rc2"
