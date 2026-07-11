from pathlib import Path
import json
import numpy as np
from pvse.editing.counterfactual import choose_strict_hard_delete
from pvse.clean.verifier import CleanVerifierConfig
from pvse.core.prototypes import DEFAULT_LOGIT_SCALE

ROOT=Path(__file__).resolve().parents[1]

def test_clean_hd_uses_submitted_raw_logit_objective():
    d=json.loads((ROOT/'tests/fixtures/clean_hd_objective_regression.json').read_text())
    decision=choose_strict_hard_delete(
        np.asarray(d['support_features_float32'],dtype=np.float32),
        np.asarray(d['support_labels_int64'],dtype=np.int64),
        np.asarray(d['query_feature_float32'],dtype=np.float32),
        a0_prediction=d['a0'], proposal=d['proposal'], scope='full', kmax=3,
        edit_cost=d['generator']['edit_cost'], logit_scale=d['generator']['temperature'])
    assert list(decision.selected_action)==d['expected_submitted_raw_logit_objective']['deleted']
    assert list(decision.selected_action)!=d['current_release_probability_objective']['deleted']


def test_clean_verifier_constants():
    cfg=CleanVerifierConfig()
    assert cfg.lambda_hurt==2.0
    assert cfg.threshold==5.35311817754396
    assert cfg.penalty=='l1' and cfg.solver=='liblinear' and cfg.C==0.5


def test_default_logit_scale(): assert DEFAULT_LOGIT_SCALE==10.0
