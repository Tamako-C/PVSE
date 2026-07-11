import numpy as np
import pytest
from pvse.editing.delete_lattice import action_count, build_action_bank
from pvse.core.prototypes import compute_prototypes, prototype_predict, weights_from_deleted
from pvse.noisy.corruption import apply_main_corruption
from pvse.noisy.reliability import PAPER_RELIABILITY_CONFIGS, reliability_weights
from pvse.noisy.policies import fixed_topk_keep, threshold_cap_keep, per_class_topr_keep
from pvse.clean.features import CLEAN_VERIFIER_FEATURES
from pvse.noisy.features import GLOBAL_RELIABILITY_FEATURES, RELIABILITY_FEATURES
from pvse.patch.correspondence import CLEAN_PATCH_FEATURES, NOISY_PATCH_FEATURES
from pvse.eval.bootstrap import paired_episode_bootstrap

@pytest.mark.parametrize(('k','expected'),[(0,1),(1,26),(2,326),(3,2626)])
def test_action_counts(k,expected): assert action_count(25,k)==expected

def test_full_action_bank_structure():
    bank=build_action_bank(25,3)
    assert len(bank)==2626
    assert bank.keep_masks.shape==(2626,25)
    assert np.bincount(bank.k_values,minlength=4).tolist()==[1,25,300,2300]
    assert len(bank.single_delete_action_ids)==25

@pytest.mark.parametrize('deleted',[(),(0,),(0,5,24)])
def test_weights_from_deleted(deleted):
    w=weights_from_deleted(25,deleted)
    assert w.shape==(25,)
    assert int(np.sum(w==0))==len(deleted)


def test_prototype_normalization_and_prediction():
    support=np.eye(5,dtype=np.float32).repeat(5,axis=0)
    labels=np.repeat(np.arange(5),5)
    protos=compute_prototypes(support,labels,way=5)
    np.testing.assert_allclose(np.linalg.norm(protos,axis=1),1,atol=1e-6)
    result=prototype_predict(support,labels,np.eye(5,dtype=np.float32),way=5)
    np.testing.assert_array_equal(result.predictions,np.arange(5))
    np.testing.assert_allclose(result.probabilities.sum(axis=1),1,atol=1e-6)

@pytest.mark.parametrize(('severity','beta','wmin'),[(20,1.5,0.0),(40,1.5,0.5),(60,1.0,0.1)])
def test_paper_reliability_constants(severity,beta,wmin):
    cfg=PAPER_RELIABILITY_CONFIGS[severity]
    assert cfg.beta==beta and cfg.w_min==wmin

@pytest.mark.parametrize(('beta','wmin'),[(1.5,0.0),(1.5,0.5),(1.0,0.1)])
def test_reliability_weight_formula(beta,wmin):
    p=np.array([0.0,0.25,0.5,0.75,1.0],dtype=np.float32)
    got=reliability_weights(p,beta=beta,w_min=wmin)
    exp=np.clip(1-beta*p,wmin,1)
    np.testing.assert_allclose(got,exp,atol=1e-7)

@pytest.mark.parametrize(('r','expected'),[(1,5),(2,10),(3,15)])
def test_main_corruption_counts_and_semantics(r,expected):
    rng=np.random.default_rng(41820)
    features=np.arange(25*4,dtype=np.float32).reshape(25,4)
    labels=np.repeat(np.arange(5),5)
    out=apply_main_corruption(features,labels,corrupt_per_class=r,rng=rng)
    assert len(out.records)==expected
    targets=[x.support_index for x in out.records]
    assert len(set(targets))==expected
    for rec in out.records:
        assert rec.source_label != rec.original_label
        np.testing.assert_array_equal(out.features[rec.support_index],features[rec.source_index])


def test_corruption_deterministic_for_seed():
    features=np.arange(25*3,dtype=np.float32).reshape(25,3); labels=np.repeat(np.arange(5),5)
    a=apply_main_corruption(features,labels,corrupt_per_class=2,rng=np.random.default_rng(9))
    b=apply_main_corruption(features,labels,corrupt_per_class=2,rng=np.random.default_rng(9))
    assert a.records==b.records
    np.testing.assert_array_equal(a.features,b.features)


def test_fixed_top3_and_threshold_cap_zero():
    p=np.zeros(25,dtype=np.float32)
    assert int(np.sum(fixed_topk_keep(p,3)==0))==3
    assert int(np.sum(threshold_cap_keep(p,threshold=0.5,max_delete=3)==0))==0


def test_per_class_topr():
    p=np.linspace(0,1,25,dtype=np.float32); labels=np.repeat(np.arange(5),5)
    keep=per_class_topr_keep(p,labels,r=1)
    assert int(np.sum(keep==0))==5
    assert all(int(np.sum(keep[labels==c]==0))==1 for c in range(5))


def test_feature_schema_lengths():
    assert len(CLEAN_VERIFIER_FEATURES)==41
    assert len(GLOBAL_RELIABILITY_FEATURES)==13
    assert len(NOISY_PATCH_FEATURES)==27
    assert len(RELIABILITY_FEATURES)==40
    assert len(CLEAN_PATCH_FEATURES)+8==41


def test_episode_bootstrap_is_deterministic_and_episode_level():
    episode=np.repeat(np.arange(8),4)
    base=np.tile([1,0,1,0],8).astype(bool)
    method=base.copy(); method[::7]=~method[::7]
    a=paired_episode_bootstrap(episode,base,method,samples=200,seed=5200)
    b=paired_episode_bootstrap(episode,base,method,samples=200,seed=5200)
    assert a==b
    assert a.resampling_unit=='episode'


def test_global_only_reliability_metadata_records_executed_feature_set():
    from pvse.experiments.miniimagenet import _paper_reliability_config_for_run

    global_only = _paper_reliability_config_for_run(20, "global_only")
    patch = _paper_reliability_config_for_run(20, "global_plus_patch")
    assert global_only.feature_set == "global_only"
    assert patch.feature_set == "global_plus_patch"
    # Shared immutable paper constants are not mutated.
    assert PAPER_RELIABILITY_CONFIGS[20].feature_set == "global_plus_patch"
    assert global_only.beta == patch.beta == 1.5
    assert global_only.w_min == patch.w_min == 0.0
