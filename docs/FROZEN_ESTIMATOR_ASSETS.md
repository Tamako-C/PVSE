# Frozen estimator assets

Each archive below is a self-contained release asset for one submitted estimator or estimator-policy combination. Assets may be downloaded and verified independently; no collection archive is required.

Every archive contains:

- a portable `JSON+NPZ` representation;
- a trusted author-environment `joblib` snapshot;
- formal-result and portable/joblib parity records;
- source digests without local source paths;
- `MANIFEST.json`, `MANIFEST.csv`, and `SHA256SUMS.txt`.

After extracting any archive, run:

```bash
pvse-verify-estimator-pack <asset-directory>
pvse-verify-estimator-pack <asset-directory> --allow-joblib
```

The first command does not deserialize joblib. The second should be used only for an asset obtained from a trusted release.

## Clean miniImageNet

| Paper scope | Asset | SHA-256 |
|---|---|---|
| Table 4 PVSE-C verifier | `pvse_clean_verifier_table4_v1.zip` | `888c9ed18b12fffa2f20e6859b8a12db9f146ce5325c5d01d9597246f3803fb8` |

## Main noisy support and budget policies

| Paper scope | Asset | SHA-256 |
|---|---|---|
| Tables 5/7, 20% PVSE-R-Soft | `pvse_r_soft_minimagenet_noise20_v1.zip` | `61fb007389111e40752a60f2ed31bcf16ac97494bea1c89e5940775a4eac42cf` |
| Tables 5/7, 40% PVSE-R-Soft | `pvse_r_soft_minimagenet_noise40_v1.zip` | `160fe2ec63e3c27321f0799e31aa5d93a054ec75fce8eca549f16ed3898bb198` |
| Tables 5/7, 60% PVSE-R-Soft | `pvse_r_soft_minimagenet_noise60_v1.zip` | `2232cbe9a4786cae1d48f7e7559ebde89ad0ae90e21e2335b83a2359908e95fb` |
| Table 6, per-class-1 policies | `pvse_r_budget_perclass1_v1.zip` | `258999d5c35423f2cb5563d7a02a27ead8e9d27126ef7b09b3bf861114ea695a` |
| Table 6, total-3 policies | `pvse_r_budget_total3_v1.zip` | `836776be8b4a83ee11595b9282f4d6753299dd2f3901db36af5e1350a08212c0` |

## TraNFS-style support replacement

| Paper scope | Asset | SHA-256 |
|---|---|---|
| Table 8, symmetric 20%, Soft | `pvse_r_soft_tranfs_symmetric_noise20_v1.zip` | `9b38eb207b34ac32f059e6f351209245008e2ba7a5d3dd393dd55259c63a7460` |
| Table 8, symmetric 20%, HD | `pvse_r_hd_tranfs_symmetric_noise20_v1.zip` | `4fd4548d09fec28de48b58db70a7b3cca2ed88aed94b0eac157b897c17b948a6` |
| Table 8, symmetric 40%, Soft | `pvse_r_soft_tranfs_symmetric_noise40_v1.zip` | `e209ca017de19004ad85cb5752e0fbe39996cab3b88a583e12625322d6d8ce7f` |
| Table 8, symmetric 40%, HD | `pvse_r_hd_tranfs_symmetric_noise40_v1.zip` | `94444dd1eecc8e0dcc4aa7915202036fbb174c14f985ab83e5ec3268828131ce` |
| Table 8, symmetric 60%, Soft | `pvse_r_soft_tranfs_symmetric_noise60_v1.zip` | `259d660d064ec3d345c9fda673cc4ac8f83ea769a1f6780b1dd31c6615acfc22` |
| Table 8, symmetric 60%, HD | `pvse_r_hd_tranfs_symmetric_noise60_v1.zip` | `db405e31ebaa71fc0c9e08110b0870aaf84cafd8589bbec5ce9a087bfd9b2438` |
| Table 8, paired 20%, Soft | `pvse_r_soft_tranfs_paired_noise20_v1.zip` | `a461e59768264064e4c74e6cffabee80d57908e4b38b4018d86575f6be099a98` |
| Table 8, paired 20%, HD | `pvse_r_hd_tranfs_paired_noise20_v1.zip` | `b3afbeaa7086c626afe91df229de053572569276581c9bf19b0be4ecd8183829` |
| Table 8, paired 40%, Soft | `pvse_r_soft_tranfs_paired_noise40_v1.zip` | `c31471eb985260181c227893e8ade1fab50dea03c082e73c934e96a3c0e35ca4` |
| Table 8, paired 40%, HD | `pvse_r_hd_tranfs_paired_noise40_v1.zip` | `ed82585151f453d351fb29ed1fbeec33fb86c78f9f66d5775229ca2b86e9a8b2` |
| Table 8, paired 60%, Soft | `pvse_r_soft_tranfs_paired_noise60_v1.zip` | `26aee75e01b044c6c799259e75903c11fc3e4af540c52183a72b3bdea8b88f88` |
| Table 8, paired 60%, HD | `pvse_r_hd_tranfs_paired_noise60_v1.zip` | `ea92dde2d90043d16aff6434b88202b7e0fcf3538564e2fa5c9cc95a3707a88a` |

## Frozen FEAT plug-in

| Paper scope | Asset | SHA-256 |
|---|---|---|
| Table 10 clean gate | `pvse_c_feat_clean_gate_v1.zip` | `a8b20d6a512bb646e63e9a89fc24c14208a6367ea44a373260d1d5e7bb171482` |
| Tables 7/10, 20% global+patch | `pvse_r_hd_feat_noise20_global_patch_v1.zip` | `aa1bc73848008ba2dacedf09b68479db885bd016b2d5a9e4d4a254f99ef0d01c` |
| Table 7, 20% global-only | `pvse_r_hd_feat_noise20_global_only_v1.zip` | `68fc273e5806933dfd830780f5ecd75118aaec9218ae08bf2ef8b6c2d58d7368` |
| Tables 7/10, 40% global+patch | `pvse_r_hd_feat_noise40_global_patch_v1.zip` | `727f01cefb262f6348a7c8352ce4b7a764bd84203355d86a608f549b6647b51f` |
| Table 7, 40% global-only | `pvse_r_hd_feat_noise40_global_only_v1.zip` | `98fafe0eba34fff5c72e6bdbfd38242316a3a17a6e836d8bd287f36219a47aae` |
| Tables 7/10, 60% global+patch | `pvse_r_hd_feat_noise60_global_patch_v1.zip` | `735271b5dff3ed8c058c6f23d0bd10414a06e9bfd54ea9ee88c008142dd3b04a` |
| Table 7, 60% global-only | `pvse_r_hd_feat_noise60_global_only_v1.zip` | `f11e85b27f0745b2caf8eba810394f127dab66d8a51e16826bbf0da79c2b8b09` |

## Maintainer builders

- `scripts/build_frozen_clean_verifier.py`
- `scripts/build_frozen_noisy_estimators.py`
- `scripts/build_frozen_tranfs_estimators.py`
- `scripts/build_frozen_feat_estimators.py`

The builders validate calibration coverage, fixed configuration selection, submitted display values, and portable/joblib numerical parity before writing deterministic archives.
