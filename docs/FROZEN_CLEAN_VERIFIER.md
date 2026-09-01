# Frozen PVSE-C clean verifier

The Table 4 clean verifier is distributed as a separate GitHub Release asset. It gates PVSE-C-Switch and both PVSE-C-HD deletion scopes, and the same frozen verifier is reused for the Table 9 fixed-transfer evaluation.

## Asset contents

- `portable/clean_verifier.json` fixes the feature order, policy, and array checksum.
- `portable/clean_verifier.npz` stores StandardScaler and logistic-regression arrays for both heads.
- `joblib_author_env/clean_verifier.joblib` is the author-environment scikit-learn snapshot.
- `parity/` records full test1000 parity against the submitted Table 4 result and portable/joblib numerical parity.
- `provenance/` records source hashes and confirms that test rows were not used for fitting or model selection.
- `MANIFEST.json`, `MANIFEST.csv`, and `SHA256SUMS.txt` cover the release contents.

The portable format is recommended for inspection and exchange. Load the joblib snapshot only from a trusted release because joblib files may execute Python objects during deserialization.

## Verify a downloaded asset

```bash
pvse-verify-estimator-pack pvse_clean_verifier_table4_v1
pvse-verify-estimator-pack pvse_clean_verifier_table4_v1 --allow-joblib
```

The first command does not load joblib. It verifies every packaged checksum and exercises portable inference. The second command additionally checks portable output against the trusted author-environment snapshot.

## Rebuild from the formal calibration records

The maintainer build command is intentionally path-explicit:

```bash
python scripts/build_frozen_clean_verifier.py \
  --calibration-rows /path/to/official_validation_runnerup_rows.csv \
  --selected-configuration /path/to/validation_selected_configuration.json \
  --formal-test-rows /path/to/formal_test1000_feature_rows.csv \
  --formal-switch-outputs /path/to/formal_test1000_switch_outputs.csv \
  --submitted-table4 artifacts/submitted/supporting_data/main_paper_exact/tables/table4_clean_test1000.csv \
  --output-dir /path/to/pvse_clean_verifier_table4_v1 \
  --zip-path /path/to/pvse_clean_verifier_table4_v1.zip
```

The builder verifies the formal calibration source digest, validation-selected parameters, episode/query coverage, row alignment, submitted display values, and portable/joblib parity before writing the archive.
