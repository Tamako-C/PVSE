# Result traceability

| Paper item | Paper config | Primary generated artifact | Submitted reference |
|---|---|---|---|
| Table 1 validation | `lattice_val.yaml` | `table1_action_density.csv` | `main_paper_exact/tables/table1_action_density.csv` |
| Table 1 test | `lattice_test.yaml` | `table1_action_density.csv` | same reference |
| Figures 2–3 | `lattice_val.yaml` | three `fig*.csv` files | `main_paper_exact/figures_data/fig*.csv` |
| Table 4 | `clean_table4.yaml` | `clean_results.json`; `clean_query_outputs.csv` | `table4_clean_test1000.csv` |
| Table 5 | `noisy_table5_{20,40,60}.yaml` | `noisy_results.json` | `table5_noisy_severity.csv` |
| Table 6 | `budget_table6.yaml` | `table6_budget_ablation_generated.csv` | `table6_budget_ablation.csv` |
| Table 7 main | `patch_ablation_table7_{20,40,60}.yaml` plus Table 5 runs | `noisy_results.json` | `table7_patch_ablation.csv` |
| Table 7 FEAT | `feat_table10.yaml` | FEAT noisy level summaries | `table7_patch_ablation.csv` |
| Table 8 | `tranfs_table8.yaml` | `table8_tranfs_style_generated.csv` | `table8_tranfs_style.csv` |
| Table 9 | `external_table9.yaml` | `table9_external_clean_generated.csv` | `table9_external_clean.csv` |
| Table 10 | `feat_table10.yaml` | FEAT clean/noisy summaries | `table10_feat_plugin.csv` |
| Table 11 | `computational_profile_table11.yaml` | `table11_computational_profile_generated.csv` | `table11_computational_profile.csv` |

The machine-readable mapping is stored in `provenance/provenance_manifest.json`.

## Comparison discipline

Result comparison uses unrounded internal values before applying the paper display format. Confidence intervals are recomputed with paired episode resampling (`B=10000`). The submitted package supplies the reference values, while the prediction runners generate outputs from images, model assets, and the formal configurations.
