# Reported experiments

This repository implements the experiments reported in the accepted paper and supplementary material.

| Paper item | Computation | Formal configuration |
|---|---|---|
| Figure 1 / Table 1 | Complete `delete <= 3` lattice over 25 supports; 2,626 actions; safe/rescue/dead and oracle analysis | `lattice_val.yaml`, `lattice_test.yaml` |
| Figures 2–3 | Rescue-action density, delete-size composition, and A0-margin composition | `lattice_val.yaml` |
| Table 4 | Clean A0, PVSE-C-Switch, PVSE-C-HD A0-only, and PVSE-C-HD FULL | `clean_table4.yaml` |
| Table 5 | Main 20/40/60% within-episode replacement protocol and PVSE-R-Soft | `noisy_table5_{20,40,60}.yaml` |
| Table 6 | Per-class-1 and total-3 soft/hard budget policies | `budget_table6.yaml` |
| Table 7 | Global-only versus global+patch reliability for the main backbone and FEAT | `patch_ablation_table7_{20,40,60}.yaml`, `feat_table10.yaml` |
| Table 8 | TraNFS-style symmetric and paired replacement at 20/40/60% | `tranfs_table8.yaml` |
| Table 9 | Fixed clean transfer to CUB, Caltech101, DTD, FGVC-Aircraft, OfficeHome, and PACS | `external_table9.yaml` |
| Table 10 | Frozen FEAT clean and noisy plug-in | `feat_table10.yaml` |
| Table 11 | Candidate counts and clean apply-rate profile | `computational_profile_table11.yaml` |
| Supplement | Calibration, features, policies, freeze rules, bootstrap protocol, and result traceability | `configs/paper/` and submitted reference artifacts |

All high-level experiment runners are available through `pvse-run-config` and the specialized command-line entry points listed in the repository README.
