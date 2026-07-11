# Real-data release verification

## Status

**Passed on 2026-07-11.** The run used real miniImageNet and external images, the exact paper ResNet-12 checkpoint, the fixed FEAT source revision and checkpoint, and reduced episode/query counts for every reported experiment family.

## Verification summary

- Validator checks: `238 / 238` passed
- Pipeline steps: `33 / 33` passed
- Paper backbone SHA-256: `763f0ae0a2233f16a9bd551ea22e2997e95dde8efdeb79270e52d9bd87bd1568`
- FEAT commit: `47bdc7c1672e00b027c67469d0291e7502918950`
- FEAT checkpoint SHA-256: `ee54213c754b2c25d07bb7b032549140d7be8d4e253f597cfa57a314696191ee`

## Covered paths

The verification exercised real image loading and transforms; strict backbone and FEAT loading; 640-D global and 640×5×5 spatial features; the 2,626-action lattice; clean switch/A0-only/FULL editing; noisy hard/soft reliability at 20/40/60%; patch ablations; Table 6 policies; TraNFS-style operators; all 22 external settings; FEAT clean/noisy integration; computational profiling; postflight label/mask invariance; and before/after submitted-artifact hashes.

The formal configurations in `configs/paper/` retain the full experiment counts used by the paper. The machine-readable verification record is [`provenance/real_data_smoke_certificate.json`](../provenance/real_data_smoke_certificate.json).
