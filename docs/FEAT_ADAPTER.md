# FEAT adapter boundary

FEAT is an independent frozen downstream model. The adapter is bound to upstream commit `47bdc7c1672e00b027c67469d0291e7502918950` and expected checkpoint SHA-256 `ee54213c754b2c25d07bb7b032549140d7be8d4e253f597cfa57a314696191ee`.

- FEAT retains its resize/crop, normalization, encoder, and native set-to-set logits.
- Clean gating uses the recovered 12-feature FEAT representation and FEAT-specific hard-delete realization.
- Noisy plug-in results use hard deletion; no FEAT soft-support-weighting claim is made.
- Both global-only and global+patch noisy branches are available for the reported Table 7 comparison.
- The upstream source/checkpoint remain external. This repository records identity and integration code only.
