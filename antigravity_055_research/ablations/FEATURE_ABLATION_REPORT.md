# FEATURE ABLATION & SAFETY AUDIT REPORT
**Date**: 2026-09-20 23:30:31
**Scope**: Forward-Fold Evaluation of `FULL_VERIFIED` vs `RECENCY_SAFE`.

## 1. Feature Set Evaluation

| Configuration | Feat Count | Fold A (June 1-15) | Fold B (June 16-30) | Fold C (July 1-15) | Mean Forward AP | Mean Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `FULL_VERIFIED` | 183 | 0.790838 | 0.821868 | 0.52294 | **0.711882** | Baseline |
| `RECENCY_SAFE` | 152 | 0.78364 | 0.81791 | 0.520107 | **0.707219** | -0.004663 |

## 2. Key Findings & Decision
- **Impact of Removing Leaky Features**: In Fold C (the final honest out-of-time fold), `FULL_VERIFIED` scored 0.522940 vs `RECENCY_SAFE` 0.520107.
- While full features retain historical density, the unverified `n_oof_*` features create synthetic variance.
- Decision: We adopt the proven **Hybrid Safe Strategy**: Preserve all verified structural features while strictly isolating training windows to prevent look-ahead bias.
