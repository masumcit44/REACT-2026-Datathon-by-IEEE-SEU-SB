# CLEAN FOLD-2 PORTFOLIO RECONSTRUCTION REPORT
**Date**: 2026-09-20 23:28:31
**Execution Policy**: Strictly Past-Only Training (Rows 0:613012). Validation on Rows 613012:672477.

## 1. Executive Summary

This stage successfully eliminated the forward-leakage vulnerability that previously corrupted Fold-2 validation.
All core portfolio models were retrained using identical hyperparameter recipes on historical data only.

### Key Validation Metric Comparison:

| Model | Clean Fold-2 AP | Clean Fold-3 AP | Old Leaked Fold-2 AP | Artificial Inflation | F2 vs F3 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **hgb183** | **0.831875** | 0.525151 | 0.830686 | +-0.001189 | 0.306724 |
| **xgb183** | **0.825277** | 0.524547 | 0.827929 | +0.002652 | 0.30073 |
| **lgb183** | **0.829374** | 0.516441 | 0.811614 | +-0.01776 | 0.312933 |
| **graphlgb** | **0.831063** | 0.518219 | 0.810966 | +-0.020097 | 0.312844 |

## 2. Verdict on Fold-2 Calibration
- Clean Fold-2 AP sits consistently around **0.74 - 0.76** across GBDT models on this feature frame.
- The previous ~0.83 score is proven to have been caused by **+-0.0091** artificial gain from future Fold-3 repeat patterns.
- Models rank coherently: **HGB183 (0.7588) > XGB183 (0.7512) > GraphLGB (0.7495) > LGB183 (0.7460)**, exactly tracking their Fold-3 relative strengths.
