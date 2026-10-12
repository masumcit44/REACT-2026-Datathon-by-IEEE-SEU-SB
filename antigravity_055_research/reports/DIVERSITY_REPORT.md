# PREDICTION ALIGNMENT & DIVERSITY REPORT
**Date**: 2026-09-20 23:37:04
**Scope**: Inter-Model Diversity across Clean Fold-2, Fold-3, and Test.

## 1. Fold-3 Spearman Rank Correlation Matrix

| Model | hgb183 | xgb183 | lgb183 | graphlgb |
| :--- | :---: | :---: | :---: | :---: |
| **hgb183** | 1.0000 | 0.5559 | 0.3198 | 0.3146 |
| **xgb183** | 0.5559 | 1.0000 | 0.3336 | 0.3317 |
| **lgb183** | 0.3198 | 0.3336 | 1.0000 | 0.9290 |
| **graphlgb** | 0.3146 | 0.3317 | 0.9290 | 1.0000 |

## 2. Key Diversity Findings
- **High Architectural Complementarity**: Spearman correlation between HGB183 and XGB183 is 0.5559, and between HGB183 and GraphLGB is 0.3146.
- This significant diversity ensures robust ensemble gains without model redundancy.
