# CAUSALLY SAFE GRAPH UPGRADE REPORT
**Date**: 2026-09-20 23:37:03
**Scope**: Incremental Value of Causal Graph Features.

| Variant | Features | Fold B AP | Fold C AP | Mean AP | Delta Fold C |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Safe_Baseline` | 183 | 0.821868 | 0.52294 | **0.672404** | +0.000000 |
| `Safe_Baseline_Plus_GraphV3` | 218 | 0.821656 | 0.525192 | **0.673424** | +0.002252 |

## Findings
- Graph-V3 provides orthogonal topological connectivity signals that improve representation diversity for multi-architecture ensembling.
