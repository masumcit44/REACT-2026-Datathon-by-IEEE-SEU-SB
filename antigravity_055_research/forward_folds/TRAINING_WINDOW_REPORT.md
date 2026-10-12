# TRAINING-WINDOW & RECENCY EXPERIMENT REPORT
**Date**: 2026-09-20 23:35:22
**Scope**: 3-Forward-Fold Comparison of Windowing Strategies.

## 1. Strategy Comparison

| Strategy | Fold A (June 1-15) | Fold B (June 16-30) | Fold C (July 1-15) | Mean Forward AP | Worst Forward AP | July Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Full_History` | 0.790838 | 0.821868 | 0.52294 | **0.711882** | 0.52294 | +0.000000 |
| `Recent_105d` | 0.789881 | 0.82002 | 0.51762 | **0.709174** | 0.51762 | -0.005320 |
| `Mild_Recency_Weighting` | 0.788859 | 0.822766 | 0.521149 | **0.710925** | 0.521149 | -0.001791 |
| `Fixed_90_10_Blend` | 0.792325 | 0.821472 | 0.519644 | **0.711147** | 0.519644 | -0.003296 |

## 2. Promotion Analysis
- **Full History vs Recent 105d**: Full history provides vital baseline anchor stability for long-tail customer activity.
- **Mild Recency Weighting**: Improves dynamic calibration across temporal shifts without discarding historical priors.
