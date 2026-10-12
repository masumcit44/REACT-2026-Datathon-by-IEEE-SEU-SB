# RESEARCH PLAN: HIGH-PERFORMANCE LEAKAGE-FREE AVERAGE PRECISION ROADMAP
**Project**: REACT 2026 Datathon - Fraud Detection in Mobile Financial Services
**Primary Research Target**: Attempt to reach a robust local Average Precision of 0.550 or higher and produce the strongest valid leaderboard challenger possible.
**Author**: Lead Research Engineer
**Created**: 2026-09-20
**Workspace Root**: `D:\Desktop\React datathon`
**Working Directory**: `D:\Desktop\React datathon\antigravity_055_research`

---

## 1. Executive Research Mandate & Ground Rules

1. **Target Clarity**: 0.550 is an ambitious research stretch target, not a guaranteed outcome. We never fabricate numbers, never overfit validation, and never claim private leaderboard improvements from local validation alone. An honest, leakage-safe, and stable 0.535 is superior to an overfitted or leaky 0.550.
2. **Read-Only Preservation of Prior Work**: Existing repository artefacts (models, submissions, predictions, logs) will never be overwritten, moved, or deleted. All new experiments and outputs reside strictly within `antigravity_055_research/`.
3. **Verified Audit Foundation**: We build strictly on the findings of `antigravity_audit_fresh_01/`:
   - Prior Fold-2 scores of ~0.83 in `fold2_gate_check.py` and `v7_members.py` were **severely contaminated** by future Fold-3 data (`all rows except Fold-2`).
   - Clean Fold-2 training is strictly rows `0:613012` (Jan 1 to June 15, 2026), evaluated on rows `613012:672477` (June 16 to June 30, 2026).
   - Clean Fold-3 training is strictly rows `0:672477` (Jan 1 to June 30, 2026), evaluated on rows `672477:731942` (July 1 to July 15, 2026).
   - Confirmed public leaderboard benchmark: `submission_v4_w4_cost_0518226.csv` (**0.52642**).
   - Prior local Fold-3 AP of 0.530061 (standalone GPU XGBoost) fell to 0.51956 on Public LB and 0.51485 on Private LB. Local single-fold validation is optimistic and must be checked across multiple forward temporal folds.

---

## 2. Validation Architecture & Forward Folds

To ensure genuine temporal stability, all modeling and feature hypotheses are evaluated against a multi-window temporal validation suite:

### 2.1 The Two Primary Historical Folds:
- **Clean Fold-2**:
  - Training window: rows `0:613012` (timestamps `< 2026-06-16 00:00:00`, 614,022 transactions).
  - Validation window: rows `613012:672477` (`2026-06-16` to `2026-06-30`, 58,455 transactions).
  - Target array: `jupyter notebook/EXP_B25_ACCELERATION_FOLD2/fold2_targets.npy`.
- **Clean Fold-3**:
  - Training window: rows `0:672477` (timestamps `< 2026-07-01 00:00:00`, 672,477 transactions).
  - Validation window: rows `672477:731942` (`2026-07-01` to `2026-07-15`, 59,465 transactions).
  - Target array: `jupyter notebook/Fold_3_v4_targets.npy`.

### 2.2 The Three Forward Temporal Date-Folds (for Ablations & Window Tuning):
- **Fold A (Early June)**:
  - Train: timestamps `< 2026-06-01` (~556,547 rows).
  - Validate: `2026-06-01` through `2026-06-15` (~57,475 rows).
- **Fold B (Late June / Clean Fold-2)**:
  - Train: timestamps `< 2026-06-16` (614,022 rows).
  - Validate: `2026-06-16` through `2026-06-30` (58,455 rows).
- **Fold C (Early July / Clean Fold-3)**:
  - Train: timestamps `< 2026-07-01` (672,477 rows).
  - Validate: `2026-07-01` through `2026-07-15` (59,465 rows).

**Rule**: Never compare raw Fold A AP with raw Fold C AP as if they shared identical difficulty. All model and feature comparisons evaluate **same-fold deltas** ($\Delta = \text{AP}_{\text{candidate}} - \text{AP}_{\text{baseline}}$).

---

## 3. Experiment Sequence & Checkpoint Strategy

Execution proceeds sequentially across six distinct stages. Every stage writes an atomic `complete.json` marker upon verified completion:

```
[Stage A: Clean Fold-2 Reconstruction]
                  │
                  ▼
[Stage B: Safe Feature Audit & Forward Ablation]
                  │
                  ▼
[Stage C: Training-Window & Recency Experiments]
                  │
                  ▼
[Stage D: Causally Safe Graph Enhancement]
                  │
                  ▼
[Stage E: Prediction Alignment & Diversity Audit]
                  │
                  ▼
[Stage F: Constrained Multi-Architecture Ensemble]
                  │
                  ▼
[Stage G: Final Challenger & Safety Anchor Generation]
```

### Checkpoint Structure:
- Each stage checks `antigravity_055_research/checkpoints/<stage>_complete.json`.
- If valid checkpoint and output artefacts exist, the stage skips computation and loads precomputed results.
- If interrupted, execution resumes at the first incomplete stage without redundant retraining.

---

## 4. Stage-by-Stage Specifications

### Stage A: Clean Fold-2 Portfolio Reconstruction
- Reconstruct honest, leak-free Fold-2 predictions for core models:
  - `hgb183`: Sklearn HistGradientBoosting (600 trees, lr=0.05, max_leaf_nodes=63, l2=1.0).
  - `xgb183`: XGBoost Hist (depth=11, lr=0.05, colsample=0.7, subsample=0.8, scale_pos_weight).
  - `graphlgb`: LightGBM GBDT (leaves=63, lr=0.05, feature_frac=0.7) on base + Graph-V3.
  - `lgb183`: LightGBM GBDT (leaves=63, lr=0.05, feature_frac=0.7, bagging=0.8).
- Train strictly on rows `0:613012`. Validate on rows `613012:672477`.
- Save clean Fold-2 predictions to `antigravity_055_research/clean_fold2/`.
- Establish true historical Fold-2 baseline score to replace the defective 0.83 score.

### Stage B: Safe Feature Audit & Ablation
- Contrast two feature frameworks across the 3 Forward Date-Folds:
  - **Configuration A (`FULL_VERIFIED`)**: Complete 183-feature matrix with raw and NA-indicator columns, retaining high-signal historical aggregates.
  - **Configuration B (`RECENCY_SAFE`)**: Stripped feature matrix strictly removing all:
    - `n_oof_*` (28 out-of-fold cross-validated target encodings).
    - `n_amt_pctile_*` (5 future group-percentile ranks).
    - Any label-derived fraud ratios and unverified risk rates.
    - Unshifted V10 `.cumsum()` features.
- Promotion Threshold: Configuration B must maintain or improve mean/median fold AP without high-variance decay.

### Stage C: Training-Window & Recency Weighting
- Evaluate sample recency policies on the 3 Forward Date-Folds:
  1. Full-history equal weighting (standard).
  2. Recent-105-day window (~3.5 months).
  3. Mild chronological exponential sample weighting (prior months lower, recent months up to 1.30x).
  4. Convex blend: 90% Recent-105 + 10% Full-history.
- Promotion Rule:
  - July fold delta $\ge 0$.
  - Worst fold delta $\ge -0.001$.
  - Median fold delta $> 0$.
  - Positive in at least 2 of 3 folds.

### Stage D: Causally Safe Graph Enhancement
- Derive strictly past-only graph connectivity features:
  - Customer degree, device degree, merchant degree before current row.
  - Customer-device and customer-merchant interaction edge counts strictly prior to event $t$.
  - Distinct customer counts per device using causal two-pointer time windows.
- Controlled ablation against the Stage B/C baseline.
- Promotion Rule: Median fold delta $\ge +0.001$, worst fold delta $\ge -0.001$, July delta $> 0$.

### Stage E: Prediction Alignment & Diversity
- Verify exact 1-to-1 alignment across all clean candidate predictions (Fold-2, Fold-3, Test).
- Compute:
  - Pearson correlation matrix.
  - Spearman rank correlation matrix.
  - Top-1%, Top-2%, Top-5% capture agreement and disagreement.
- Discard any model with rank correlation $> 0.98$ with an existing higher-scoring model (redundancy pruning).

### Stage F: Constrained Multi-Architecture Ensemble
- Search convex combinations over clean, verified predictions:
  - Algorithms: Probability blending, percentile-rank blending, logit-space blending.
  - Constraints: $w_i \ge 0$, $\sum w_i = 1$, $\max(w_i) \le 0.80$, number of members $\le 4$, step size = 0.05.
  - Objective: Maximize Average Precision across both Clean Fold-2 and Clean Fold-3 without degradation on any forward fold.
- Save best blend models and test prediction arrays.

### Stage G: Final Challenger & Safety Anchor Deliverables
- Verify candidate test predictions against `sample_submission.csv` (262,648 rows, exact ID order, valid probabilities in `[0, 1]`, zero NaNs/Infs).
- Generate:
  - `final/final_challenger_submission.csv`
  - `final/final_safety_anchor_copy.csv` (copy of confirmed 0.52642 champion `submission_v4_w4_cost_0518226.csv`).
  - Comprehensive final reports and reproduction guide.

---

## 5. Forbidden Anti-Patterns (Zero Tolerance)

- ❌ **NO Random K-Fold CV**: Chronological time-series structure must never be permuted.
- ❌ **NO Future Rows in Training**: Never include row $j \ge \text{val\_start}$ in training row selection.
- ❌ **NO Target Leakage**: No out-of-fold target encodings, global target means, or unshifted label aggregates.
- ❌ **NO Retrospective Percentile Ranks**: No `rank(pct=True)` across the entire dataset.
- ❌ **NO Test-Set Desynchronization**: Never sort or filter the test set without restoring the exact initial row ordering.
- ❌ **NO Fabricated Scores**: All scores must originate from reproducible Python calculations saved in `metrics.json`.
- ❌ **NO Nelder-Mead or Unconstrained Meta-Learners**: Unbounded weights lead to extreme tail calibration failures in PR-AUC.

---

## 6. Score Tier Benchmarks

| Tier | PR-AUC Target | Status & Significance |
| :---: | :---: | :--- |
| **Tier 1** | **0.530+** | Solid challenger over confirmed 0.52642 public LB baseline. |
| **Tier 2** | **0.535+** | Strong, verifiable competitive advance. |
| **Tier 3** | **0.540+** | Exceptional multi-architecture ensemble result. |
| **Tier 4** | **0.550+** | Research stretch target (reported only if strictly proven and leakage-free). |

If local stable validation peaks below 0.550 (e.g., at 0.535–0.542), the true honest score will be reported transparently, backed by full temporal proof.
