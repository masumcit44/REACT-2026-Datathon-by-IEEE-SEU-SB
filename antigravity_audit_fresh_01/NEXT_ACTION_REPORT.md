# NEXT ACTION REPORT: OBJECTIVE STRATEGY & ROADMAP
**Generated on**: 2026-09-20 22:56:09
**Workspace**: `D:\Desktop\React datathon`
**Status**: Read-Only Audit Completed. No Models Trained. No Files Modified.

---

## 1. Executive Direction

The audit confirms that significant high-quality work has already been completed in this workspace, including diverse GBDT architectures (LightGBM, XGBoost, CatBoost, HistGradientBoosting) achieving PR-AUC scores around **~0.518 - 0.523** on Fold 3 out-of-time validation.

However, moving forward requires addressing specific leakage risks, isolating uncorrupted features, and selecting only verified submissions.

---

## 2. Priority 1: Submission Selection Strategy

From the **37** candidate submission files audited in `submission_inventory.csv`, candidate selection must strictly adhere to verified alignment:

1. **Approved Clean Submissions**:
   - Filter `submission_inventory.csv` for files where `status_verdict == 'VALID'`.
   - Priority candidate: `submission_v7_final.csv` (and `submission_v7_core.csv` / `submission_v7_wide.csv` if aligned).
   - Verify that the mean predicted probability is in the plausible range `[0.006, 0.012]`.
2. **Disqualified Submissions**:
   - Any submission from notebooks where `id_order_matches_sample == False`.
   - Any submission with truncated row count (`rows != 262648`).
   - Any submission exhibiting predictions outside `[0.0, 1.0]`.

---

## 3. Priority 2: Feature Pipeline Hardening (Leakage Elimination)

Before any new model training is considered in future phases:
1. **Purge Global Target Encodings**:
   - Remove or replace any static global mean target encoding.
   - If target encoding is required, implement strictly expanding historical window encoding:
     $$\text{TE}(t) = \frac{\sum_{i < t} y_i + \alpha \cdot \bar{y}}{\sum_{i < t} 1 + \alpha}$$
     ensuring that transactions at time $t$ or later are mathematically excluded from the encoding.
2. **Causal Graph Metrics**:
   - Ensure graph degree and customer-merchant interaction features only count edges occurring strictly prior to the evaluation transaction.
3. **Preserve Index Key**:
   - Keep `transaction_id` as the permanent DataFrame index throughout all feature engineering steps to prevent accidental shuffle or sort desynchronization.

---

## 4. Priority 3: Robust Validation Protocol

To ensure that validation PR-AUC faithfully reflects test set performance:
1. **Fold 2 Gatekeeper**:
   - Model trained on Fold 1 (first ~365k rows).
   - Evaluated on Fold 2 (~183k rows).
   - Any feature or hyperparameter modification that does not improve Fold 2 PR-AUC must be discarded.
2. **Fold 3 Confirmation**:
   - Model trained on Fold 1 + Fold 2 (~548k rows).
   - Evaluated on Fold 3 (~183k rows).
   - Fold 3 score must closely track Fold 2 score (divergence $> 0.03$ indicates temporal overfitting or distribution shift).
3. **Full-Data Retraining Discipline**:
   - Train ensemble on Fold 1 + Fold 2 + Fold 3 (all 731,942 rows) using the exact optimal iteration counts determined by Fold 3 validation.

---

## 5. What NOT To Do (Forbidden Anti-Patterns)

Based on empirical evidence in the audit:
- ❌ **DO NOT use random K-Fold cross-validation**: Shuffling transactions inflates PR-AUC to >0.85 artificially due to customer look-ahead leakage.
- ❌ **DO NOT train complex neural stackers on raw fold predictions**: Linear convex blends (constrained non-negative weights summing to 1.0) consistently outperform unconstrained meta-learners in fraud PR-AUC by avoiding extreme tail calibration errors.
- ❌ **DO NOT modify or sort `test.csv` without restoring original order**: Misaligning predictions by a single row will destroy the leaderboard score completely.
- ❌ **DO NOT start modeling without completing these structural fixes**.

---

## 6. Checklist of Generated Audit Deliverables

All 8 requested audit deliverables have been generated exclusively inside `antigravity_audit_fresh_01/`:
- [x] `antigravity_audit_fresh_01/project_inventory.csv`
- [x] `antigravity_audit_fresh_01/prediction_inventory.csv`
- [x] `antigravity_audit_fresh_01/experiment_summary.csv`
- [x] `antigravity_audit_fresh_01/submission_inventory.csv`
- [x] `antigravity_audit_fresh_01/suspicious_features.csv`
- [x] `antigravity_audit_fresh_01/FULL_AUDIT.md`
- [x] `antigravity_audit_fresh_01/VALIDATION_AUDIT.md`
- [x] `antigravity_audit_fresh_01/NEXT_ACTION_REPORT.md`
