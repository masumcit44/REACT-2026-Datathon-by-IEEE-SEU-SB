# VALIDATION AUDIT REPORT: TEMPORAL SPLIT & LEAKAGE ANALYSIS
**Generated on**: 2026-09-20 22:56:09
**Audit Scope**: Validation Code Integrity, Temporal Partitioning, Target Encodings, and Row Alignment
**Target Metric**: PR-AUC (Average Precision) on Fraud Class

---

## 1. Executive Validation Verdict

> **Validation Integrity Assessment**: **MIXED / HIGH RISK IF UNGUARDED**
> 
> While several pipelines (`v7_members.py`, `v8_conservative_blends.py`) implement temporal fold splitting, multiple critical vulnerabilities were uncovered in script implementations across `v9`, `v10`, and ad-hoc blending routines:
> 1. **Global feature pre-computation** without strict fold isolation.
> 2. **Target encoding risk** in graph and category features.
> 3. **Array index alignment risks** when merging predictions from disparate runs.

---

## 2. Temporal Split Architecture

### 2.1 The 3-Fold Temporal Split Scheme
In transaction fraud detection, random K-fold splits are strictly invalid because future transactions would leak into historical training data. The primary validation design in this workspace employs an out-of-time temporal scheme across `train.csv` (731,942 transactions):

- **Total Chronological Span**: Monotonically sorted by `timestamp`.
- **Fold 1**: Historical buffer / warm-up window (~365,972 rows).
- **Fold 2 (Intermediate Out-of-Time Test)**: ~182,985 rows.
  - *Correct Training Set*: Strictly Fold 1.
  - *Correct Validation Set*: Fold 2.
- **Fold 3 (Final Out-of-Time Validation)**: ~182,985 rows.
  - *Correct Training Set*: Fold 1 + Fold 2.
  - *Correct Validation Set*: Fold 3.
- **Competition Test Set (`test.csv`)**: 262,648 rows strictly subsequent to Fold 3.

### 2.2 Critical Check: Did Fold-3 Rows Leak Into Fold-2 Training?
We audited all scripts that train Fold 2 models (`v5_pipeline.py`, `v6_pipeline.py`, `v7_members.py`, `fold2_gate_check.py`, `v9_train.py`, etc.):
- **Finding**: In `v7_members.py` and `fold2_gate_check.py`, Fold 2 training explicitly uses `X.iloc[:fold2_start]` (Fold 1 only), and Fold 3 training uses `X.iloc[:fold3_start]` (Fold 1 + Fold 2). This conforms to temporal causality.
- **Hazard Identified in v9 / v10 graph scripts**: In `v9_graph_a.py`, `v9_graph_b.py`, and `v10_feature_gen.py`, bipartite graphs and customer-merchant interaction frequencies are constructed using **all available training rows simultaneously** before fold slicing. Consequently, merchant degree, customer degree, and interaction edge weights for Fold 2 validation contain edges that occurred during Fold 3.
  - *Impact*: Mild to moderate look-ahead leakage in graph topological features.

---

## 3. Target Encoding & Fraud Rate Features Audit

### 3.1 Suspicious Feature Names Scan
The workspace Parquet files and scripts were audited for suspicious names including `oof`, `target_enc`, `fraud_rate`, `mean_target`, `risk`:

| Source File | Feature Name | Severity | Risk Classification | Context & Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_merchant_id` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_raw_merchant_id` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_count_merchant_id` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_device_id` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_raw_device_id` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_count_device_id` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_customer_id` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_raw_customer_id` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_count_customer_id` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_merchant_category` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_raw_merchant_category` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_count_merchant_category` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_device_type` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_raw_device_type` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_count_device_type` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_location` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_raw_location` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_count_location` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_payment_method` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_raw_payment_method` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_count_payment_method` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_transaction_type` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_raw_transaction_type` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_count_transaction_type` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| `COLAB_REQUIRED_FILES/test_combined.parquet` | `n_oof_merch_x_amt` | **High** | Target / Feature Encoding Risk | Contains explicit target encoding / OOF / fraud rate term in column name |
| ... *(265 additional entries in suspicious_features.csv)* | | | | |


### 3.2 Target Leakage Deep Dive:
1. **Pre-split Global Target Means**:
   - Any feature computed as `df.groupby('merchant_id')['fraud'].transform('mean')` on the entire dataset leaks the validation label directly into the training row.
   - In `v4` feature generation, global merchant fraud rates were calculated without out-of-fold cross-fitting or causal time masking.
2. **Expanding Window vs. Static Aggregation**:
   - Customer historical fraud counts must strictly use past transactions (e.g. `closed='left'` or `.shift(1)`).
   - Scripts using unshifted cumulative statistics inadvertently leak the current transaction's outcome into its own features.

---

## 4. Prediction Array Alignment & Integrity Check

### 4.1 Row Alignment Verification
Ensembling multiple model outputs via numpy operations (e.g., `0.4 * p1 + 0.6 * p2`) is only valid if both arrays share the identical row order.
- **Test Set Alignment**:
  - `test.csv` has 262,648 rows.
  - Any sorting by `timestamp` or `customer_id` during feature extraction permutes the rows.
  - If a model predicts on a sorted test dataframe and the predictions are assigned back as `sample_submission['fraud'] = preds` without resetting or mapping by `transaction_id`, the predictions will be assigned to the wrong transactions!
- **Audit Findings**:
  - `v7_finalize.py` and `v8_conservative_blends.py` preserve test ordering by mapping via `test['transaction_id']`.
  - However, in several exploratory notebook scripts (`AGENT_EXP_RAPIDFIRE`, `EXP_CHANNEL_TRANSITIONS`), `.sort_values(['customer_id', 'timestamp'])` was executed in-place on the test dataframe without restoring the initial index before writing predictions.
  - **Verdict**: Only submissions generated with explicit ID verification (`id_order_matches_sample == True` in `submission_inventory.csv`) are reliable.

---

## 5. Summary Validation Checklist

| Validation Safeguard | Status | Comments |
| :--- | :---: | :--- |
| **Strict Temporal Splitting** | **PARTIAL** | Core fold split is chronological, but global feature extraction leaks graph/frequency stats. |
| **No Fold 3 in Fold 2 Training** | **PASSED** | Core modeling scripts train Fold 2 on Fold 1 only. |
| **Zero Random K-Fold Splitting** | **PASSED** | Random K-fold is avoided in primary pipelines; 3-fold temporal is standard. |
| **Leak-Free Target Encoding** | **FAILED** | Global target encodings and unshifted group aggregates detected in early feature parquets. |
| **Proven Row Alignment** | **CONDITIONAL** | Top-level submission scripts (`v7_finalize.py`) pass; experimental notebook exports fail. |
