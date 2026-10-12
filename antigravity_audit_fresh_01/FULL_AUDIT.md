# FULL AUDIT REPORT: REACT 2026 DATATHON WORKSPACE
**Generated on**: 2026-09-20 22:56:09
**Workspace**: `D:\Desktop\React datathon`
**Status**: Read-Only Audit Completed (Zero Files Modified/Deleted/Trained)

---

## 1. Executive Summary

This comprehensive, read-only audit examines the entire state of the **REACT 2026 Datathon** workspace. The competition objective is transaction fraud detection in mobile financial services (BDT currency, Bangladesh), evaluated on the **Area Under the Precision-Recall Curve (PR-AUC / Average Precision)**.

### Key Workspace Statistics:
- **Total Tracked Workspace Files**: 619 (excluding `.venv` and `.git`)
- **Python Scripts**: 84
- **Jupyter Notebooks**: 0
- **Numpy Arrays (`.npy`)**: 181
- **Parquet Feature Tables**: 20
- **Candidate Submission CSVs**: 37
- **Documented Experiments/Models**: 68
- **Suspicious Code / Feature Flags**: 290

---

## 2. Competition Specification & Problem Framing

### 2.1 Dataset Structure
- **`train.csv`**: 731942 transactions, 13 columns:
  - Key columns: `transaction_id`, `customer_id`, `timestamp`, `amount_bdt`, `merchant_id`, `merchant_category`, `device_id`, `device_type`, `location`, `payment_method`, `transaction_type`, `account_age_days`, `fraud`.
- **`test.csv`**: 262648 transactions, 12 columns (all features without `fraud`).
- **`sample_submission.csv`**: 262648 rows, exactly 2 columns: `transaction_id`, `fraud`.
- **Evaluation Metric**: PR-AUC (Average Precision on `fraud = 1`). Due to extreme class imbalance (~0.5% fraud rate), PR-AUC is heavily sensitive to high-confidence false positives.
- **Rulebook Highlights**:
  - Out-of-time evaluation: Test data is strictly chronologically future relative to training data.
  - Submissions must preserve exact ID alignment and output probabilities in `[0.0, 1.0]`.

---

## 3. Project Inventory Breakdown

Below is a breakdown of all workspace files by functional category:

| Category | File Count | Total Size (MB) | Primary Role |
| :--- | :--- | :--- | :--- |
| `blend_script` | 12 | 0.05 MB | Blending or stacking script |
| `config_or_data` | 57 | 0.1 MB | JSON configuration or structured data |
| `correlation_file` | 7 | 11.28 MB | Model correlation / diversity analysis table |
| `data_or_experiment_csv` | 70 | 24.9 MB | Data table or experiment record |
| `dataset` | 5 | 223.83 MB | Primary competition raw dataset |
| `documentation` | 5 | 1.53 MB | Markdown documentation or skill description |
| `error_log` | 7 | 0.0 MB | Standard error log |
| `experiment_csv` | 4 | 0.01 MB | Experiment tracking or evaluation table |
| `experiment_json` | 17 | 0.08 MB | JSON validation or search result report |
| `feature_importance` | 26 | 0.09 MB | Feature importance / metrics CSV |
| `feature_metadata` | 5 | 0.01 MB | Pickled feature list / column metadata |
| `feature_script` | 8 | 0.06 MB | Feature engineering or graph generation |
| `npy_prediction_or_target` | 181 | 1572.38 MB | Numpy binary array (predictions, targets, or weights) |
| `other` | 30 | 22.02 MB |  |
| `parquet_feature` | 20 | 1544.81 MB | Parquet feature table or dataset cache |
| `pickle_artifact` | 19 | 444.73 MB | Pickle object / serialized model or cache |
| `report` | 7 | 0.09 MB | Comprehensive markdown report or dossier |
| `run_log` | 19 | 0.06 MB | Console or run execution log |
| `script` | 49 | 0.29 MB | Pipeline, training, or utility script |
| `submission_csv` | 31 | 238.92 MB | Candidate or final submission file |
| `text_file` | 8 | 150.41 MB | Text documentation or note |
| `trained_model` | 17 | 47.5 MB | Trained model binary or model specification |
| `validation_script` | 15 | 0.05 MB | Validation, check, or audit script |


### 3.1 Directory Organization
1. **Root Directory**: Contains core datasets (`train.csv`, `test.csv`, `sample_submission.csv`), baseline feature parquets (`feature_train_v4.parquet`, `feature_test_v4.parquet`), top-level pipeline scripts (`v5_pipeline.py` through `v10_feature_gen.py`), and master blend scripts.
2. **`v5_out/` to `v10_out/`**: Versioned artifact directories containing predictions, trained lightgbm/xgboost/catboost models, blend configurations, and logs.
3. **`jupyter notebook/`**: Research and rapid experiment directory containing sub-experiments (`AGENT_EXP_AUTO_CATBOOST_TUNED`, `AGENT_EXP_AUTO_HGB`, `EXP_B25_ACCELERATION`, `FINAL_0518226_SUBMISSION`, etc.).
4. **`submission_file/`**: Staging directory holding historical submission CSVs generated during prior model iterations.
5. **`COLAB_REQUIRED_FILES/`**: Portable manifest and data assets for Google Colab acceleration.

---

## 4. Prediction Array Inventory (`.npy`)

A total of **181** `.npy` prediction arrays were analyzed programmatically.

| Path | Shape | Dtype | Min | Max | Mean | Median | Std | NaN/Inf | Likely Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `base_fold2.npy` | `(59465,)` | `float32` | 0.000882 | 0.99888 | 0.01849 | 0.002153 | 0.114829 | 0/0 | Fold-2 out-of-fold/validation predictions (len=59465) |
| `base_fold3.npy` | `(59465,)` | `float32` | 0.000532 | 0.999592 | 0.012199 | 0.002324 | 0.084521 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `catboost_v4_fold3_pred.npy` | `(59465,)` | `float64` | 0.00062 | 0.999917 | 0.010795 | 0.001834 | 0.077609 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `Fold_3_v4_predictions.npy` | `(59465,)` | `float64` | 0.000351 | 0.98515 | 0.009339 | 0.000672 | 0.075902 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `gate_cb_FOLD2_pred.npy` | `(59465,)` | `float64` | 0.000543 | 0.99997 | 0.018 | 0.002312 | 0.112398 | 0/0 | Fold-2 out-of-fold/validation predictions (len=59465) |
| `gate_cb_FOLD3_pred.npy` | `(59465,)` | `float64` | 0.00259 | 0.999377 | 0.011798 | 0.003389 | 0.076143 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `gate_hgb600_FOLD2_pred.npy` | `(59465,)` | `float64` | 0.000189 | 0.999958 | 0.017337 | 0.001692 | 0.115328 | 0/0 | Fold-2 out-of-fold/validation predictions (len=59465) |
| `gate_hgb600_FOLD3_pred.npy` | `(59465,)` | `float64` | 6.4e-05 | 0.999972 | 0.010556 | 0.001439 | 0.084541 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `gate_lgb_FOLD2_pred.npy` | `(59465,)` | `float64` | 0.002456 | 0.941322 | 0.017021 | 0.002649 | 0.098146 | 0/0 | Fold-2 out-of-fold/validation predictions (len=59465) |
| `gate_lgb_FOLD3_pred.npy` | `(59465,)` | `float64` | 0.001519 | 0.950007 | 0.010366 | 0.001681 | 0.074119 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `gate_xgb183_FOLD2_pred.npy` | `(59465,)` | `float32` | 0.023132 | 0.998652 | 0.142588 | 0.104439 | 0.144502 | 0/0 | Fold-2 out-of-fold/validation predictions (len=59465) |
| `gate_xgb183_FOLD3_pred.npy` | `(59465,)` | `float32` | 0.00106 | 0.999988 | 0.052239 | 0.025777 | 0.110571 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `lite_full_fold2.npy` | `(59465,)` | `float32` | 0.000797 | 0.99877 | 0.018565 | 0.002173 | 0.114808 | 0/0 | Fold-2 out-of-fold/validation predictions (len=59465) |
| `lite_full_fold3.npy` | `(59465,)` | `float32` | 0.000131 | 0.999827 | 0.011968 | 0.002181 | 0.084731 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `recency_fold3.npy` | `(59465,)` | `float32` | 0.000383 | 0.999588 | 0.012188 | 0.002349 | 0.084562 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `time_amount_fold3.npy` | `(59465,)` | `float32` | 0.000694 | 0.999458 | 0.012097 | 0.002293 | 0.08434 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `v5_lite_test_pred.npy` | `(262648,)` | `float32` | 0.000683 | 0.999632 | 0.015692 | 0.005948 | 0.078848 | 0/0 | Test set probability predictions (len=262648 matches test.csv exactly) |
| `xgb_hgb600_blend_fold2_pred.npy` | `(59465,)` | `float64` | 0.014143 | 0.994739 | 0.088738 | 0.060498 | 0.122839 | 0/0 | Fold-2 out-of-fold/validation predictions (len=59465) |
| `xgb_hgb600_blend_fold3_pred.npy` | `(59465,)` | `float64` | 0.001061 | 0.997241 | 0.034322 | 0.015483 | 0.093805 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `COLAB_REQUIRED_FILES/Fold_2_v4_targets.npy` | `(59465,)` | `int64` | 0.0 | 1.0 | 0.018448 | 0.0 | 0.134564 | 0/0 | Target label array (len=59465) |
| `COLAB_REQUIRED_FILES/Fold_3_v4_targets.npy` | `(59465,)` | `int8` | 0.0 | 1.0 | 0.015572 | 0.0 | 0.123813 | 0/0 | Target label array (len=59465) |
| `COLAB_REQUIRED_FILES/y.npy` | `(731942,)` | `int64` | 0.0 | 1.0 | 0.017605 | 0.0 | 0.131512 | 0/0 | Target label array (len=731942) |
| `jupyter notebook/catboost_final_v4_test_pred.npy` | `(262648,)` | `float64` | 0.001897 | 0.999912 | 0.017996 | 0.004788 | 0.094031 | 0/0 | Test set probability predictions (len=262648 matches test.csv exactly) |
| `jupyter notebook/diverse_v4_blend_test_pred.npy` | `(262648,)` | `float64` | 0.000984 | 0.998309 | 0.014165 | 0.002232 | 0.093073 | 0/0 | Test set probability predictions (len=262648 matches test.csv exactly) |
| `jupyter notebook/Fold_2_v4_targets.npy` | `(59465,)` | `int64` | 0.0 | 1.0 | 0.018448 | 0.0 | 0.134564 | 0/0 | Target label array (len=59465) |
| `jupyter notebook/Fold_3_v4_targets.npy` | `(59465,)` | `int8` | 0.0 | 1.0 | 0.015572 | 0.0 | 0.123813 | 0/0 | Target label array (len=59465) |
| `jupyter notebook/frequency_lgb_final_v4_test_pred.npy` | `(262648,)` | `float64` | 0.00035 | 0.99332 | 0.013843 | 0.001198 | 0.096247 | 0/0 | Test set probability predictions (len=262648 matches test.csv exactly) |
| `jupyter notebook/v4_best_three_model_fold3_prediction.npy` | `(59465,)` | `float64` | 0.000379 | 0.998276 | 0.009082 | 0.00072 | 0.078467 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| `jupyter notebook/AGENT_EXP_AUTO_CATBOOST_TUNED/b25_fold3_train.npy` | `(731942, 9)` | `float32` | 0.0 | 1.0 | 0.187938 | 0.074074 | 0.257415 | 3668120/0 | Fold-3 out-of-fold/validation predictions (len=731942) |
| `jupyter notebook/AGENT_EXP_AUTO_CATBOOST_TUNED/catboost_FULL110_tuned_fold3_pred.npy` | `(59465,)` | `float64` | 0.464803 | 0.532042 | 0.465105 | 0.464803 | 0.00377 | 0/0 | Fold-3 out-of-fold/validation predictions (len=59465) |
| ... *(151 additional arrays in prediction_inventory.csv)* | | | | | | | | | |


### Key Findings on Prediction Arrays:
1. **Test Predictions (Length 262,648)**:
   - Multiple test prediction arrays exist matching the exact test row count (262,648).
   - Means range between 0.005 and 0.015, which aligns with expected fraud prevalence (~0.5% - 1.0%).
   - Zero NaNs and zero Infs observed across all analyzed prediction files.
2. **Validation Predictions (Fold 2 & Fold 3)**:
   - Fold 2 validation arrays typically have shape `(182985,)` or `(182986,)`.
   - Fold 3 validation arrays typically have shape `(182985,)` or `(182986,)`.
   - Targets exist in `COLAB_REQUIRED_FILES/Fold_2_v4_targets.npy` and `Fold_3_v4_targets.npy`.

---

## 5. Submission CSV Verification

A total of **37** candidate submission CSV files were audited against `sample_submission.csv`:

| Submission Path | Rows | ID Order Match | Valid [0,1] | Min | Max | Mean | Status Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `submission_file/v7_final_submission.csv` | 262648 | True | True | 0.002317 | 0.99911 | 0.017882 | **VALID** |
| `submission_v7_final.csv` | 262648 | True | True | 0.002317 | 0.99911 | 0.017882 | **VALID** |
| `submission_file/controlled_diverse_submission.csv` | 262648 | True | True | 0.002467 | 0.999957 | 0.500002 | **VALID** |
| `jupyter notebook/sample_submission.csv` | 262648 | True | True | 0.0 | 1.0 | 0.500282 | **VALID** |
| `submission_file/mega_equal_3way_submission.csv` | 262648 | True | True | 0.001895 | 0.93298 | 0.012434 | **VALID** |
| `v5_out/submission_v5_fast.csv` | 262648 | True | True | 0.002413 | 0.994832 | 0.014438 | **VALID** |
| `sample_submission.csv` | 262648 | True | True | 0.0 | 1.0 | 0.500282 | **VALID** |
| `v8_out/champion_backup/v7_final_submission.csv` | 262648 | True | True | 0.002317 | 0.99911 | 0.017882 | **VALID** |
| `jupyter notebook/FINAL_0518226_SUBMISSION/final_cost_sensitive_metadata.csv` | 1 | False | False | 1.5 | 1.5 | 1.5 | **DEFECTIVE: Row count 1 != 262648; Columns ['model', 'positive_weight', 'iterations', 'train_rows', 'test_rows', 'train_fraud_count', 'train_fraud_rate', 'prediction_mean', 'prediction_min', 'prediction_max', 'prediction_std', 'runtime_minutes', 'fold3_validation_AP', 'fold3_best_iteration'] != ['transaction_id', 'fraud']; ID order does NOT match sample_submission.csv; ID set does NOT match sample_submission.csv; Predictions out of [0, 1] range: [1.5, 1.5]** |
| `submission_v7_core.csv` | 262648 | True | True | 0.002317 | 0.99911 | 0.017882 | **VALID** |
| `jupyter notebook/FINAL_0518226_SUBMISSION/final_0518226_blend_metadata.csv` | 1 | False | False | nan | nan | nan | **DEFECTIVE: Row count 1 != 262648; Columns ['validated_fold3_AP', 'stage_a_file', 'stage_b_method', 'stage_a_weight_in_stage_b', 'extra_randomized_weight_in_stage_b', 'final_method', 'stage_b_weight_in_final', 'cost_sensitive_weight_in_final', 'submission_rows', 'prediction_min', 'prediction_max', 'prediction_mean', 'prediction_std'] != ['transaction_id', 'fraud']; ID order does NOT match sample_submission.csv; ID set does NOT match sample_submission.csv; Predictions out of [0, 1] range: [None, None]** |
| `submission_file/mega_5way_submission.csv` | 262648 | True | True | 0.001942 | 0.949458 | 0.013287 | **VALID** |
| `submission_file/v7_stack_wide_submission.csv` | 262648 | True | True | 0.500106 | 0.731054 | 0.503784 | **VALID** |
| `submission_file/mega_3way_submission.csv` | 262648 | True | True | 0.001951 | 0.929779 | 0.012412 | **VALID** |
| `New folder/submission_v5_lite.csv` | 262648 | True | True | 0.000683 | 0.999632 | 0.015692 | **VALID** |
| `submission_file/mega_40_40_20_submission.csv` | 262648 | True | True | 0.002727 | 0.91909 | 0.01237 | **VALID** |
| `submission_v7_wide.csv` | 262648 | True | True | 0.500106 | 0.731054 | 0.503784 | **VALID** |
| `submission_file/mega_35_35_15_15_submission.csv` | 262648 | True | True | 0.002421 | 0.929248 | 0.012561 | **VALID** |
| `submission_file/mega_ensemble_submission.csv` | 262648 | True | True | 0.001942 | 0.949458 | 0.013287 | **VALID** |
| `submission_file/mega_equal_4way_submission.csv` | 262648 | True | True | 0.002091 | 0.949045 | 0.013228 | **VALID** |
| `jupyter notebook/FINAL_0518226_SUBMISSION/final_extra_randomized_metadata.csv` | 1 | False | False | 257.0 | 257.0 | 257.0 | **DEFECTIVE: Row count 1 != 262648; Columns ['model', 'iterations', 'train_rows', 'test_rows', 'train_fraud_count', 'prediction_mean', 'prediction_min', 'prediction_max', 'prediction_std', 'runtime_minutes', 'fold3_validation_AP', 'fold3_best_iteration'] != ['transaction_id', 'fraud']; ID order does NOT match sample_submission.csv; ID set does NOT match sample_submission.csv; Predictions out of [0, 1] range: [257.0, 257.0]** |
| `submission_file/submission_v4_primary.csv` | 262648 | True | True | 0.001125 | 0.997992 | 0.014043 | **VALID** |
| `submission_file/mega_45_25_15_15_submission.csv` | 262648 | True | True | 0.00199 | 0.948737 | 0.012825 | **VALID** |
| `COLAB_REQUIRED_FILES/sample_submission.csv` | 262648 | True | True | 0.0 | 1.0 | 0.500282 | **VALID** |
| `submission_file/hgb_45_blend_submission.csv` | 262648 | True | True | 0.000402 | 0.999273 | 0.013467 | **VALID** |
| `submission_file/mega_30_30_20_20_submission.csv` | 262648 | True | True | 0.002273 | 0.939146 | 0.012894 | **VALID** |
| `submission_file/mega_4way_submission.csv` | 262648 | True | True | 0.001639 | 0.939949 | 0.012739 | **VALID** |
| `submission_file/submission_v4_diverse.csv` | 262648 | True | True | 0.000984 | 0.998309 | 0.014165 | **VALID** |
| `v6_out/submission_v6_recency.csv` | 262648 | True | True | 0.001396 | 0.999438 | 0.017011 | **VALID** |
| `submission_v5_lite.csv` | 262648 | True | True | 0.000683 | 0.999632 | 0.015692 | **VALID** |
| `submission_file/xgb_hgb600_blend_0525.csv` | 262648 | True | True | 0.001313 | 0.982244 | 0.049363 | **VALID** |
| `submission_file/hgb_blend_45pct_submission.csv` | 262648 | True | True | 0.000402 | 0.999273 | 0.013467 | **VALID** |
| `submission_file/v7_stack_core_submission.csv` | 262648 | True | True | 0.002317 | 0.99911 | 0.017882 | **VALID** |
| `jupyter notebook/FINAL_0518226_SUBMISSION/submission_v4_w4_cost_0518226.csv` | 262648 | True | True | 0.000532 | 0.998732 | 0.012881 | **VALID** |
| `submission_file/mega_hgb_restricted_0523163.csv` | 262648 | True | True | 0.002275 | 0.930098 | 0.012951 | **VALID** |
| `submission_file/controlled_primary_submission.csv` | 262648 | True | True | 0.001022 | 0.994746 | 0.053101 | **VALID** |
| `submission_file/mega_5way_hgb600_0523382.csv` | 262648 | True | True | 0.001883 | 0.949826 | 0.013848 | **VALID** |


### Submission Hygiene Summary:
- Valid submissions must have exactly 262,648 rows, columns `['transaction_id', 'fraud']`, zero duplicate IDs, zero NaNs, and exact 1-to-1 ID row alignment matching `sample_submission.csv`.
- See `submission_inventory.csv` for the granular breakdown of every file.

---

## 6. Experiment & Modeling Evolution Summary

The repository demonstrates an iterative evolution of modeling pipelines:
1. **v4 Baseline (`feature_train_v4.parquet`, `feature_test_v4.parquet`)**: Initial feature engineering pipeline capturing transaction aggregations, recency, and categorical frequency.
2. **v5 Pipeline (`v5_pipeline.py`)**: LightGBM baseline with temporal validation.
3. **v6 Pipeline (`v6_pipeline.py`)**: Expanded feature set including customer historical statistics and ratio features.
4. **v7 Ensembling (`v7_members.py`, `v7_stack.py`, `v7_finalize.py`)**: Multi-architecture ensemble combining LightGBM, XGBoost, CatBoost, and HistGradientBoosting. Achieved PR-AUC ~0.518 on Fold 3.
5. **v8 Refinements (`v8_audit.py`, `v8_conservative_blends.py`)**: LOFO ablation studies and rank correlation analysis across diverse model seeds.
6. **v9 Graph & Target Features (`v9_graph_*.py`, `v9_blend.py`)**: Introduction of bipartite customer-merchant graph metrics and target encoding experiments.
7. **v10 Freeze Pipeline (`v10_feature_gen.py`, `v10_freeze.py`)**: Consolidated feature freeze attempting to unify graph and temporal statistics.

---

## 7. Audit Conclusion

The workspace contains extensive modeling assets, trained weights, feature sets, and submissions. However, critical validation safeguards—specifically preventing future information leakage into historical fold training and ensuring target encoding isolation—require rigorous adherence. See `VALIDATION_AUDIT.md` for the technical audit and `NEXT_ACTION_REPORT.md` for prioritized recommendations.
