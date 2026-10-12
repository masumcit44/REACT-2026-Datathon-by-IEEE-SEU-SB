# PROJECT MASTER DOSSIER — REACT 2026 Datathon

> **Generated:** 2026-09-14  
> **Metric:** Average Precision (PR-AUC) on fraud detection  
> **Status:** Active competition — best public LB = 0.52642, best fold-3 blend = 0.527088 (unsubmitted)

---

## PROJECT STATUS SNAPSHOT

| Item | Value |
|------|-------|
| **Current Best Public LB** | **0.52642** (`nested_0518226` submission) |
| **Current Best Local (fold-3)** | **0.527088** (`xgb_hgb600_blend` — XGB183×0.57 + HGB600×0.34 + mega_lgb×0.09 + mega_cb×0.002) |
| **Current Best Individual Model** | HGB183 (sklearn HistGB, 600 iter, 183 feats) — fold-3 AP = **0.524681** |
| **Champion Reference** | `best_extended_cost_global` — LGB 101 feats, fold-3 AP = 0.518226 |
| **Public LB Gap to 0.58 Target** | 0.05358 (10.3% relative) |

## TOP 10 EXPERIMENTS (ranked by fold-3 AP)

| Rank | Experiment | Model | Features | Fold-3 AP | Gain vs Champion | Status |
|------|-----------|-------|----------|-----------|-----------------|--------|
| 1 | xgb_hgb600_blend | Blend (XGB+HGB+LGB+CB) | Mixed | **0.527088** | +0.008862 | SUBMITREADY, fold-2 gate pending |
| 2 | hgb183_fulldata | HistGradientBoosting | 183 | **0.524681** | +0.006455 | DONE (preds on Colab W3) |
| 3 | xgb183_fulldata | XGBoost (hist) | 215 | **0.523993** | +0.005767 | DONE |
| 4 | rapidfire_best4way | 4-way blend | Mixed | 0.523483 | +0.005257 | DONE |
| 5 | rapidfire_best5way | 5-way blend | Mixed | 0.523615 | +0.005389 | DONE |
| 6 | mega_5way_hgb600 | 5-way blend | Mixed | 0.523382 | +0.005156 | SUBMITREADY |
| 7 | restricted_set_blend | 4-way restricted | Mixed | 0.523163 | +0.004937 | SUBMITTED |
| 8 | g_hgbt_blend_w45 | G+HGBt@45% | 101+124 | 0.521640 | +0.003414 | Cycle-2 champion |
| 9 | g_hgb600_blend | G+HGB600@50% | 101+124 | 0.521354 | +0.003128 | PASS |
| 10 | best_3way_g_hgbt_acc | 3-way G/HGBt/ACC | 101+124+110 | 0.521920 | +0.003694 | PASS |

## TOP 10 UNTRIED IDEAS

| Rank | Idea | Expected Impact | Effort | Status |
|------|------|----------------|--------|--------|
| 1 | HGB183 test pred → 5-way blend (HGB183+XGB183+HGB600+mega_lgb+mega_cb) | HIGH (best individual is 0.524681) | LOW (download preds) | BLOCKED: preds on retiring Colab W3 |
| 2 | Graph-feature LGB full-data (22 graph feats, screen passed +0.00118) | MEDIUM | DONE (running on W2) | RUNNING |
| 3 | CatBoost-183 full-data (1500 it, lr.03, depth 10) | MEDIUM | DONE (running on W1) | RUNNING |
| 4 | Time-decay LGB (half-life weighting, aggressive_pipeline.py) | MEDIUM | MEDIUM | Script exists, unvalidated |
| 5 | Adversarial validation + IsolationForest features | MEDIUM | MEDIUM | Script exists (AGENT_EXP_AGGRESSIVE_V2) |
| 6 | Stacking head (LR on OOF logits of top-5) | MEDIUM | LOW | Requires OOF predictions from 5 models |
| 7 | Multi-fold harness (folds 1-4 for robust gating) | MEDIUM | LOW | Script exists, not yet run |
| 8 | Full-7 feature block with HGB (failed cheap screen at +0.001) | LOW | LOW | Failed gate |
| 9 | CatBoost multi-seed ensemble (5 seeds × 110 features) | LOW | LOW | Script exists |
| 10 | OOF target encoding + LGB (time-expanded) | LOW | MEDIUM | Script exists |

---

# 1. DATASET OVERVIEW

## Raw Competition Data

| Property | Value |
|----------|-------|
| **Train rows** | **731,942** |
| **Test rows** | **262,648** |
| **Total** | **994,590** |
| **Target column** | `fraud` (binary, only in train) |
| **Fraud rate (train)** | **0.01761 (1.761%)** |
| **Train columns** | 13 (12 shared + `fraud`) |
| **Test columns** | 12 |

## Column List

| Column | Dtype | Unique (train) | Missing % | Description |
|--------|-------|-----------------|-----------|-------------|
| transaction_id | str | 731,942 | 0% | Unique transaction identifier |
| customer_id | str | 38,689 | 0% | Customer who initiated transaction |
| timestamp | str → datetime64 | 711,498 | 0% | Transaction time (2026-01-01 00:00:43 to 2026-01-03 18:49:02) |
| amount_bdt | float64 | — | 0% | Transaction amount in BDT |
| merchant_id | str | 4,178 | 0% | Merchant receiving payment |
| merchant_category | str | 12 | 0.60% | Merchant category |
| device_id | str | 19,089 | 0% | Device used for transaction |
| device_type | str | 5 | 0.40% | Type of device |
| location | str | 45 | 0.51% | Transaction location |
| payment_method | str | 5 | 0% | Payment method used |
| transaction_type | str | 5 | 0% | Type of transaction |
| account_age_days | int64 | — | 0% | Age of account in days |
| fraud | int64 | 2 (0/1) | 0% | **Target** (1.761% positive) |

## Date Range

| Split | Date Range |
|-------|-----------|
| Train (full) | 2026-01-01 00:00:43 → 2026-01-03 18:49:02 |
| Test | After train (chronologically sorted) |

## Cardinality Summary

| Category | Count | Note |
|----------|-------|------|
| transaction_id | 731,942 | Unique per row |
| customer_id | 38,689 | ~19 txns/customer avg |
| merchant_id | 4,178 | ~175 txns/merchant avg |
| device_id | 19,089 | Shared devices present |
| merchant_category | 12 | Low cardinality |
| device_type | 5 | Low cardinality |
| location | 45 | Medium cardinality |
| payment_method | 5 | Low cardinality |
| transaction_type | 5 | Low cardinality |

## Missing Values (train)

| Column | Missing % |
|--------|-----------|
| merchant_category | 0.60% |
| location | 0.51% |
| device_type | 0.40% |

All other columns: 0% missing.

---

# 2. VALIDATION DESIGN

## Fold Strategy

**Temporal three-fold split** — data is chronologically ordered by timestamp. The two most recent windows serve as independent validation folds; the rest is training data.

```
fold1 = [0     : 613012)    — 613,012 rows (training pool, 83.7%)
fold2 = [613012: 672477)    —  59,465 rows (secondary gate, 8.1%)
fold3 = [672477: 731942)    —  59,465 rows (primary eval, 8.1%)
```

## Fold Sizes and Fraud Rates

| Fold | Rows | Fraud Rate | Purpose |
|------|------|-----------|---------|
| fold1 (train) | 613,012 | 1.772% | Training data for all models |
| fold2 (val) | 59,465 | 1.845% | Secondary stability gate |
| fold3 (eval) | 59,465 | 1.557% | Primary model comparison |
| **Total** | **731,942** | **1.761%** | — |

## Why This Split Was Chosen

1. **Temporal holdout:** Test set is chronologically after training. Folds simulate this by using the most recent training data as validation.
2. **Fold-3 as primary gate:** All model comparisons and champion decisions use fold-3 AP. This is the "main score".
3. **Fold-2 as secondary stability gate:** A model/blend must not drop more than -0.001 AP on fold-2 vs the champion fold-2 baseline. This guards against overfitting to the fold-3 temporal pattern.
4. **Three-fold structure:** Train on fold1 → validate on fold2 (secondary) → validate on fold3 (primary) → full-data fit for test predictions.

## Fold-2 Gate Rule

```python
gate_pass = (fold2_AP >= fold3_AP - 0.001)
```

Evidence: `fold2_gate_check.py` line 9: *"gate: fold-2 AP >= fold3_AP - 0.001"*

## Alternate Temporal Fold-2

A date-based fold-2 is also defined in `rebuild_fold2_temporal.py`:
- Train: timestamp < 2026-06-16 (614,022 rows)
- Val: timestamp >= 2026-06-16 AND < 2026-07-01 (58,455 rows)

This is a subset of the index-based fold-2 (58,455 of the 59,465 rows).

---

# 3. COMPLETE FEATURE INVENTORY

## Summary

| Feature Set | Count | Used By |
|-------------|-------|---------|
| V4 Core | 101 | All models (baseline) |
| NA Indicators | 23 | XGB183, HGB variants |
| Mega New Features | 82 | Mega LGB/CB/HGB, Rapidfire |
| B25 Acceleration | 9 | Acceleration experiments |
| F7 Custom | 7 | FINAL7 experiment (failed gate) |
| Graph Features | 22 | Graph LGB (running on W2) |
| **Maximum Total** | **215** | XGB183 (101+82+23+9) |

## V4 Core Features (101)

### BASE — Raw Identifiers + Amount (10)

| # | Feature | Type |
|---|---------|------|
| 1 | customer_id | categorical |
| 2 | amount_bdt | float |
| 3 | merchant_id | categorical |
| 4 | merchant_category | categorical |
| 5 | device_id | categorical |
| 6 | device_type | categorical |
| 7 | location | categorical |
| 8 | payment_method | categorical |
| 9 | transaction_type | categorical |
| 10 | account_age_days | int |

### TIME (8)

| # | Feature | Type |
|---|---------|------|
| 11 | hour | int |
| 12 | day_of_week | int |
| 13 | day_of_month | int |
| 14 | month | int |
| 15 | is_weekend | binary |
| 16 | is_night | binary |
| 17 | is_business_hour | binary |
| 18 | amount_log | float |

### CUSTOMER VELOCITY (28)

| # | Feature | Type |
|---|---------|------|
| 19 | customer_previous_count | int |
| 20 | customer_time_since_previous_min | float |
| 21 | customer_previous_amount | float |
| 22 | customer_previous_amount_sum | float |
| 23 | customer_previous_amount_mean | float |
| 24 | customer_amount_difference | float |
| 25 | customer_amount_ratio | float |
| 26 | customer_previous_amount_mean_v2 | float |
| 27 | customer_previous_amount_std | float |
| 28 | customer_amount_zscore | float |
| 29 | customer_amount_abs_zscore | float |
| 30 | customer_transactions_previous_1h | int |
| 31 | customer_transactions_previous_6h | int |
| 32 | customer_transactions_previous_24h | int |
| 33 | customer_transactions_previous_7d | int |
| 34 | customer_burst_10min | int |
| 35 | customer_burst_30min | int |
| 36 | customer_burst_60min | int |
| 37 | customer_high_velocity_1h | binary |
| 38 | customer_high_velocity_24h | binary |
| 39 | customer_amount_sum_previous_1h | float |
| 40 | customer_amount_mean_previous_1h | float |
| 41 | customer_amount_sum_previous_6h | float |
| 42 | customer_amount_mean_previous_6h | float |
| 43 | customer_amount_sum_previous_24h | float |
| 44 | customer_amount_mean_previous_24h | float |
| 45 | customer_amount_ratio_recent_1h | float |
| 46 | customer_amount_ratio_recent_6h | float |
| 47 | customer_amount_ratio_recent_24h | float |
| 48 | customer_amount_total_24h_with_current | float |

### DEVICE VELOCITY (18)

| # | Feature | Type |
|---|---------|------|
| 49 | device_previous_count | int |
| 50 | device_previous_unique_customers | int |
| 51 | device_is_shared | binary |
| 52 | device_highly_shared | binary |
| 53 | device_previous_customer | categorical |
| 54 | device_time_since_previous_min | float |
| 55 | device_customer_changed | binary |
| 56 | device_transactions_previous_10min | int |
| 57 | device_transactions_previous_1h | int |
| 58 | device_transactions_previous_6h | int |
| 59 | device_transactions_previous_24h | int |
| 60 | device_burst_10min | int |
| 61 | device_burst_1h | int |
| 62 | device_burst_6h | int |
| 63 | device_recent_1h_share | float |
| 64 | device_recent_24h_share | float |

### MERCHANT VELOCITY (16)

| # | Feature | Type |
|---|---------|------|
| 65 | merchant_previous_count | int |
| 66 | merchant_previous_unique_customers | int |
| 67 | merchant_previous_customer | categorical |
| 68 | merchant_time_since_previous_min | float |
| 69 | merchant_customer_changed | binary |
| 70 | merchant_transactions_previous_10min | int |
| 71 | merchant_transactions_previous_1h | int |
| 72 | merchant_transactions_previous_6h | int |
| 73 | merchant_transactions_previous_24h | int |
| 74 | merchant_recent_1h_share | float |
| 75 | merchant_recent_24h_share | float |
| 76 | merchant_burst_10min | int |
| 77 | merchant_burst_1h | int |
| 78 | merchant_burst_6h | int |

### LOCATION / CROSS-ENTITY COUNTS (4)

| # | Feature | Type |
|---|---------|------|
| 79 | location_previous_count | int |
| 80 | customer_device_previous_count | int |
| 81 | customer_merchant_previous_count | int |
| 82 | customer_location_previous_count | int |

### RELATIONSHIP / TRANSITION (11)

| # | Feature | Type |
|---|---------|------|
| 83 | is_new_device_for_customer | binary |
| 84 | is_new_merchant_for_customer | binary |
| 85 | is_new_location_for_customer | binary |
| 86 | customer_previous_device | categorical |
| 87 | customer_previous_merchant | categorical |
| 88 | customer_previous_location | categorical |
| 89 | customer_previous_hour | int |
| 90 | device_changed_from_previous | binary |
| 91 | merchant_changed_from_previous | binary |
| 92 | location_changed_from_previous | binary |
| 93 | hour_difference_from_previous | float |

### RISK / CROSS-ENTITY BURST (8)

| # | Feature | Type |
|---|---------|------|
| 94 | rapid_customer_switch_on_device | binary |
| 95 | very_rapid_customer_switch_on_device | binary |
| 96 | new_device_and_customer_burst | binary |
| 97 | new_location_and_customer_burst | binary |
| 98 | shared_device_recent_burst | binary |
| 99 | large_amount_and_new_device | binary |
| 100 | large_amount_and_new_location | binary |
| 101 | large_amount_and_customer_burst | binary |

## NA Indicator Features (23)

Added at training time for columns with missing values. Binary flags: `feature__is_na`.

Applied to: `customer_time_since_previous_min`, `customer_previous_amount`, `customer_previous_amount_mean`, `customer_amount_difference`, `customer_amount_ratio`, `customer_previous_hour`, `hour_difference_from_previous`, `customer_previous_amount_mean_v2`, `customer_previous_amount_std`, `customer_amount_zscore`, `customer_amount_abs_zscore`, `device_time_since_previous_min`, `device_recent_1h_share`, `device_recent_24h_share`, `merchant_time_since_previous_min`, `customer_amount_mean_previous_1h`, `customer_amount_mean_previous_6h`, `customer_amount_mean_previous_24h`, `customer_amount_ratio_recent_1h`, `customer_amount_ratio_recent_6h`, `customer_amount_ratio_recent_24h`, `merchant_recent_1h_share`, `merchant_recent_24h_share`.

**Total with NA indicators: 124 features**

## Mega Features (82 new)

Built in `AGENT_EXP_MEGA_V2/build_features_v2.py`. Prefixed `n_` in parquet.

| Group | Count | Examples |
|-------|-------|---------|
| Base (hour/day/amount) | 4 | hour, day_of_week, is_weekend, is_night |
| Amount engineering | 8 | amount, amount_log, amount_sqrt, amount_round100/500, account_age_days/log, is_new_account |
| Graph co-occurrence | 13 | cust_cumcount, dev_cumcount, merch_cumcount, cust_merch_count, cust_dev_count, dev_merch_count, is_first_cust_merch/dev/merch, cust_count_log, dev_count_log, merch_count_log |
| Amount rank | 8 | amt_pctile_merch/cust/dev/merch_cat/dev_type, amt_vs_cust_mean, amt_ratio_cust_mean |
| Temporal velocity | 10 | hours_since_cust/dev/merch, days_since_cust/cust_merch, amt_change_cust, amt_ratio_cust, cust/dev/merch_tx_in_24h/1h, is_dormant_reactivation |
| OOF target encoding | 24 | oof_{entity}, oof_raw_{entity}, oof_count_{entity} for 8 entities (merchant_id, device_id, customer_id, merchant_category, device_type, location, payment_method, transaction_type) |
| Categorical codes | 5 | merchant_category_code, device_type_code, location_code, payment_method_code, transaction_type_code |
| Interactions | 7 | amt_x_cust_count, amt_x_merch_count, cust_count_x_dev_count, merch_distinct_x_amt, dev_distinct_x_amt, oof_merch_x_amt, oof_cust_x_amt |
| Cumulative extras | 3 | dev_distinct_cust_alltime, merchant_alltime_count, amt_pctile_in_merch/cust |

**Total mega: 82 new features**

## B25 Acceleration Features (9)

Velocity ratio features capturing burst behavior at multiple time windows.

| Feature | Description |
|---------|-------------|
| b25_customer_1h_to_24h_ratio | Customer txns in 1h / 24h |
| b25_customer_6h_to_24h_ratio | Customer txns in 6h / 24h |
| b25_customer_1h_to_6h_ratio | Customer txns in 1h / 6h |
| b25_device_1h_to_24h_ratio | Device txns in 1h / 24h |
| b25_device_6h_to_24h_ratio | Device txns in 6h / 24h |
| b25_device_1h_to_6h_ratio | Device txns in 1h / 6h |
| b25_merchant_1h_to_24h_ratio | Merchant txns in 1h / 24h |
| b25_merchant_6h_to_24h_ratio | Merchant txns in 6h / 24h |
| b25_merchant_1h_to_6h_ratio | Merchant txns in 1h / 6h |

## F7 Custom Features (7)

Built in `AGENT_EXP_AUTO_FINAL7/build_f7.py`. Failed cheap screen (+0.001 < +0.002 gate).

| Feature | Description |
|---------|-------------|
| dev_dtype_cust | Device-type per customer |
| dev_loc_cust | Device-location per customer |
| num_dev_axes_cust | Number of device axes per customer |
| is_new_in_last_30d | New account in last 30 days |
| dev_new_in_last_30d | New device in last 30 days |
| amt_pct_cust | Amount percentile within customer |
| mch_growth_7d | Merchant growth 7-day |

## Graph Features (22)

Built on Colab W2. Passed cheap screen: base LGB 0.527920 → +graph 0.529101, gain +0.001180. Full-data LGB running.

## Categorical Features (13 in model_features_v4.pkl)

customer_id, merchant_id, merchant_category, device_id, device_type, location, payment_method, transaction_type, customer_previous_device, customer_previous_merchant, customer_previous_location, device_previous_customer, merchant_previous_customer.

---

# 4. EXPERIMENT TRACKER SUMMARY

| Experiment ID | Status | Model | Features | Fold-3 AP | Gain | Notes | Owner |
|--------------|--------|-------|----------|-----------|------|-------|-------|
| champion | DONE-LOCKED | LGB (best_extended_cost_global) | 101 | 0.518226 | 0.000 | CURRENT CHAMPION (fold-3) | LOCAL |
| champion_public | DONE-LOCKED | nested_0518226 | 101 | 0.526420 | +0.008 | Public leaderboard after submission | LOCAL |
| hgb600 | DONE | HGB | 124 | 0.519930 | +0.002 | BEST INDIVIDUAL (original) | LOCAL |
| hgb_tuned | DONE | HGB | 124 | 0.518915 | +0.001 | Second HGB variant | LOCAL |
| mega_lgb | DONE | LGB | 183 | 0.515853 | -0.002 | Has test pred | COLAB2 |
| mega_cb | DONE | CB | 183 | 0.505725 | -0.013 | Has test pred | COLAB1 |
| channel_transitions | DONE | LGB | 102 | 0.506201 | -0.012 | REJECTED | COLAB2 |
| entity_age_density | DONE | LGB | 102 | 0.506878 | -0.011 | REJECTED | COLAB2 |
| population_percentile | DONE | LGB | 102 | 0.506527 | -0.012 | REJECTED | COLAB2 |
| device_merchant_novelty | DONE | LGB | 102 | 0.506000 | -0.012 | REJECTED | COLAB2 |
| top3_acceleration | DONE | LGB | 106 | 0.512409 | -0.006 | REJECTED (full-9 better) | COLAB3 |
| top7_acceleration | DONE | LGB | 110 | 0.512186 | -0.006 | REJECTED (full-9 better) | COLAB2 |
| acceleration_cheap | DONE | LGB | 110 | 0.510953 | -0.007 | cheap PASS, full-9 REJECTED | COLAB2 |
| acceleration_tuned | DONE | LGB | 110 | 0.512894 | -0.005 | ACCEPTED member | LOCAL |
| w4_extra_random_lgb | DONE | LGB | 101 | 0.511359 | -0.007 | ACCEPTED member | LOCAL |
| w4_global_blend | DONE | blend | — | 0.516533 | -0.002 | ACCEPTED member | NONE |
| cost_sensitive_lgb | DONE | LGB | 101 | 0.512163 | -0.006 | ACCEPTED member | LOCAL |
| cost_global_blend | DONE | blend | — | 0.518194 | -0.000 | ACCEPTED | LOCAL |
| goss_lgb | DONE | LGB | goss | 0.512844 | -0.005 | REJECTED | LOCAL |
| altseed_lgb | DONE | LGB | 183 | 0.512202 | -0.006 | RAPIDFIRE | LOCAL |
| deep_imp_lgb | DONE | LGB | 183 | 0.509486 | -0.009 | RAPIDFIRE | LOCAL |
| patience_lgb | DONE | LGB | 183 | 0.500732 | -0.017 | RAPIDFIRE | LOCAL |
| extreme_lgb | DONE | LGB | 183 | 0.505470 | -0.013 | RAPIDFIRE | LOCAL |
| g_hgb600_blend | DONE-PASS | blend | — | 0.521354 | +0.003 | 2-way blend PASS gate | LOCAL |
| g_hgbt_blend_w45 | DONE-RECOMMENDED | blend | — | 0.521640 | +0.003 | RECOMMENDED NEW CHAMPION | LOCAL |
| best_3way_g_hgbt_acc | DONE-PASS | blend | 50/40/10 | 0.521920 | +0.004 | BEST FLAG blend | LOCAL |
| restricted_set_blend | DONE-SUBMITREADY | blend | champ05/hgbt45/megalgb35/megacb15 | 0.523163 | +0.005 | SUBMITTED TO KAGGLE | LOCAL |
| mega_5way_blend | DONE | lgb+cb+hgb | — | 0.523419 | — | saved preds | LOCAL |
| rapidfire_best4way | DONE | lgb+hgb | — | 0.523483 | — | 4way equal - TOP BLENDS | LOCAL |
| rapidfire_best5way | DONE | lgb+hgb | — | 0.523615 | — | 5way | LOCAL |
| hgb600_fulldata | DONE | HGB | 124 | 0.520187 | +0.002 | FULL-DATA test pred | LOCAL |
| xgb_hgb600_blend | DONE-SUBMITREADY | blend | XGB183.57/HGB600.34/LGB.09/CB.002 | 0.527088 | +0.009 | BLEND fold-3, fold-2 gate pending | LOCAL |
| mega_5way_hgb600 | DONE-SUBMITREADY | blend | champ05/hgb600.25/hgbt30/megalgb.25/megacb.15 | 0.523382 | +0.005 | 5-way with real hgb600 test pred | LOCAL |
| cb183_fulldata | RUNNING | CB | 183 | 0 | 0 | CatBoost-183 full-data on W1 | COL1 |
| hgb183_fulldata | DONE | HGB | 183 | 0.524681 | +0.006 | BEST INDIVIDUAL - preds on Colab W3 | COL3 |
| xgb183_fulldata | DONE | XGB | 215 (183+NA) | 0.523993 | +0.006 | BEST XGB - preds on W4 | LOCAL |
| graph_features_v2 | DONE | feat | raw | 0.529101 | +0.011 | GRAPH SCREEN: base=0.527920 +graph=0.529101 | COL2 |
| graph_fulldata_lgb | RUNNING | LGB | 183+22 | 0 | 0 | LGB on combined+graph full-data on W2 | COL2 |

---

# 5. BEST MODELS

## Individual Models (ranked by fold-3 AP)

| Rank | Model | Algorithm | Features | Fold-3 AP | Params | Test Pred | Verified |
|------|-------|-----------|----------|-----------|--------|-----------|----------|
| 1 | HGB183 | HistGradientBoosting | 183 | **0.524681** | max_iter=600, lr=0.05, max_leaf=63, min_samples=100, l2=1.0 | YES (Colab W3, NOT downloaded) | YES |
| 2 | XGB183 | XGBoost (hist) | 215 | **0.523993** | depth=11, lr=0.05, subsample=0.8, colsample=0.7, spw=55.8, best_iter=408 | YES (AGENT_EXP_XGB) | YES |
| 3 | HGB600 full-data | HistGradientBoosting | 124 | **0.520187** | max_iter=600, lr=0.05, max_leaf=63, min_samples=100, l2=1.0, max_bins=255 | YES (AGENT_EXP_AUTO_HGB) | YES |
| 4 | HGB600 repro | HistGradientBoosting | 124 | **0.519930** | max_iter=600, lr=0.05, max_leaf=63, min_samples=100, l2=1.0 | NO | YES |
| 5 | HGB tuned | HistGradientBoosting | 124 | **0.518915** | max_iter=1000, lr=0.03, max_leaf=127, min_samples=60, l2=5.0 | YES (fulldata) | YES |
| 6 | Champion LGB | LightGBM | 101 | **0.518226** | leaves=63, lr=0.05, n_est=600, min_child=100, alpha=0.5, lambda=5.0 | YES | YES |
| 7 | Mega LGB | LightGBM | 183 | **0.515853** | leaves=127, lr=0.025, n_est=2000 | YES | YES |
| 8 | Acceleration-tuned LGB | LightGBM | 110 | **0.512894** | Tuned b25 | YES | YES |
| 9 | GOSS LGB | LightGBM | 183 | **0.512844** | GOSS boosting | — | rapidfire |
| 10 | Altseed LGB | LightGBM | 183 | **0.512202** | seed=9999 | — | rapidfire |
| 11 | Cost-sensitive LGB | LightGBM | 101 | **0.512163** | positive_weight=1.5 | YES | YES |
| 12 | Extra-randomized LGB | LightGBM | 101 | **0.511359** | Randomized params | YES | YES |
| 13 | Mega CatBoost | CatBoost | 183 | **0.505725** | iterations=1500, lr=0.05, depth=8 | YES | YES |

## Blend Models (ranked by fold-3 AP)

| Rank | Recipe | Fold-3 AP | Gain | Status |
|------|--------|-----------|------|--------|
| 1 | XGB183×0.57 + HGB600×0.34 + mega_lgb×0.09 + mega_cb×0.002 | **0.527088** | +0.0089 | SUBMITREADY (fold-2 gate pending) |
| 2 | 5-way rapidfire (champ×.05/hgb600×.25/hgbt×.30/megalgb×.25/megacb×.15) | **0.523615** | +0.0054 | All test preds exist |
| 3 | mega_5way_hgb600 (real hgb600 pred) | **0.523382** | +0.0052 | SUBMITREADY |
| 4 | Restricted champ×05/hgbt×45/megalgb×35/megacb×15 | **0.523163** | +0.0049 | SUBMITTED |
| 5 | 4-way equal (hgb600+hgbt+megalgb+megacb) | **0.523483** | +0.0053 | Equal weights |
| 6 | 3-way G/HGBt/ACC 50/40/10 | **0.521920** | +0.0037 | PASS global gate |
| 7 | G + HGBt×45% | **0.521640** | +0.0034 | Cycle-2 champion |
| 8 | G + HGB600×50% | **0.521354** | +0.0031 | PASS |

---

# 6. BLENDING HISTORY

| # | Blend Members | Weights | Fold-3 AP | Fold-2 AP | Public LB | Verdict |
|---|---------------|---------|-----------|-----------|-----------|---------|
| 1 | XGB183 + HGB600 + mega_lgb + mega_cb | 57/34/9/0.2 | 0.527088 | PENDING | — | SUBMITREADY |
| 2 | Champ + HGB600 + HGBt + mega_lgb + mega_cb | 5/25/30/25/15 | 0.523615 | — | — | DONE |
| 3 | Champ + HGB600 + HGBt + mega_lgb + mega_cb | 5/25/30/25/15 | 0.523382 | — | — | SUBMITREADY |
| 4 | Champ + HGBt + mega_lgb + mega_cb | 5/45/35/15 | 0.523163 | — | 0.52642 | SUBMITTED |
| 5 | HGB600 + HGBt + mega_lgb + mega_cb | 25/25/25/25 | 0.523483 | — | — | EQUAL |
| 6 | G + HGBt + Accel | 50/40/10 | 0.521920 | — | — | PASS |
| 7 | G + HGBt | 55/45 | 0.521640 | — | — | CYCLE-2 CHAMP |
| 8 | G + HGB600 | 50/50 | 0.521354 | — | — | PASS |

## Blend Strategy Notes

- **Logit-space blending** used for 2-way and 3-way combinations
- **Probability-space blending** for final submission (weighted average of raw probabilities)
- **Forward greedy selection** in logit space attempted (AGENT_EXP_AUTO_BLEND_SEARCH) — add-one approach failed (all candidates reduced AP)
- **CatBoost earns its 0.2% weight** despite worst individual AP (0.5057) because it has the most diverse predictions (Spearman rank correlation 0.60-0.72 vs other models)

---

# 7. REJECTED IDEAS

## Rejected Feature Ideas

| Feature | Fold-3 AP | Gain | Gate | Reason |
|---------|-----------|------|------|--------|
| channel_transitions | 0.506201 | -0.012025 | REJECTED_AT_CHEAP_SCREEN | Gain only +0.000184 (validation noise) |
| population_percentile | 0.506527 | -0.011699 | REJECTED_AT_CHEAP_SCREEN | Gain +0.000510 (below +0.002) |
| entity_age_density | 0.506878 | -0.011348 | REJECTED | Gain +0.000861 (below +0.002) |
| device_merchant_novelty | 0.506000 | -0.012226 | REJECTED | Insufficient detail |
| F7 block (7 features) | 0.493732 | +0.000995 | FAIL | Below +0.002 cheap screen gate |

## Rejected Model Variants

| Model | Fold-3 AP | Gain | Reason |
|-------|-----------|------|--------|
| Patience LGB | 0.500732 | -0.017494 | Worst rapidfire variant |
| Extreme LGB | 0.505470 | -0.012756 | Overly constrained |
| Deep importance LGB | 0.509486 | -0.008740 | Too complex |
| Altseed LGB (seed=9999) | 0.512202 | -0.006024 | Diversity only, not improvement |

## Rejected Acceleration Variants

| Variant | Fold-3 AP | Reason |
|---------|-----------|--------|
| top3_acceleration (106F) | 0.512409 | Superseded by full-9 |
| top7_acceleration (110F) | 0.512186 | Superseded by full-9 |
| acceleration_cheap (110F) | 0.510953 | Cheap PASS but full-9 REJECTED |

## Rejected CatBoost Pruning

| Experiment | Unpruned AP | Pruned AP | Gain | Decision |
|------------|-------------|-----------|------|----------|
| CB pruning (94F vs 101F) | 0.511112 | 0.508746 | -0.002366 | REJECT PRUNING |

## Final Decision Lock-In

Source: `EXP_LOCKED_PLAN_FINAL_DECISION/step6_final_decision.json`

```
CASE C — both population_percentile and channel_transition REJECTED_AT_CHEAP_SCREEN.
Final fold3 ensemble AP: 0.518226, public LB: 0.52642.
Recommendation: keep current submission as final.
```

---

# 8. PENDING IDEAS (ranked by priority)

## Priority 1 — BLOCKING (must do before next submit)

1. **FOLD-2 GATE for xgb_hgb600_blend (0.527088 fold-3)**
   - Fold-2 AP must be >= fold-3 - 0.001 = 0.526088
   - Submission file: `submission_file/xgb_hgb600_blend_0525.csv`
   - Script: `fold2_gate_check.py` (at project root)
   - Expected public LB: > 0.53

## Priority 2 — HIGH VALUE

2. **Download HGB183 test pred from Colab W3**
   - HGB183 is BEST INDIVIDUAL at 0.524681 fold-3
   - Test prediction exists on Colab W3 but NOT downloaded
   - Once available, rerun 5-way blend optimizer: HGB183 + XGB183 + HGB600 + mega_lgb + mega_cb

## Priority 3 — RUNNING

3. **Graph-feature LGB full-data** — RUNNING on W2
   - Screen passed: base 0.527920 → +graph 0.529101, gain +0.00118
   - 22 new graph features

4. **CatBoost-183 full-data** — RUNNING on W1
   - 1500 iterations, lr=0.03, depth=10

## Priority 4 — MEDIUM

5. **Time-decay LGB** — Script exists in `AGENT_EXP_AGGRESSIVE_V2/aggressive_pipeline.py`
6. **Adversarial validation + IsolationForest features** — Script exists
7. **Multi-fold harness (folds 1-4)** — Script exists
8. **Stacking head (LR on OOF logits)** — Requires OOF from top-5 models

## Priority 5 — LOW

9. **CatBoost multi-seed ensemble (5 seeds)** — Script exists
10. **OOF target encoding + LGB** — Script exists

---

# 9. FILE INVENTORY

## Root-Level Files

| Path | Purpose | Size |
|------|---------|------|
| train.csv | Competition training data | 87 MB |
| test.csv | Competition test data | 31 MB |
| sample_submission.csv | Submission template | 4.7 MB |
| data_dictionary.csv | Column descriptions | 1.1 KB |
| DATATHON_RULEBOOK.pdf | Competition rules | 1.6 MB |
| experiment_tracker.csv | Master experiment log (39 entries) | 3.3 KB |
| best_models.md | Ranked model/blend summary | 3.0 KB |
| fold2_gate_check.py | Fold-2 gate script for xgb_hgb600_blend | 10 KB |
| xgb183_fulldata.py | XGB183 full-data training script | 3.0 KB |
| build_5way_0523615.py | 5-way blend submission builder | 2.6 KB |
| verify_colab.py | Colab verification utility | 1.9 KB |
| feature_train_v4.parquet | V4 features (train, duplicate) | 154 MB |
| feature_test_v4.parquet | V4 features (test, duplicate) | 64 MB |
| Fold_3_v4_predictions.npy | Old fold3 predictions | 476 KB |
| catboost_v4_fold3_pred.npy | Old CatBoost fold3 predictions | 476 KB |

## Submission Files

| File | Fraud Mean | Notes |
|------|-----------|-------|
| submission_v4_primary.csv | 0.014043 | Early V4 primary |
| submission_v4_diverse.csv | 0.014165 | Early V4 diverse |
| mega_ensemble_submission.csv | 0.013287 | Mega 4-way ensemble |
| mega_3way_submission.csv | 0.012412 | Mega 3-way |
| mega_4way_submission.csv | 0.012739 | Mega 4-way |
| mega_5way_submission.csv | 0.013287 | Mega 5-way |
| mega_30_30_20_20_submission.csv | 0.012894 | Weighted 30/30/20/20 |
| mega_35_35_15_15_submission.csv | 0.012561 | Weighted 35/35/15/15 |
| mega_40_40_20_submission.csv | 0.012370 | Weighted 40/40/20 |
| mega_45_25_15_15_submission.csv | 0.012825 | Weighted 45/25/15/15 |
| mega_equal_3way_submission.csv | 0.012434 | Equal 3-way |
| mega_equal_4way_submission.csv | 0.013228 | Equal 4-way |
| mega_hgb_restricted_0523163.csv | 0.012951 | **SUBMITTED** (0.523163) |
| mega_5way_hgb600_0523382.csv | 0.013848 | SUBMITREADY (0.523382) |
| hgb_45_blend_submission.csv | 0.013467 | G+HGBt@45% |
| hgb_blend_45pct_submission.csv | 0.013467 | Same (duplicate) |
| xgb_hgb600_blend_0525.csv | 0.049363 | **BEST YET** (0.527088, pending gate) |
| submission_v4_w4_cost_0518226.csv | 0.012881 | **CHAMPION** (public LB 0.52642) |

## Experiment Folders (jupyter notebook/)

| Folder | Purpose |
|--------|---------|
| AGENT_EXP_AGGRESSIVE_V2/ | Adversarial + IF features, time-decay |
| AGENT_EXP_AUTO_BLEND_SEARCH/ | Add-one + greedy forward blend optimization |
| AGENT_EXP_AUTO_CATBOOST_TUNED/ | CatBoost tuned variants + b25 acceleration |
| AGENT_EXP_AUTO_FINAL7/ | F7 custom features (failed gate) |
| AGENT_EXP_AUTO_HGB/ | HGB600, HGB_tuned, blend search |
| AGENT_EXP_AUTO_HGB_VERIFY/ | HGB fold-2 temporal verification |
| AGENT_EXP_AUTO_REPROD/ | Champion LGB reproduction |
| AGENT_EXP_CATBOOST_PRUNING/ | CatBoost feature pruning |
| AGENT_EXP_MEGA_V2/ | Mega features (183F) + LGB/CB/HGB training |
| AGENT_EXP_RAPIDFIRE/ | 5 LGB variants + rapidfire blend |
| AGENT_EXP_XGB/ | XGB183 (215F) |
| EXP_B25_ACCELERATION/ | Acceleration cheap screen |
| EXP_B25_ACCELERATION_FOLD2/ | Fold-2 stability check |
| EXP_B25_ACCELERATION_TOP3/ | Top-3 acceleration |
| EXP_B25_ACCELERATION_TOP7/ | Top-7 acceleration |
| EXP_B25_ACCELERATION_TUNED/ | Tuned acceleration |
| EXP_CATBOOST_FEATURE_PRUNING/ | CatBoost importance for pruning |
| EXP_CHANNEL_TRANSITIONS/ | Channel transition features (rejected) |
| EXP_ENTITY_AGE_DENSITY/ | Entity age density (rejected) |
| EXP_LOCKED_PLAN_FINAL_DECISION/ | Final decision lock-in |
| EXP_POPULATION_AMOUNT_PERCENTILE/ | Population percentile (rejected) |
| EXP_W2B_COST_SENSITIVE/ | Cost-sensitive LGB (champion source) |
| EXP_W4_EXTRA_RANDOMIZED/ | Extra-randomized LGB |
| FINAL_0518226_SUBMISSION/ | Champion submission artifacts |

## Key Data Files (jupyter notebook/)

| File | Purpose |
|------|---------|
| train_combined.parquet | 183-feature training data |
| test_combined.parquet | 183-feature test data |
| y.npy | Target array (731,942) |
| Fold_2_v4_targets.npy | Fold-2 targets (59,465) |
| Fold_3_v4_targets.npy | Fold-3 targets (59,465) |
| model_features_v4.pkl | Feature name list (101 V4 core) |
| categorical_features_v4.pkl | Categorical feature names (13) |

---

# 10. MODEL DIVERSITY ANALYSIS

## Pearson Correlation Matrix

| Model | G_final | W4_stageB | G_extra_rand | G_CB | G_HGB600 | G_HGBtuned | G_TA_accel |
|-------|---------|-----------|--------------|------|----------|------------|------------|
| G_final | 1.000 | 0.996 | 0.980 | 0.977 | 0.968 | 0.972 | 0.991 |
| W4_stageB | 0.996 | 1.000 | 0.978 | 0.977 | 0.968 | 0.971 | 0.990 |
| G_extra_rand | 0.980 | 0.978 | 1.000 | 0.942 | 0.945 | 0.948 | 0.971 |
| G_CB | 0.977 | 0.977 | 0.942 | 1.000 | 0.948 | 0.953 | 0.973 |
| G_HGB600 | 0.968 | 0.968 | 0.945 | 0.948 | 1.000 | 0.972 | 0.965 |
| G_HGBtuned | 0.972 | 0.971 | 0.948 | 0.953 | 0.972 | 1.000 | 0.970 |
| G_TA_accel | 0.991 | 0.990 | 0.971 | 0.973 | 0.965 | 0.970 | 1.000 |

## Spearman Rank Correlation

| Model | G_final | W4_stageB | G_extra_rand | G_CB | G_HGB600 | G_HGBtuned | G_TA_accel |
|-------|---------|-----------|--------------|------|----------|------------|------------|
| G_final | 1.000 | 0.953 | 0.858 | 0.716 | 0.709 | 0.698 | 0.926 |
| W4_stageB | 0.953 | 1.000 | 0.844 | 0.718 | 0.704 | 0.691 | 0.924 |
| G_extra_rand | 0.858 | 0.844 | 1.000 | 0.675 | 0.681 | 0.667 | 0.843 |
| G_CB | 0.716 | 0.718 | 0.675 | 1.000 | 0.604 | 0.613 | 0.703 |
| G_HGB600 | 0.709 | 0.704 | 0.681 | 0.604 | 1.000 | 0.853 | 0.698 |
| G_HGBtuned | 0.698 | 0.691 | 0.667 | 0.613 | 0.853 | 1.000 | 0.686 |
| G_TA_accel | 0.926 | 0.924 | 0.843 | 0.703 | 0.698 | 0.686 | 1.000 |

## Key Diversity Observations

1. **CatBoost is most diverse** vs all others: rank correlation 0.60-0.72. Earns 0.2% blend weight despite worst individual AP.
2. **HGB600 and HGB_tuned** are highly correlated (Spearman 0.853) — limited incremental diversity.
3. **G_final and W4_stageB** are nearly identical (Pearson 0.996, Spearman 0.953).
4. **Best blend** exploits XGB/HGB diversity as the main gain driver.

## Ranking Diagnostics (Precision/Capture at K%)

| Model | P@1% | C@1% | P@2% | C@2% | P@5% | C@5% |
|-------|------|------|------|------|------|------|
| G_champion | 0.7559 | 0.4849 | 0.3936 | 0.5054 | 0.1732 | 0.5562 |
| HGB600 | 0.7559 | 0.4849 | 0.3978 | 0.5108 | 0.1692 | 0.5432 |
| HGB_tuned | 0.7626 | 0.4892 | 0.3970 | 0.5097 | 0.1695 | 0.5443 |

**Missed fraud overlap:** HGB600 vs G Jaccard = 0.931 (93% overlap in missed cases). Limited diversity on missed fraud.

---

# 11. FEATURE IMPORTANCE ANALYSIS

## LGB Mega (183 features) — Gain Importance

| Rank | Feature | Gain |
|------|---------|------|
| 1 | merchant_previous_customer | 718 |
| 2 | device_previous_customer | 538 |
| 3 | customer_id | 491 |
| 4 | device_id | 421 |
| 5 | merchant_id | 330 |
| 6 | customer_previous_device | 302 |

Source: `AGENT_EXP_MEGA_V2/lgb_mega_importance.csv`

## CatBoost FULL110 Tuned — Gain Importance

| Rank | Feature | Gain |
|------|---------|------|
| 1 | customer_amount_ratio | 12.62 |
| 2 | customer_location_previous_count | 10.71 |
| 3 | is_new_location_for_customer | 8.29 |
| 4 | customer_device_previous_count | 7.32 |
| 5 | device_id | 7.28 |
| 6 | merchant_recent_1h_share | 6.34 |
| 7 | customer_merchant_previous_count | 5.59 |
| 8 | customer_id | 5.18 |

Source: `AGENT_EXP_AUTO_CATBOOST_TUNED/catboost_FULL110_tuned_importance.csv`

## CatBoost Pruned (94F) — Gain Importance

| Rank | Feature | Gain |
|------|---------|------|
| 1 | customer_amount_mean_previous_24h | 1.417 |
| 2 | merchant_category | 1.286 |
| 3 | customer_transactions_previous_7d | 1.260 |
| 4 | customer_amount_ratio | 1.229 |
| 5 | customer_amount_ratio_previous_7d | 1.196 |
| 6 | is_new_location_for_customer | 1.045 |
| 7 | merchant_recent_1h_share | 0.859 |
| 8 | device_id | 0.780 |

Source: `AGENT_EXP_CATBOOST_PRUNING/catboost_pruned_importance.csv`

## B25 Acceleration Features — Gain Importance

| Rank | Feature | Gain | Splits |
|------|---------|------|--------|
| 1 | b25_customer_1h_to_24h_ratio | 59010 | 63 |
| 2 | b25_merchant_6h_to_24h_ratio | 41180 | 217 |
| 3 | b25_customer_6h_to_24h_ratio | 21760 | 27 |
| 4 | b25_merchant_1h_to_24h_ratio | 3132 | 140 |
| 5 | b25_device_6h_to_24h_ratio | 1480 | 42 |
| 6 | b25_customer_1h_to_6h_ratio | 651 | 100 |
| 7 | b25_merchant_1h_to_6h_ratio | 49 | 72 |
| 8 | b25_device_1h_to_24h_ratio | 0.86 | 3 |
| 9 | b25_device_1h_to_6h_ratio | 0.55 | 2 |

Source: `EXP_B25_ACCELERATION/acceleration_feature_importance.csv`

## Global Top Features Summary

The most consistently important features across models are: **merchant_previous_customer**, **device_previous_customer**, **customer_id**, **device_id**, **merchant_id**, **customer_previous_device**, **customer_amount_ratio**, **is_new_location_for_customer**, **merchant_recent_1h_share**, and **customer_location_previous_count**.

---

# 12. ERROR ANALYSIS

## False Negative Patterns

From ranking diagnostics (ranking_diagnostics.csv):

- **At top-1% precision:** All models capture ~48.5% of fraud. This means 51.5% of fraud cases are ranked below the top 1% of transactions.
- **At top-5% precision:** Champion LGB captures 55.6% of fraud. HGB captures 54.3%.
- **HGB600 vs G overlap:** Jaccard similarity of missed fraud = 0.931 (93% overlap). Both models miss the same hard fraud cases.
- **HGB_tuned vs G overlap:** Jaccard = 0.911 (slightly more diverse on missed cases).

## Stability Issues

- **Fold-2 vs fold-3 fraud rate divergence:** fold-2 fraud rate = 1.845%, fold-3 = 1.557%. The ~18% relative difference means fold-3 is harder to detect fraud in, potentially explaining why fold-3 AP is generally lower.
- **Verification bug:** Initial fold-2 verification showed HGB600 AP = 0.018 (clearly wrong). Fixed in `rebuild_fold2_temporal.py`. Corrected fold-2 temporal AP = 0.8205.

## Drift Problems

See Section 13 for detailed drift analysis.

---

# 13. TRAIN-TEST DRIFT AUDIT

## Entity Age Density Drift

Source: `EXP_ENTITY_AGE_DENSITY/entity_age_density_diagnostics.csv`

| Feature | Train Mean | Test Mean | Ratio | Drift |
|---------|-----------|-----------|-------|-------|
| device_days_since_first_seen | 94.78 | 221.37 | 2.33× | **SEVERE** |
| merchant_days_since_first_seen | 98.27 | 225.79 | 2.30× | **SEVERE** |
| device_activity_per_age_day | 0.448 | 0.438 | 0.98× | Stable |
| merchant_activity_per_age_day | 222.4 | 233.7 | 1.05× | Stable |

**Key finding:** `device_days_since_first_seen` and `merchant_days_since_first_seen` are 2.3× higher in test. This is expected temporally (test entities are older) but creates drift for these features.

## Channel Transition Drift

Source: `EXP_CHANNEL_TRANSITIONS/step5a_channel_feature_diagnostics.csv`

| Feature | Train Rate | Test Rate | Delta |
|---------|-----------|-----------|-------|
| customer_payment_method_changed_from_previous | 0.696 | 0.731 | +0.035 |
| customer_transaction_type_changed_from_previous | 0.609 | 0.640 | +0.031 |
| customer_device_type_changed_from_previous | 0.123 | 0.125 | +0.002 |

Payment method and transaction type change rates are slightly higher in test (+3-4%).

## New Entities in Test

From entity_age_density diagnostics:
- **merchant_days_since_first_seen** unique count: 196 (train) → 250 (test). **54 new merchants** appear in test that were not in train.

---

# 14. LEAKAGE AUDIT

## Status by Feature Group

| Feature Group | Status | Risk Level | Notes |
|---------------|--------|------------|-------|
| Base V4 (101 features) | SAFE | NONE | All built with past-only lookups, confirmed by code |
| Mega features (82) | SAFE | NONE | Built with same temporal-only approach |
| NA indicator features (23) | SAFE | NONE | Derived from NaN patterns, applied separately |
| B25 Acceleration (9) | SAFE | NONE | Windowed ratios using past data only |
| OOF target encoding (24) | SAFE | NONE | Proper out-of-fold by construction |
| Entity Age Density | SAFE | LOW DRIFT | Temporal features, safe but drift present |
| Population Percentile | SAFE | NONE | Past-only percentile ranking |
| Channel Transitions | SAFE | NONE | Binary flags for changed fields |
| F7 Custom (7) | SAFE | NONE | All from historical lookups |
| Graph Features (22) | UNCERTAIN | UNKNOWN | Built on Colab, not yet reviewed locally |
| Adversarial/IF (V2) | UNCERTAIN | UNKNOWN | Not yet implemented in production |

## Key Leakage Safeguards

1. All velocity/count features use cumulative sums computed up to (but not including) the current row.
2. OOF target encoding uses James-Stein shrinkage with prior=20, proper out-of-fold splits.
3. No target information leaks into features — verified by code inspection.
4. No features were rejected due to leakage. All rejections were due to insufficient AP gain.

---

# 15. RESEARCH RECOMMENDATIONS

> *Summarized from information already present in the repository. No inventions.*

## Highest Priority Next Experiments

1. **Submit xgb_hgb600_blend after fold-2 gate.** Expected public LB > 0.53. Script ready, just needs execution.

2. **Download HGB183 test predictions from Colab W3.** Best individual model (0.524681), preds stranded on retiring Colab. Then rerun 5-way blend optimizer.

3. **Evaluate graph-feature LGB full-data results** when W2 completes. Graph features showed +0.00118 gain in screen.

4. **Evaluate CatBoost-183 full-data results** when W1 completes. New blend member candidate.

5. **Re-optimize global blend** once all running experiments land: add HGB183, XGB183, graph-LGB, CatBoost-183 to the pool and re-run weight optimization.

---

# 16. APPENDIX

## Folder Tree

```
D:\Desktop\React datathon\
├── .venv/                          (Python virtual environment)
├── COLAB_REQUIRED_FILES/           (Minimal upload set for Colab workers)
│   ├── Fold_2_v4_targets.npy
│   ├── Fold_3_v4_targets.npy
│   ├── categorical_features_v4.pkl
│   ├── file_manifest.txt
│   ├── model_features_v4.pkl
│   ├── sample_submission.csv
│   ├── test_combined.parquet
│   ├── train_combined.parquet
│   └── y.npy
├── DATATHON_RULEBOOK.pdf
├── data/
│   ├── feature_test_v4.parquet
│   ├── feature_train_v4.parquet
│   ├── test.csv
│   └── train.csv
├── jupyter notebook/
│   ├── AGENT_EXP_AGGRESSIVE_V2/
│   ├── AGENT_EXP_AUTO_BLEND_SEARCH/
│   ├── AGENT_EXP_AUTO_CATBOOST_TUNED/
│   ├── AGENT_EXP_AUTO_FINAL7/
│   ├── AGENT_EXP_AUTO_HGB/
│   ├── AGENT_EXP_AUTO_HGB_VERIFY/
│   ├── AGENT_EXP_AUTO_REPROD/
│   ├── AGENT_EXP_CATBOOST_PRUNING/
│   ├── AGENT_EXP_MEGA_V2/
│   ├── AGENT_EXP_RAPIDFIRE/
│   ├── AGENT_EXP_XGB/
│   ├── EXP_B25_ACCELERATION/
│   ├── EXP_B25_ACCELERATION_FOLD2/
│   ├── EXP_B25_ACCELERATION_TOP3/
│   ├── EXP_B25_ACCELERATION_TOP7/
│   ├── EXP_B25_ACCELERATION_TUNED/
│   ├── EXP_CATBOOST_FEATURE_PRUNING/
│   ├── EXP_CHANNEL_TRANSITIONS/
│   ├── EXP_ENTITY_AGE_DENSITY/
│   ├── EXP_LOCKED_PLAN_FINAL_DECISION/
│   ├── EXP_POPULATION_AMOUNT_PERCENTILE/
│   ├── EXP_W2B_COST_SENSITIVE/
│   ├── EXP_W4_EXTRA_RANDOMIZED/
│   ├── FINAL_0518226_SUBMISSION/
│   ├── feature_train_v4.parquet
│   ├── feature_test_v4.parquet
│   ├── model_features_v4.pkl
│   ├── categorical_features_v4.pkl
│   ├── Fold_2_v4_targets.npy
│   ├── Fold_3_v4_targets.npy
│   ├── y.npy
│   └── *.npy (various predictions)
├── submission_file/                (17 submission CSVs)
├── experiment_tracker.csv
├── best_models.md
├── build_5way_0523615.py
├── fold2_gate_check.py
├── xgb183_fulldata.py
└── verify_colab.py
```

## Environment Setup

- **OS:** Windows 11 (win32)
- **Python:** 3.14 (Local), 3.10+ (Colab)
- **Key libraries:** LightGBM, XGBoost, CatBoost, scikit-learn, pandas, numpy
- **Hardware (local):** CPU training (no GPU for local models)
- **Colab workers:** GPU-enabled (CatBoost GPU, LightGBM GPU)

## Worker Assignments (historical)

| Worker | Type | Tasks Completed |
|--------|------|----------------|
| W1 (tab_1) | Colab | CB mega, CB tuned, CB pruning, CB acceleration |
| W2 (tab_2) | Colab | Mega LGB, graph features, channel/entity/pop features |
| W3 (tab_3) | Colab (RETIRED) | HGB183, rapidfire LGB variants |
| W4 (local) | CPU | HGB600, XGB183, blend search, fold-2 gate, cost-sensitive |

## Submission History

| # | Date | Filename | Fold-3 AP | Public LB | Notes |
|---|------|----------|-----------|-----------|-------|
| 1 | Early | submission_v4_primary.csv | — | — | Early V4 |
| 2 | Early | submission_v4_diverse.csv | — | — | Early V4 diverse |
| 3 | Mid | mega_hgb_restricted_0523163.csv | 0.523163 | — | Restricted 4-way |
| 4 | Current | submission_v4_w4_cost_0518226.csv | 0.518226 | **0.52642** | **CHAMPION** |
| 5 | Pending | xgb_hgb600_blend_0525.csv | 0.527088 | — | Best blend, gate pending |
| 6 | Pending | mega_5way_hgb600_0523382.csv | 0.523382 | — | 5-way, ready |

---

*End of PROJECT_MASTER_DOSSIER.md*
