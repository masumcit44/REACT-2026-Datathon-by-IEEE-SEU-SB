"""
Efficient mega-feature pipeline v2. Avoids slow per-group iteration.
Uses vectorized operations and simplified approximations.
"""
import numpy as np
import pandas as pd
import os
import pickle
import time

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_MEGA_V2")
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
t0 = time.time()
tr = pd.read_csv(r"train.csv", parse_dates=["timestamp"])
te = pd.read_csv(r"test.csv", parse_dates=["timestamp"])
print(f"Train: {tr.shape}, Test: {te.shape}")

y = tr["fraud"].values.astype(int)
print(f"Fraud rate: {y.mean():.4f}")

# Combine
tr["split"] = "train"
te["split"] = "test"
te["fraud"] = -1
all_data = pd.concat([tr, te], ignore_index=True)
all_data = all_data.sort_values("timestamp").reset_index(drop=True)

train_mask = all_data["split"] == "train"
test_mask = all_data["split"] == "test"

# Save metadata
np.save(os.path.join(OUT, "y.npy"), y)

feat = pd.DataFrame(index=all_data.index)
N = len(all_data)
print(f"Total rows: {N}")

# ============================================================
# 1. BASE FEATURES
# ============================================================
print("\n[1/6] Base features...")
feat["hour"] = all_data["timestamp"].dt.hour
feat["day_of_week"] = all_data["timestamp"].dt.dayofweek
feat["is_weekend"] = (feat["day_of_week"] >= 5).astype(np.int8)
feat["is_night"] = ((feat["hour"] >= 22) | (feat["hour"] <= 5)).astype(np.int8)
feat["amount"] = all_data["amount_bdt"].values
feat["amount_log"] = np.log1p(feat["amount"].values)
feat["amount_sqrt"] = np.sqrt(feat["amount"].values)
feat["amount_round100"] = (feat["amount"].values % 100 == 0).astype(np.int8)
feat["amount_round500"] = (feat["amount"].values % 500 == 0).astype(np.int8)
feat["account_age_days"] = all_data["account_age_days"].fillna(0).values
feat["account_age_log"] = np.log1p(feat["account_age_days"].values)
feat["is_new_account"] = (feat["account_age_days"].values < 30).astype(np.int8)
feat["is_very_new"] = (feat["account_age_days"].values < 7).astype(np.int8)
feat["amt_x_age"] = feat["amount_log"] * feat["account_age_log"]

# Categorical codes
for col in ["merchant_category", "device_type", "location", "payment_method", "transaction_type"]:
    feat[f"{col}_code"] = all_data[col].fillna("missing").astype("category").cat.codes.astype(np.int32)

print(f"  {feat.shape[1]} features")

# ============================================================
# 2. CUMULATIVE COUNTS (vectorized)
# ============================================================
print("\n[2/6] Cumulative counts...")

# These are fast vectorized operations
feat["cust_cumcount"] = all_data.groupby("customer_id").cumcount().values
feat["dev_cumcount"] = all_data.groupby("device_id").cumcount().values
feat["merch_cumcount"] = all_data.groupby("merchant_id").cumcount().values

# Cross-entity counts
feat["cust_merch_count"] = all_data.groupby(["customer_id", "merchant_id"]).cumcount().values
feat["cust_dev_count"] = all_data.groupby(["customer_id", "device_id"]).cumcount().values
feat["dev_merch_count"] = all_data.groupby(["device_id", "merchant_id"]).cumcount().values

# First-time flags
feat["is_first_cust_merch"] = (feat["cust_merch_count"] == 0).astype(np.int8)
feat["is_first_cust_dev"] = (feat["cust_dev_count"] == 0).astype(np.int8)
feat["is_first_dev_merch"] = (feat["dev_merch_count"] == 0).astype(np.int8)

# Log counts
feat["cust_count_log"] = np.log1p(feat["cust_cumcount"])
feat["dev_count_log"] = np.log1p(feat["dev_cumcount"])
feat["merch_count_log"] = np.log1p(feat["merch_cumcount"])

print(f"  {feat.shape[1]} features")

# ============================================================
# 3. AMOUNT RANK FEATURES (vectorized)
# ============================================================
print("\n[3/6] Amount rank features...")

feat["amt_pctile_merch"] = all_data.groupby("merchant_id")["amount_bdt"].rank(pct=True).values
feat["amt_pctile_cust"] = all_data.groupby("customer_id")["amount_bdt"].rank(pct=True).values
feat["amt_pctile_dev"] = all_data.groupby("device_id")["amount_bdt"].rank(pct=True).values
feat["amt_pctile_merch_cat"] = all_data.groupby("merchant_category")["amount_bdt"].rank(pct=True).values
feat["amt_pctile_dev_type"] = all_data.groupby("device_type")["amount_bdt"].rank(pct=True).values

# Amount deviation from customer mean (past-only approximation using expanding mean)
print("  Amount deviation from customer mean...")
cust_amt_cumsum = all_data.groupby("customer_id")["amount_bdt"].cumsum().values
cust_amt_cumcount = all_data.groupby("customer_id").cumcount().values
cust_amt_cumcount_safe = np.maximum(cust_amt_cumcount, 1)
cust_amt_mean = (cust_amt_cumsum - all_data["amount_bdt"].values) / cust_amt_cumcount_safe
feat["amt_vs_cust_mean"] = (all_data["amount_bdt"].values - cust_amt_mean) / (cust_amt_mean + 1)
feat["amt_ratio_cust_mean"] = all_data["amount_bdt"].values / (cust_amt_mean + 1)

print(f"  {feat.shape[1]} features")

# ============================================================
# 4. TEMPORAL VELOCITY FEATURES
# ============================================================
print("\n[4/6] Temporal velocity features...")

# Time gaps (using groupby shift - fast)
feat["hours_since_cust"] = ((all_data["timestamp"] - all_data.groupby("customer_id")["timestamp"].shift(1)).dt.total_seconds() / 3600).values
feat["hours_since_dev"] = ((all_data["timestamp"] - all_data.groupby("device_id")["timestamp"].shift(1)).dt.total_seconds() / 3600).values
feat["hours_since_merch"] = ((all_data["timestamp"] - all_data.groupby("merchant_id")["timestamp"].shift(1)).dt.total_seconds() / 3600).values

# Days since
feat["days_since_cust"] = feat["hours_since_cust"] / 24
feat["days_since_cust_merch"] = ((all_data["timestamp"] - all_data.groupby(["customer_id", "merchant_id"])["timestamp"].shift(1)).dt.total_seconds() / 86400).values

# Amount change vs last
cust_last_amt = all_data.groupby("customer_id")["amount_bdt"].shift(1).values
feat["amt_change_cust"] = (all_data["amount_bdt"].values - cust_last_amt) / (cust_last_amt + 1)
feat["amt_ratio_cust"] = all_data["amount_bdt"].values / (cust_last_amt + 1)

# Velocity flags
feat["cust_tx_in_24h"] = (feat["hours_since_cust"] <= 24).astype(np.int8)
feat["cust_tx_in_1h"] = (feat["hours_since_cust"] <= 1).astype(np.int8)
feat["dev_tx_in_24h"] = (feat["hours_since_dev"] <= 24).astype(np.int8)
feat["dev_tx_in_1h"] = (feat["hours_since_dev"] <= 1).astype(np.int8)
feat["merch_tx_in_24h"] = (feat["hours_since_merch"] <= 24).astype(np.int8)
feat["is_dormant_reactivation"] = (feat["days_since_cust_merch"] > 30).astype(np.int8)

print(f"  {feat.shape[1]} features")

# ============================================================
# 5. OOF TARGET ENCODING (vectorized expanding)
# ============================================================
print("\n[5/6] OOF target encoding...")

global_rate = y.mean()
prior = 20

for col in ["merchant_id", "device_id", "customer_id", "merchant_category",
            "device_type", "location", "payment_method", "transaction_type"]:
    print(f"  {col}...")

    # Manual expanding computation - fully aligned, fast
    grp_ids = all_data[col].values
    fraud_vals = all_data["fraud"].replace(-1, np.nan).values  # test fraud=-1 -> NaN

    # Sort indices by (group, original_order) to ensure temporal order within groups
    order = np.argsort(grp_ids, kind="mergesort")  # stable sort preserves time order
    sorted_groups = grp_ids[order]
    sorted_fraud = fraud_vals[order]

    # Compute expanding sum and count per group
    exp_sum_sorted = np.zeros(N)
    exp_count_sorted = np.zeros(N)
    prev_group = None
    running_sum = 0.0
    running_count = 0
    for i in order:
        g = sorted_groups[i] if False else grp_ids[i]  # use original
        pass

    # Simpler approach: use cumsum within groups
    # Create group ids and compute within-group cumsum
    from collections import defaultdict
    group_sums = defaultdict(float)
    group_counts = defaultdict(int)
    oof_sum = np.zeros(N)
    oof_count = np.zeros(N)

    for i in range(N):
        g = grp_ids[i]
        f = fraud_vals[i]
        if not np.isnan(f):
            oof_sum[i] = group_sums[g]
            oof_count[i] = group_counts[g]
            group_sums[g] += f
            group_counts[g] += 1
        else:
            oof_sum[i] = group_sums[g]
            oof_count[i] = group_counts[g]

    # James-Stein
    feat[f"oof_{col}"] = np.where(oof_count > 0,
                                    (oof_sum + prior * global_rate) / (oof_count + prior),
                                    global_rate)
    feat[f"oof_raw_{col}"] = np.where(oof_count > 0, oof_sum / oof_count, global_rate)
    feat[f"oof_count_{col}"] = oof_count

print(f"  {feat.shape[1]} features")

# ============================================================
# 6. INTERACTION FEATURES
# ============================================================
print("\n[6/6] Interaction features...")

feat["amt_x_cust_count"] = feat["amount_log"] * feat["cust_count_log"]
feat["amt_x_merch_count"] = feat["amount_log"] * feat["merch_count_log"]
feat["cust_count_x_dev_count"] = feat["cust_count_log"] * feat["dev_count_log"]
feat["merch_distinct_x_amt"] = np.log1p(feat["merch_cumcount"].values) * feat["amount_log"]
feat["dev_distinct_x_amt"] = np.log1p(feat["dev_cumcount"].values) * feat["amount_log"]
feat["oof_merch_x_amt"] = feat["oof_merchant_id"] * feat["amount_log"]
feat["oof_cust_x_amt"] = feat["oof_customer_id"] * feat["amount_log"]

print(f"  {feat.shape[1]} features total")

# ============================================================
# SAVE
# ============================================================
print("\nSaving...")
feat = feat.replace([np.inf, -np.inf], np.nan)

X_train_new = feat[train_mask.values].copy()
X_test_new = feat[test_mask.values].copy()

X_train_new.to_parquet(os.path.join(OUT, "train_new_features.parquet"))
X_test_new.to_parquet(os.path.join(OUT, "test_new_features.parquet"))

# Combined with V4
mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
tr_v4 = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
te_v4 = pd.read_parquet(os.path.join(BASE, "feature_test_v4.parquet"))

# Add prefix to new features to avoid column name conflicts
X_train_new.columns = [f"n_{c}" for c in X_train_new.columns]
X_test_new.columns = [f"n_{c}" for c in X_test_new.columns]

X_train_all = pd.concat([tr_v4[mfeat].reset_index(drop=True), X_train_new.reset_index(drop=True)], axis=1)
X_test_all = pd.concat([te_v4[mfeat].reset_index(drop=True), X_test_new.reset_index(drop=True)], axis=1)

# Categorical handling
for c in cats:
    X_train_all[c] = X_train_all[c].fillna("missing").astype("category").cat.codes
    X_test_all[c] = X_test_all[c].fillna("missing").astype("category").cat.codes

# Cast to float32
for c in X_train_all.columns:
    try:
        if X_train_all[c].dtype.kind == 'f':
            X_train_all[c] = X_train_all[c].astype(np.float32)
            X_test_all[c] = X_test_all[c].astype(np.float32)
    except Exception:
        pass

X_train_all.to_parquet(os.path.join(OUT, "train_combined.parquet"))
X_test_all.to_parquet(os.path.join(OUT, "test_combined.parquet"))

print(f"\nCombined train: {X_train_all.shape}")
print(f"Combined test: {X_test_all.shape}")
print(f"Total time: {time.time()-t0:.0f}s")
print("DONE")
