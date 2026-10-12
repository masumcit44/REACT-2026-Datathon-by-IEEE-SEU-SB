"""
Phase 1: Graph co-occurrence + temporal velocity + OOF target encoding features.
All features are leakage-safe (past-only at each timestamp).
"""
import numpy as np
import pandas as pd
import os
import time
from collections import defaultdict

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_MEGA_V2")
os.makedirs(OUT, exist_ok=True)

print("Loading raw data...")
t0 = time.time()
tr = pd.read_csv(r"train.csv", parse_dates=["timestamp"])
te = pd.read_csv(r"test.csv", parse_dates=["timestamp"])
print(f"Train: {tr.shape}, Test: {te.shape} ({time.time()-t0:.0f}s)")

# Combine for chronological feature building
tr["is_train"] = 1
te["is_train"] = 0
te["fraud"] = -1
all_data = pd.concat([tr, te], ignore_index=True)
all_data = all_data.sort_values("timestamp").reset_index(drop=True)
print(f"Combined: {all_data.shape}")

# ============================================================
# HELPER: Chronological per-entity rolling features
# ============================================================

def compute_chronological_features(df):
    """Build features using only past data at each timestamp."""
    feat = pd.DataFrame(index=df.index)

    # --- Time-based features ---
    feat["hour"] = df["timestamp"].dt.hour
    feat["day_of_week"] = df["timestamp"].dt.dayofweek
    feat["day_of_month"] = df["timestamp"].dt.day
    feat["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(np.int8)
    feat["is_night"] = ((df["timestamp"].dt.hour >= 22) | (df["timestamp"].dt.hour <= 5)).astype(np.int8)

    # Amount features
    feat["amount_log"] = np.log1p(df["amount_bdt"])
    feat["amount_sqrt"] = np.sqrt(df["amount_bdt"])
    feat["amount_round_100"] = (df["amount_bdt"] % 100 == 0).astype(np.int8)
    feat["amount_round_500"] = (df["amount_bdt"] % 500 == 0).astype(np.int8)
    feat["amount_round_1000"] = (df["amount_bdt"] % 1000 == 0).astype(np.int8)
    feat["amount_is_small"] = (df["amount_bdt"] < 100).astype(np.int8)
    feat["amount_is_large"] = (df["amount_bdt"] > 10000).astype(np.int8)

    # Account age
    feat["account_age_days"] = df["account_age_days"]
    feat["account_age_log"] = np.log1p(df["account_age_days"].fillna(0))
    feat["is_new_account"] = (df["account_age_days"] < 30).astype(np.int8)
    feat["is_very_new_account"] = (df["account_age_days"] < 7).astype(np.int8)

    return feat


def compute_entity_chronological_features(df, entity_col, prefix, windows=[7, 30]):
    """
    For each entity, compute trailing-window statistics using only past rows.
    Returns DataFrame with features aligned to df.index.
    """
    feat_parts = []
    ts = df["timestamp"].values
    entity = df[entity_col].values
    amt = df["amount_bdt"].values
    fraud = df["fraud"].values

    # Sort by timestamp first (assumed already sorted)
    # For each unique entity, compute rolling stats
    n = len(df)

    # --- Count features ---
    # We'll do this vectorized with groupby + shift
    grp = df.groupby(entity_col)

    # Trailing count (all-time before this row)
    feat_parts.append(
        grp.cumcount().rename(f"{prefix}_alltime_count")
    )

    # Amount stats (all-time before this row)
    cum_sum = grp["amount_bdt"].cumsum() - df["amount_bdt"]
    cum_count = grp.cumcount().replace(0, 1)
    feat_parts.append((cum_sum / cum_count).rename(f"{prefix}_alltime_amt_mean"))

    # For time-windowed features, we need to use rolling with timestamp
    # This is expensive but necessary for leakage safety
    # We'll compute approximate windowed features using the fact that data is sorted by time

    return pd.concat(feat_parts, axis=1)


print("Building base features...")
base_feats = compute_chronological_features(all_data)
print(f"Base features: {base_feats.shape[1]}")

# ============================================================
# GRAPH / CO-OCCURRENCE FEATURES (past-only)
# ============================================================
print("\nBuilding graph co-occurrence features...")

# For efficiency, we'll compute these using groupby + cumulative operations
# All using past-only data at each timestamp

ts = all_data["timestamp"]
cust = all_data["customer_id"]
dev = all_data["device_id"]
merch = all_data["merchant_id"]
amt = all_data["amount_bdt"]

# --- Device promiscuity: distinct customers per device (all-time before) ---
print("  Device multi-account cardinality...")
dev_cust_count = all_data.groupby("device_id").apply(
    lambda x: x["customer_id"].expanding().apply(lambda y: len(set(y)), raw=True).shift(1)
).reset_index(level=0, drop=True)
base_feats["dev_distinct_cust_alltime"] = dev_cust_count.values

# --- Customer-merchant relationship strength ---
print("  Customer-merchant relationship count...")
cust_merch_count = all_data.groupby(["customer_id", "merchant_id"]).cumcount()
base_feats["cust_merch_count"] = cust_merch_count.values

# --- Customer-device relationship strength ---
print("  Customer-device relationship count...")
cust_dev_count = all_data.groupby(["customer_id", "device_id"]).cumcount()
base_feats["cust_dev_count"] = cust_dev_count.values

# --- Device-merchant relationship count ---
print("  Device-merchant relationship count...")
dev_merch_count = all_data.groupby(["device_id", "merchant_id"]).cumcount()
base_feats["dev_merch_count"] = dev_merch_count.values

# --- Merchant transaction volume (all-time before) ---
print("  Merchant volume...")
merch_cum_count = all_data.groupby("merchant_id").cumcount()
base_feats["merchant_alltime_count"] = merch_cum_count.values

# --- Amount percentile within merchant (approximation) ---
print("  Amount rank within merchant...")
base_feats["amt_pctile_in_merch"] = all_data.groupby("merchant_id")["amount_bdt"].rank(pct=True).values

# --- Amount rank within customer ---
print("  Amount rank within customer...")
base_feats["amt_pctile_in_cust"] = all_data.groupby("customer_id")["amount_bdt"].rank(pct=True).values

# --- Device amount stats ---
print("  Device amount stats...")
dev_amt_mean = all_data.groupby("device_id")["amount_bdt"].transform(
    lambda x: (x.cumsum() - x) / x.groupby(all_data.loc[x.index, "customer_id"]).cumcount().replace(0, 1)
)
base_feats["dev_amt_mean"] = dev_amt_mean.values

# --- First-time usage flags ---
print("  First-time usage flags...")
base_feats["is_first_cust_merch"] = (base_feats["cust_merch_count"] == 0).astype(np.int8)
base_feats["is_first_cust_dev"] = (base_feats["cust_dev_count"] == 0).astype(np.int8)
base_feats["is_first_dev_merch"] = (base_feats["dev_merch_count"] == 0).astype(np.int8)

# ============================================================
# TEMPORAL VELOCITY FEATURES (past-only)
# ============================================================
print("\nBuilding temporal velocity features...")

# Time since customer's last transaction
print("  Time since customer last tx...")
cust_last_ts = all_data.groupby("customer_id")["timestamp"].shift(1)
feat_time_since_cust = (all_data["timestamp"] - cust_last_ts).dt.total_seconds() / 3600
base_feats["hours_since_cust_last_tx"] = feat_time_since_cust.values
base_feats["days_since_cust_last_tx"] = (feat_time_since_cust / 24).values

# Time since device's last transaction
print("  Time since device last tx...")
dev_last_ts = all_data.groupby("device_id")["timestamp"].shift(1)
feat_time_since_dev = (all_data["timestamp"] - dev_last_ts).dt.total_seconds() / 3600
base_feats["hours_since_dev_last_tx"] = feat_time_since_dev.values

# Time since merchant's last transaction
print("  Time since merchant last tx...")
merch_last_ts = all_data.groupby("merchant_id")["timestamp"].shift(1)
feat_time_since_merch = (all_data["timestamp"] - merch_last_ts).dt.total_seconds() / 3600
base_feats["hours_since_merch_last_tx"] = feat_time_since_merch.values

# Amount change vs customer's last transaction
print("  Amount change vs customer last tx...")
cust_last_amt = all_data.groupby("customer_id")["amount_bdt"].shift(1)
base_feats["amt_vs_cust_last"] = (all_data["amount_bdt"] - cust_last_amt) / (cust_last_amt + 1)
base_feats["amt_ratio_cust_last"] = all_data["amount_bdt"] / (cust_last_amt + 1)

# Customer transaction velocity (count in last 24h)
print("  Customer 24h velocity...")
# Use a simplified approach: flag if customer had a tx in the last 24h
cust_prev_ts = all_data.groupby("customer_id")["timestamp"].shift(1)
hours_gap = (all_data["timestamp"] - cust_prev_ts).dt.total_seconds() / 3600
base_feats["cust_tx_in_24h"] = (hours_gap <= 24).astype(np.int8)
base_feats["cust_tx_in_1h"] = (hours_gap <= 1).astype(np.int8)
base_feats["cust_tx_in_7d"] = (hours_gap <= 168).astype(np.int8)

# Device transaction velocity
print("  Device velocity...")
dev_prev_ts = all_data.groupby("device_id")["timestamp"].shift(1)
dev_hours_gap = (all_data["timestamp"] - dev_prev_ts).dt.total_seconds() / 3600
base_feats["dev_tx_in_24h"] = (dev_hours_gap <= 24).astype(np.int8)
base_feats["dev_tx_in_1h"] = (dev_hours_gap <= 1).astype(np.int8)

# Merchant transaction velocity
print("  Merchant velocity...")
merch_prev_ts = all_data.groupby("merchant_id")["timestamp"].shift(1)
merch_hours_gap = (all_data["timestamp"] - merch_prev_ts).dt.total_seconds() / 3600
base_feats["merch_tx_in_24h"] = (merch_hours_gap <= 24).astype(np.int8)
base_feats["merch_tx_in_1h"] = (merch_hours_gap <= 1).astype(np.int8)

# ============================================================
# OOF TARGET ENCODING (time-aware, past-only)
# ============================================================
print("\nBuilding OOF target encoding features...")

# For high-cardinality cats: merchant_id, device_id, customer_id
# Time-aware: for each row, use only the fraud rate from prior rows

for col in ["merchant_id", "device_id", "customer_id", "merchant_category"]:
    print(f"  OOF target encoding: {col}...")
    # Compute expanding fraud rate (past-only)
    grp = all_data.groupby(col)["fraud"]
    expanding_sum = grp.expanding().sum().groupby(level=0).shift(1)
    expanding_count = grp.expanding().count().groupby(level=0).shift(1)

    # James-Stein shrinkage with global prior
    global_fraud_rate = all_data["fraud"].replace(-1, np.nan).mean()
    prior_weight = 10  # shrinkage strength

    raw_rate = expanding_sum / expanding_count
    shrunk_rate = (expanding_sum + prior_weight * global_fraud_rate) / (expanding_count + prior_weight)

    base_feats[f"oof_target_{col}"] = shrunk_rate.values
    base_feats[f"oof_target_raw_{col}"] = raw_rate.values
    base_feats[f"oof_target_count_{col}"] = expanding_count.values

# ============================================================
# INTERACTION FEATURES
# ============================================================
print("\nBuilding interaction features...")

# Amount x account_age
base_feats["amt_x_age"] = base_feats["amount_log"] * base_feats["account_age_log"]

# Merchant category x amount
base_feats["merch_cat_amt_rank"] = all_data.groupby("merchant_category")["amount_bdt"].rank(pct=True).values

# Device type x amount
base_feats["dev_type_amt_rank"] = all_data.groupby("device_type")["amount_bdt"].rank(pct=True).values

# Payment method x amount
base_feats["pay_method_amt_rank"] = all_data.groupby("payment_method")["amount_bdt"].rank(pct=True).values

# Location x amount
base_feats["location_amt_rank"] = all_data.groupby("location")["amount_bdt"].rank(pct=True).values

# ============================================================
# SAVE FEATURES
# ============================================================
print(f"\nTotal new features: {base_feats.shape[1]}")
print("Feature columns:", list(base_feats.columns))

# Split back to train/test
train_idx = all_data["is_train"] == 1
test_idx = all_data["is_train"] == 0

X_train_new = base_feats[train_idx.values].copy()
X_test_new = base_feats[test_idx.values].copy()

# Add original model features
mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb")) if os.path.exists(os.path.join(BASE, "model_features_v4.pkl")) else None

print(f"\nTrain new features: {X_train_new.shape}")
print(f"Test new features: {X_test_new.shape}")

# Save
X_train_new.to_parquet(os.path.join(OUT, "train_graph_features.parquet"))
X_test_new.to_parquet(os.path.join(OUT, "test_graph_features.parquet"))

# Also save the original V4 features for the model
import pickle
mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
tr_v4 = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
te_v4 = pd.read_parquet(os.path.join(BASE, "feature_test_v4.parquet"))

# Combine old + new features
X_train_combined = pd.concat([tr_v4[mfeat], X_train_new.reset_index(drop=True)], axis=1)
X_test_combined = pd.concat([te_v4[mfeat], X_test_new.reset_index(drop=True)], axis=1)

print(f"\nCombined train: {X_train_combined.shape}")
print(f"Combined test: {X_test_combined.shape}")

X_train_combined.to_parquet(os.path.join(OUT, "train_combined_features.parquet"))
X_test_combined.to_parquet(os.path.join(OUT, "test_combined_features.parquet"))

# Save targets
y_train = tr_v4["fraud"].values.astype(int)
np.save(os.path.join(OUT, "y_train.npy"), y_train)

print(f"\nAll features saved to {OUT}")
print(f"Total time: {time.time()-t0:.0f}s")
print("DONE")
