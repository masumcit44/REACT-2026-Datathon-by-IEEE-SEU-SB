"""
Phase 2: Aggressive approaches to hit 0.59
- Adversarial validation features
- Isolation Forest anomaly scores
- Time-decayed training weights
- Segment-aware features
- Better stacking
"""
import numpy as np
import pandas as pd
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score as ap

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AGGRESSIVE_V2")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# LOAD RAW + V4 DATA
# ============================================================
print("Loading data...")
t0 = time.time()
tr_raw = pd.read_csv(r"train.csv", parse_dates=["timestamp"])
te_raw = pd.read_csv(r"test.csv", parse_dates=["timestamp"])
tr_v4 = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
te_v4 = pd.read_parquet(os.path.join(BASE, "feature_test_v4.parquet"))
y = tr_raw["fraud"].values.astype(int)

mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))

# Load mega features
X_mega_train = pd.read_parquet(os.path.join(BASE, "AGENT_EXP_MEGA_V2", "train_new_features.parquet"))
X_mega_test = pd.read_parquet(os.path.join(BASE, "AGENT_EXP_MEGA_V2", "test_new_features.parquet"))

print(f"Train: {tr_raw.shape}, Test: {te_raw.shape} ({time.time()-t0:.0f}s)")

# ============================================================
# 1. ADVERSARIAL VALIDATION FEATURES
# ============================================================
print("\n[1/5] Adversarial validation...")
t1 = time.time()

# Combine train/test for adversarial
n_tr = len(tr_raw)
n_te = len(te_raw)
adv_y = np.concatenate([np.zeros(n_tr), np.ones(n_te)])
adv_X = pd.concat([tr_v4[mfeat].reset_index(drop=True), te_v4[mfeat].reset_index(drop=True)], axis=1)

# Train adversarial classifier
from sklearn.ensemble import RandomForestClassifier
adv_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=4242, n_jobs=-1)

# Subsample for speed
idx_sub = np.random.RandomState(4242).choice(len(adv_y), min(100000, len(adv_y)), replace=False)
adv_model.fit(adv_X.iloc[idx_sub], adv_y[idx_sub])

# Predict domain probability (how test-like is each row?)
adv_pred_train = adv_model.predict_proba(adv_X.iloc[:n_tr])[:, 1]
adv_pred_test = adv_model.predict_proba(adv_X.iloc[n_tr:])[:, 1]

print(f"  Adversarial train mean: {adv_pred_train.mean():.4f}")
print(f"  Adversarial test mean: {adv_pred_test.mean():.4f}")

# Feature importance for drift
adv_imp = pd.Series(adv_model.feature_importances_, index=mfeat).sort_values(ascending=False)
print(f"  Top drifting features: {list(adv_imp.head(10).index)}")

# Save as features
adv_train_df = pd.DataFrame({"adversarial_domain_prob": adv_pred_train})
adv_test_df = pd.DataFrame({"adversarial_domain_prob": adv_pred_test})

# Add top drift features as ratios
for feat_name in adv_imp.head(20).index:
    tr_vals = tr_v4[feat_name].values
    te_vals = te_v4[feat_name].values
    tr_median = np.nanmedian(tr_vals)
    te_median = np.nanmedian(te_vals)
    if tr_median != 0:
        ratio = te_median / tr_median
    else:
        ratio = 1.0
    # Feature drift magnitude
    adv_train_df[f"drift_{feat_name}"] = tr_vals / (tr_median + 1e-8)
    adv_test_df[f"drift_{feat_name}"] = te_vals / (tr_median + 1e-8)

print(f"  Adversarial features: {adv_train_df.shape[1]} ({time.time()-t1:.0f}s)")

# ============================================================
# 2. ANOMALY SCORE FEATURES
# ============================================================
print("\n[2/5] Anomaly scores...")
t2 = time.time()

# Use V4 numeric features for anomaly detection
num_feats = [c for c in mfeat if c not in cats]
X_num_train = tr_v4[num_feats].fillna(0).astype(np.float32)
X_num_test = te_v4[num_feats].fillna(0).astype(np.float32)

# Isolation Forest
iso = IsolationForest(n_estimators=200, max_samples=0.8, contamination=0.02,
                      random_state=4242, n_jobs=-1)
iso.fit(X_num_train.iloc[:50000])  # subsample for speed

iso_train = -iso.score_samples(X_num_train)  # higher = more anomalous
iso_test = -iso.score_samples(X_num_test)

anom_train = pd.DataFrame({
    "isolation_forest_score": iso_train,
    "iso_rank_train": pd.Series(iso_train).rank(pct=True).values
})
anom_test = pd.DataFrame({
    "isolation_forest_score": iso_test,
    "iso_rank_test": pd.Series(iso_test).rank(pct=True).values
})

print(f"  IF train mean: {iso_train.mean():.4f}, test mean: {iso_test.mean():.4f}")
print(f"  Anomaly features: {anom_train.shape[1]} ({time.time()-t2:.0f}s)")

# ============================================================
# 3. TIME-DECAY WEIGHTS + TEMPORAL FEATURES
# ============================================================
print("\n[3/5] Temporal features...")
t3 = time.time()

# Transaction recency (days from start of training)
ts = tr_raw["timestamp"]
ts_min = ts.min()
days_from_start = (ts - ts_min).dt.total_seconds() / 86400
max_days = days_from_start.max()

# Exponential decay weight (more recent = higher weight)
decay_rate = 0.005  # ~60-day half-life
time_weights = np.exp(-decay_rate * (max_days - days_from_start.values))

temporal_train = pd.DataFrame({
    "days_from_start": days_from_start.values,
    "time_weight_exp": time_weights,
    "time_weight_linear": 1 - days_from_start.values / max_days,
})

ts_test = te_raw["timestamp"]
days_from_start_test = (ts_test - ts_min).dt.total_seconds() / 86400
temporal_test = pd.DataFrame({
    "days_from_start": days_from_start_test.values,
    "time_weight_exp": np.exp(-decay_rate * (max_days - days_from_start_test.values)),
    "time_weight_linear": 1 - days_from_start_test.values / max_days,
})

# Time-based features from raw timestamps
for df, temporal_df in [(tr_raw, temporal_train), (te_raw, temporal_test)]:
    temporal_df["hour"] = df["timestamp"].dt.hour
    temporal_df["dow"] = df["timestamp"].dt.dayofweek
    temporal_df["is_month_start"] = (df["timestamp"].dt.day <= 3).astype(np.int8)
    temporal_df["is_month_end"] = (df["timestamp"].dt.day >= 28).astype(np.int8)

# Days before test period start (2026-07-16)
test_start = pd.Timestamp("2026-07-16")
temporal_train["days_before_test"] = (test_start - tr_raw["timestamp"]).dt.total_seconds() / 86400
temporal_test["days_before_test"] = (test_start - te_raw["timestamp"]).dt.total_seconds() / 86400

print(f"  Temporal features: {temporal_train.shape[1]} ({time.time()-t3:.0f}s)")

# ============================================================
# 4. COMBINE ALL NEW FEATURES
# ============================================================
print("\n[4/5] Combining features...")

# Combine mega + adversarial + anomaly + temporal
X_train_new = pd.concat([
    X_mega_train.reset_index(drop=True),
    adv_train_df.reset_index(drop=True),
    anom_train.reset_index(drop=True),
    temporal_train.reset_index(drop=True),
], axis=1)

X_test_new = pd.concat([
    X_mega_test.reset_index(drop=True),
    adv_test_df.reset_index(drop=True),
    anom_test.reset_index(drop=True),
    temporal_test.reset_index(drop=True),
], axis=1)

# Add V4 features
X_train_all = pd.concat([tr_v4[mfeat].reset_index(drop=True), X_train_new], axis=1)
X_test_all = pd.concat([te_v4[mfeat].reset_index(drop=True), X_test_new], axis=1)

# Categorical handling
for c in cats:
    X_train_all[c] = X_train_all[c].fillna("missing").astype("category").cat.codes
    X_test_all[c] = X_test_all[c].fillna("missing").astype("category").cat.codes

# Fill NaN and cast
for c in X_train_all.columns:
    X_train_all[c] = X_train_all[c].fillna(-999).astype(np.float32)
    X_test_all[c] = X_test_all[c].fillna(-999).astype(np.float32)

print(f"Combined train: {X_train_all.shape}")
print(f"Combined test: {X_test_all.shape}")

# Save
X_train_all.to_parquet(os.path.join(OUT, "train_all_features.parquet"))
X_test_all.to_parquet(os.path.join(OUT, "test_all_features.parquet"))

# ============================================================
# 5. TRAIN MODELS WITH TIME-DECAY WEIGHTS
# ============================================================
print("\n[5/5] Training models...")
N_VAL = len(np.load(os.path.join(BASE, "Fold_3_v4_targets.npy")))
y_f3 = np.load(os.path.join(BASE, "Fold_3_v4_targets.npy"))

Xtr, Xva = X_train_all.iloc[:-N_VAL], X_train_all.iloc[-N_VAL:]
ytr = y[:-N_VAL]
yva = y[-N_VAL:]

# Time-decay weights for training
train_weights = time_weights[:len(ytr)]

# Cat indices
cat_idx = [i for i, c in enumerate(X_train_all.columns) if c in cats]

# --- Model A: LightGBM with time-decay weights ---
print("\n  Model A: LGB + time-decay weights...")
import lightgbm as lgb

params = {
    "boosting_type": "gbdt", "objective": "binary",
    "num_leaves": 127, "max_depth": -1,
    "learning_rate": 0.025, "n_estimators": 2500,
    "subsample_for_bin": 200000,
    "min_child_samples": 100,
    "subsample": 0.85, "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.5, "reg_lambda": 5.0,
    "random_state": 4242, "n_jobs": 8, "verbosity": -1
}

t0 = time.time()
m_lgb_w = lgb.LGBMClassifier(**params)
m_lgb_w.fit(Xtr, ytr, sample_weight=train_weights,
            categorical_feature=cat_idx,
            eval_set=[(Xva, y_f3)],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
pred_lgb_w_va = m_lgb_w.predict_proba(Xva)[:, 1]
ap_lgb_w = ap(y_f3, pred_lgb_w_va)
print(f"    LGB+decay fold-3 AP: {ap_lgb_w:.6f} ({time.time()-t0:.0f}s)")

# Full data
t0 = time.time()
m_lgb_w_full = lgb.LGBMClassifier(**params)
m_lgb_w_full.set_params(n_estimators=m_lgb_w.best_iteration_)
m_lgb_w_full.fit(X_train_all, y, sample_weight=time_weights, categorical_feature=cat_idx)
pred_lgb_w_test = m_lgb_w_full.predict_proba(X_test_all)[:, 1]
print(f"    LGB+decay full: {time.time()-t0:.0f}s")
np.save(os.path.join(OUT, "lgb_decay_test.npy"), pred_lgb_w_test)

# --- Model B: LGB without decay (baseline with new features) ---
print("\n  Model B: LGB baseline (no decay)...")
t0 = time.time()
m_lgb_b = lgb.LGBMClassifier(**params)
m_lgb_b.fit(Xtr, ytr, categorical_feature=cat_idx,
            eval_set=[(Xva, y_f3)],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
pred_lgb_b_va = m_lgb_b.predict_proba(Xva)[:, 1]
ap_lgb_b = ap(y_f3, pred_lgb_b_va)
print(f"    LGB baseline fold-3 AP: {ap_lgb_b:.6f} ({time.time()-t0:.0f}s)")

t0 = time.time()
m_lgb_b_full = lgb.LGBMClassifier(**params)
m_lgb_b_full.set_params(n_estimators=m_lgb_b.best_iteration_)
m_lgb_b_full.fit(X_train_all, y, categorical_feature=cat_idx)
pred_lgb_b_test = m_lgb_b_full.predict_proba(X_test_all)[:, 1]
print(f"    LGB baseline full: {time.time()-t0:.0f}s")
np.save(os.path.join(OUT, "lgb_baseline_test.npy"), pred_lgb_b_test)

# --- Model C: HGB with new features ---
print("\n  Model C: HGB with all features...")
t0 = time.time()
m_hgb = HistGradientBoostingClassifier(
    max_iter=1000, learning_rate=0.03, max_leaf_nodes=127,
    min_samples_leaf=60, l2_regularization=5.0, max_bins=255,
    random_state=4242, early_stopping=False, validation_fraction=None)
m_hgb.fit(Xtr.values, ytr)
pred_hgb_va = m_hgb.predict_proba(Xva.values)[:, 1]
ap_hgb = ap(y_f3, pred_hgb_va)
print(f"    HGB fold-3 AP: {ap_hgb:.6f} ({time.time()-t0:.0f}s)")

t0 = time.time()
m_hgb_full = HistGradientBoostingClassifier(
    max_iter=m_hgb.n_iter_, learning_rate=0.03, max_leaf_nodes=127,
    min_samples_leaf=60, l2_regularization=5.0, max_bins=255,
    random_state=4242, early_stopping=False, validation_fraction=None)
m_hgb_full.fit(X_train_all.values, y)
pred_hgb_test = m_hgb_full.predict_proba(X_test_all.values)[:, 1]
print(f"    HGB full: {time.time()-t0:.0f}s")
np.save(os.path.join(OUT, "hgb_all_test.npy"), pred_hgb_test)

# ============================================================
# 6. BLEND AND SUBMIT
# ============================================================
print("\n" + "="*60)
print("BLEND EVALUATION")
print("="*60)

# Load existing predictions
champ_f3 = np.load(os.path.join(BASE, "EXP_W2B_COST_SENSITIVE/best_extended_cost_global_fold3_pred.npy"))
hgb_old_f3 = np.load(os.path.join(BASE, "AGENT_EXP_AUTO_HGB", "HGB1_pred.npy"))
lgb_mega_f3 = np.load(os.path.join(BASE, "AGENT_EXP_MEGA_V2", "lgb_mega_fold3_pred.npy"))

print("All fold-3 APs:")
models_f3 = {
    "Champion": champ_f3, "HGB_old": hgb_old_f3, "LGB_mega": lgb_mega_f3,
    "LGB_decay": pred_lgb_w_va, "LGB_baseline": pred_lgb_b_va, "HGB_all": pred_hgb_va
}
for n, p in models_f3.items():
    print(f"  {n}: {ap(y_f3, p):.6f}")

# Best blend search
best = (0, None)
model_list = list(models_f3.items())
for i in range(len(model_list)):
    for j in range(i+1, len(model_list)):
        for k in range(j+1, len(model_list)):
            for w1 in np.arange(0.15, 0.55, 0.05):
                for w2 in np.arange(0.1, 0.5, 0.05):
                    w3 = 1 - w1 - w2
                    if w3 < 0.05: continue
                    blend = w1*model_list[i][1] + w2*model_list[j][1] + w3*model_list[k][1]
                    s = ap(y_f3, blend)
                    if s > best[0]:
                        best = (s, (model_list[i][0], model_list[j][0], model_list[k][0]), (round(w1,2), round(w2,2), round(w3,2)))

print(f"\nBest 3-way: AP={best[0]:.6f} models={best[1]} weights={best[2]}")
print(f"Gain: {best[0]-0.518226:+.6f}")

# 4-way best
best4 = (0, None)
for i in range(len(model_list)):
    for j in range(i+1, len(model_list)):
        for k in range(j+1, len(model_list)):
            for l_idx in range(k+1, len(model_list)):
                for w1 in np.arange(0.1, 0.4, 0.05):
                    for w2 in np.arange(0.1, 0.4, 0.05):
                        for w3 in np.arange(0.1, 0.3, 0.05):
                            w4 = 1 - w1 - w2 - w3
                            if w4 < 0.05: continue
                            blend = w1*model_list[i][1] + w2*model_list[j][1] + w3*model_list[k][1] + w4*model_list[l_idx][1]
                            s = ap(y_f3, blend)
                            if s > best4[0]:
                                best4 = (s, [model_list[i][0], model_list[j][0], model_list[k][0], model_list[l_idx][0]], [round(w1,2), round(w2,2), round(w3,2), round(w4,2)])

print(f"Best 4-way: AP={best4[0]:.6f} models={best4[1]} weights={best4[2]}")
print(f"Gain: {best4[0]-0.518226:+.6f}")

# ============================================================
# 7. SAVE SUBMISSIONS
# ============================================================
print("\nSaving submissions...")
ss = pd.read_csv(r"sample_submission.csv")

champ_test = np.load(os.path.join(BASE, "FINAL_0518226_SUBMISSION", "final_nested_0518226_test_pred.npy"))
hgb_old_test = np.load(os.path.join(BASE, "AGENT_EXP_AUTO_HGB_VERIFY", "HGB_tuned_fulldata_test_pred.npy"))
lgb_mega_test = np.load(os.path.join(BASE, "AGENT_EXP_MEGA_V2", "lgb_mega_test_pred.npy"))

all_test = {
    "Champion": champ_test, "HGB_old": hgb_old_test, "LGB_mega": lgb_mega_test,
    "LGB_decay": pred_lgb_w_test, "LGB_baseline": pred_lgb_b_test, "HGB_all": pred_hgb_test
}

# Save best 3-way
w = best[2]
m = best[1]
blend_3way = w[0]*all_test[m[0]] + w[1]*all_test[m[1]] + w[2]*all_test[m[2]]
sub = ss.copy(); sub["fraud"] = blend_3way
sub.to_csv(os.path.join(OUT, "submission_best3way.csv"), index=False)
sub.to_csv(r"submission_file/aggressive_best3way_submission.csv", index=False)
print(f"  Best 3-way: mean={blend_3way.mean():.6f}")

# Save best 4-way
w4 = best4[2]; m4 = best4[1]
blend_4way = w4[0]*all_test[m4[0]] + w4[1]*all_test[m4[1]] + w4[2]*all_test[m4[2]] + w4[3]*all_test[m4[3]]
sub = ss.copy(); sub["fraud"] = blend_4way
sub.to_csv(os.path.join(OUT, "submission_best4way.csv"), index=False)
sub.to_csv(r"submission_file/aggressive_best4way_submission.csv", index=False)
print(f"  Best 4-way: mean={blend_4way.mean():.6f}")

# Save LGB decay solo (might be better if time-decay helps on test)
sub = ss.copy(); sub["fraud"] = pred_lgb_w_test
sub.to_csv(r"submission_file/aggressive_lgb_decay_submission.csv", index=False)
print(f"  LGB decay solo: mean={pred_lgb_w_test.mean():.6f}")

# Save LGB baseline solo
sub = ss.copy(); sub["fraud"] = pred_lgb_b_test
sub.to_csv(r"submission_file/aggressive_lgb_baseline_submission.csv", index=False)
print(f"  LGB baseline solo: mean={pred_lgb_b_test.mean():.6f}")

# Equal weight all 6
blend_equal = sum(all_test.values()) / 6
sub = ss.copy(); sub["fraud"] = blend_equal
sub.to_csv(r"submission_file/aggressive_equal6_submission.csv", index=False)
print(f"  Equal 6-way: mean={blend_equal.mean():.6f}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("DONE")
