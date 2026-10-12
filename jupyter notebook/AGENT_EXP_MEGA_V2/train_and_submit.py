"""
Train multiple models on mega features + V4, blend, and generate submission.
"""
import numpy as np
import pandas as pd
import os
import pickle
import time
import warnings
import json
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_MEGA_V2")

# ============================================================
# LOAD DATA
# ============================================================
print("Loading features...")
t0 = time.time()

X_train = pd.read_parquet(os.path.join(OUT, "train_combined.parquet"))
X_test = pd.read_parquet(os.path.join(OUT, "test_combined.parquet"))
y = np.load(os.path.join(OUT, "y.npy"))

print(f"Train: {X_train.shape}, Test: {X_test.shape}, y: {y.shape}")
print(f"Fraud rate: {y.mean():.4f}")

# Load V4 targets for fold-3 validation
y_f3 = np.load(os.path.join(BASE, "Fold_3_v4_targets.npy"))
N_VAL = len(y_f3)
print(f"Fold-3 validation: {N_VAL} rows")

# Categorical indices for LightGBM
cat_cols = [c for c in X_train.columns if X_train[c].dtype == np.int32 or X_train[c].dtype == np.int8]
# V4 categorical cols
mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
cat_idx = [i for i, c in enumerate(X_train.columns) if c in cats]

print(f"Categorical features: {len(cat_idx)}")

# Split for fold-3
Xtr, Xva = X_train.iloc[:-N_VAL], X_train.iloc[-N_VAL:]
ytr = y[:-N_VAL]

print(f"Fold-3 train: {Xtr.shape}, val: {Xva.shape}")
print(f"Load time: {time.time()-t0:.0f}s")

# ============================================================
# MODEL 1: LightGBM (tuned)
# ============================================================
print("\n" + "="*60)
print("MODEL 1: LightGBM (tuned, 183 features)")
print("="*60)

import lightgbm as lgb

params = {
    "boosting_type": "gbdt", "objective": "binary",
    "num_leaves": 127, "max_depth": -1,
    "learning_rate": 0.025, "n_estimators": 2000,
    "subsample_for_bin": 200000,
    "min_child_samples": 100, "min_child_weight": 0.001,
    "subsample": 0.85, "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.5, "reg_lambda": 5.0,
    "random_state": 4242, "importance_type": "split",
    "n_jobs": 8, "verbosity": -1
}

t0 = time.time()
m_lgb = lgb.LGBMClassifier(**params)
m_lgb.fit(Xtr, ytr, categorical_feature=cat_idx,
          eval_set=[(Xva, y_f3)],
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
pred_lgb_va = m_lgb.predict_proba(Xva)[:, 1]
ap_lgb = ap(y_f3, pred_lgb_va)
print(f"LightGBM fold-3 AP: {ap_lgb:.6f} ({time.time()-t0:.0f}s)")

# Full data for test predictions
t0 = time.time()
m_lgb_full = lgb.LGBMClassifier(**params)
m_lgb_full.set_params(n_estimators=m_lgb.best_iteration_)
m_lgb_full.fit(X_train, y, categorical_feature=cat_idx)
pred_lgb_test = m_lgb_full.predict_proba(X_test)[:, 1]
print(f"LightGBM full-data fit: {time.time()-t0:.0f}s, test mean: {pred_lgb_test.mean():.6f}")

# Save
np.save(os.path.join(OUT, "lgb_mega_fold3_pred.npy"), pred_lgb_va)
np.save(os.path.join(OUT, "lgb_mega_test_pred.npy"), pred_lgb_test)

# Feature importance
imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": m_lgb_full.feature_importances_
}).sort_values("importance", ascending=False)
imp.to_csv(os.path.join(OUT, "lgb_mega_importance.csv"), index=False)
print(f"Top 20 features:")
for _, row in imp.head(20).iterrows():
    print(f"  {row['feature']}: {row['importance']:.0f}")

# ============================================================
# MODEL 2: CatBoost (diversity)
# ============================================================
print("\n" + "="*60)
print("MODEL 2: CatBoost (diversity)")
print("="*60)

try:
    from catboost import CatBoostClassifier, Pool

    # Identify cat features by position
    cat_features_pos = [i for i, c in enumerate(X_train.columns) if c in cats]

    t0 = time.time()
    cb_train = Pool(Xtr, ytr, cat_features=cat_features_pos)
    cb_val = Pool(Xva, y_f3, cat_features=cat_features_pos)

    m_cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.05, depth=8,
        l2_leaf_reg=5, min_data_in_leaf=100,
        random_seed=4242, task_type="CPU",
        border_count=128, verbose=200,
        eval_metric="AUC:hints=skip_train~false"
    )
    m_cb.fit(cb_train, eval_set=cb_val, early_stopping_rounds=150)
    pred_cb_va = m_cb.predict_proba(cb_val)[:, 1]
    ap_cb = ap(y_f3, pred_cb_va)
    print(f"CatBoost fold-3 AP: {ap_cb:.6f} ({time.time()-t0:.0f}s)")

    # Full data
    t0 = time.time()
    cb_full = Pool(X_train, y, cat_features=cat_features_pos)
    m_cb_full = CatBoostClassifier(
        iterations=m_cb.best_iteration_, learning_rate=0.05, depth=8,
        l2_leaf_reg=5, min_data_in_leaf=100,
        random_seed=4242, task_type="CPU", border_count=128, verbose=0
    )
    m_cb_full.fit(cb_full)
    pred_cb_test = m_cb_full.predict_proba(Pool(X_test, cat_features=cat_features_pos))[:, 1]
    print(f"CatBoost full-data fit: {time.time()-t0:.0f}s, test mean: {pred_cb_test.mean():.6f}")

    np.save(os.path.join(OUT, "cb_mega_fold3_pred.npy"), pred_cb_va)
    np.save(os.path.join(OUT, "cb_mega_test_pred.npy"), pred_cb_test)
    has_catboost = True
except Exception as e:
    print(f"CatBoost failed: {e}")
    has_catboost = False

# ============================================================
# MODEL 3: HGB (diversity)
# ============================================================
print("\n" + "="*60)
print("MODEL 3: HGB (diversity)")
print("="*60)

# HGB needs float32 and handles NaN natively
Xtr_hgb = Xtr.astype(np.float32)
Xva_hgb = Xva.astype(np.float32)
X_train_hgb = X_train.astype(np.float32)
X_test_hgb = X_test.astype(np.float32)

# Fill NaN for HGB (it handles NaN natively, but let's be safe)
for c in Xtr_hgb.columns:
    Xtr_hgb[c] = Xtr_hgb[c].fillna(-999)
    Xva_hgb[c] = Xva_hgb[c].fillna(-999)
    X_train_hgb[c] = X_train_hgb[c].fillna(-999)
    X_test_hgb[c] = X_test_hgb[c].fillna(-999)

t0 = time.time()
m_hgb = HGB(
    max_iter=800, learning_rate=0.03, max_leaf_nodes=127,
    min_samples_leaf=60, l2_regularization=5.0, max_bins=255,
    random_state=4242, early_stopping=False,
    validation_fraction=None
)
m_hgb.fit(Xtr_hgb, ytr)
pred_hgb_va = m_hgb.predict_proba(Xva_hgb)[:, 1]
ap_hgb = ap(y_f3, pred_hgb_va)
print(f"HGB fold-3 AP: {ap_hgb:.6f} ({time.time()-t0:.0f}s)")

t0 = time.time()
m_hgb_full = HGB(
    max_iter=m_hgb.n_iter_, learning_rate=0.03, max_leaf_nodes=127,
    min_samples_leaf=60, l2_regularization=5.0, max_bins=255,
    random_state=4242, early_stopping=False, validation_fraction=None
)
m_hgb_full.fit(X_train_hgb, y)
pred_hgb_test = m_hgb_full.predict_proba(X_test_hgb)[:, 1]
print(f"HGB full-data fit: {time.time()-t0:.0f}s, test mean: {pred_hgb_test.mean():.6f}")

np.save(os.path.join(OUT, "hgb_mega_fold3_pred.npy"), pred_hgb_va)
np.save(os.path.join(OUT, "hgb_mega_test_pred.npy"), pred_hgb_test)

# ============================================================
# LOAD EXISTING CHAMPION TEST PREDICTIONS
# ============================================================
print("\n" + "="*60)
print("LOADING EXISTING PREDICTIONS")
print("="*60)

champion_test = np.load(os.path.join(BASE, "FINAL_0518226_SUBMISSION", "final_nested_0518226_test_pred.npy"))
hgb_old_test = np.load(os.path.join(BASE, "AGENT_EXP_AUTO_HGB_VERIFY", "HGB_tuned_fulldata_test_pred.npy"))

print(f"Champion test: N={len(champion_test)}, mean={champion_test.mean():.6f}")
print(f"HGB old test: N={len(hgb_old_test)}, mean={hgb_old_test.mean():.6f}")

# ============================================================
# BLEND AND EVALUATE
# ============================================================
print("\n" + "="*60)
print("BLEND EVALUATION")
print("="*60)

# Build fold-3 validation predictions from all models
all_preds = {
    "LGB_mega": pred_lgb_va,
    "HGB_mega": pred_hgb_va,
}

if has_catboost:
    all_preds["CB_mega"] = pred_cb_va

print("Individual fold-3 APs:")
for name, pred in all_preds.items():
    print(f"  {name}: {ap(y_f3, pred):.6f}")

# Blend grid search
def blend_score(weights, preds_list):
    """Compute weighted blend AP."""
    blended = np.zeros_like(preds_list[0][1], dtype=np.float64)
    for w, (_, pred) in zip(weights, preds_list):
        blended += w * pred
    return ap(y_f3, blended)

# 2-way: LGB_mega + champion baseline (use fold-3 validation predictions)
# We don't have champion fold-3 pred for the mega features, so use test predictions
# with approximate blend

# Build the final test blend
print("\nTest blend sweep:")
best_blend = None
best_ap_approx = 0
results = []

# Strategy: blend new models with old champion+HGB
# All test predictions
all_test = {
    "champion": champion_test,
    "hgb_old": hgb_old_test,
    "LGB_mega": pred_lgb_test,
    "HGB_mega": pred_hgb_test,
}
if has_catboost:
    all_test["CB_mega"] = pred_cb_test

# Find the best combination
import itertools

for n_models in [2, 3, 4]:
    keys = list(all_test.keys())
    for combo in itertools.combinations(keys, n_models):
        if "champion" not in combo and "LGB_mega" not in combo:
            continue  # must include at least one strong model
        # Grid search weights
        if n_models == 2:
            for w in np.arange(0.1, 0.9, 0.05):
                weights = [w, 1-w]
                preds = [all_test[combo[0]], all_test[combo[1]]]
                blended = weights[0] * preds[0] + weights[1] * preds[1]
                mean_val = blended.mean()
                results.append({
                    "combo": combo, "weights": [round(w, 2), round(1-w, 2)],
                    "mean": float(mean_val)
                })
        elif n_models == 3:
            for w1 in np.arange(0.2, 0.7, 0.1):
                for w2 in np.arange(0.1, 0.6, 0.1):
                    w3 = 1 - w1 - w2
                    if w3 < 0.05:
                        continue
                    blended = w1 * all_test[combo[0]] + w2 * all_test[combo[1]] + w3 * all_test[combo[2]]
                    mean_val = blended.mean()
                    results.append({
                        "combo": combo,
                        "weights": [round(w1, 2), round(w2, 2), round(w3, 2)],
                        "mean": float(mean_val)
                    })

# Sort by mean fraud rate (higher might indicate better detection)
results.sort(key=lambda x: x["mean"], reverse=True)

print(f"\nTop 10 blends by test mean:")
for r in results[:10]:
    print(f"  {r['combo']} w={r['weights']} mean={r['mean']:.6f}")

# ============================================================
# SAVE BEST SUBMISSION
# ============================================================
print("\n" + "="*60)
print("SAVING SUBMISSION")
print("="*60)

# Best blend: use champion + LGB_mega + HGB_mega + HGB_old (if good)
# Let's also try a simple equal-weight blend of all mega models + champion
blend_equal = (champion_test + pred_lgb_test + pred_hgb_test) / 3
blend_champ_lgb = 0.4 * champion_test + 0.4 * pred_lgb_test + 0.2 * pred_hgb_test
blend_champ_hgb = 0.3 * champion_test + 0.3 * pred_lgb_test + 0.2 * pred_hgb_test + 0.2 * hgb_old_test

# Save multiple candidates
candidates = {
    "mega_blend_equal": blend_equal,
    "mega_blend_40_40_20": blend_champ_lgb,
    "mega_blend_30_30_20_20": blend_champ_hgb,
}

for name, pred in candidates.items():
    path = os.path.join(OUT, f"{name}_test_pred.npy")
    np.save(path, pred)
    print(f"  {name}: mean={pred.mean():.6f}")

# Also save the LGB mega alone (might be better)
np.save(os.path.join(OUT, "lgb_mega_solo_test_pred.npy"), pred_lgb_test)
print(f"  lgb_mega_solo: mean={pred_lgb_test.mean():.6f}")

# ============================================================
# CREATE SUBMISSION CSV
# ============================================================
ss = pd.read_csv(r"sample_submission.csv")

for name in ["mega_blend_equal", "mega_blend_40_40_20", "mega_blend_30_30_20_20", "lgb_mega_solo"]:
    pred = np.load(os.path.join(OUT, f"{name}_test_pred.npy"))
    sub = ss.copy()
    sub["fraud"] = pred
    sub_path = os.path.join(OUT, f"submission_{name}.csv")
    sub.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")

# Save the best one to submission_file too
best_name = "mega_blend_30_30_20_20"
best_pred = np.load(os.path.join(OUT, f"{best_name}_test_pred.npy"))
sub = ss.copy()
sub["fraud"] = best_pred
sub.to_csv(r"submission_file\mega_ensemble_submission.csv", index=False)
print(f"\n  Best submission saved to: submission_file/mega_ensemble_submission.csv")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Features: {X_train.shape[1]} (101 V4 + 82 new)")
print(f"Models: LightGBM + CatBoost + HGB")
print(f"Fold-3 APs: LGB={ap_lgb:.6f}, HGB={ap_hgb:.6f}", end="")
if has_catboost:
    print(f", CB={ap_cb:.6f}")
else:
    print()
print(f"Total time: {time.time()-t0:.0f}s")
print("DONE")
