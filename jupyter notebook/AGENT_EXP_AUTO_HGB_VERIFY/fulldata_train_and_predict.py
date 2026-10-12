import numpy as np, pandas as pd, os, pickle, time, warnings, json
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as ap

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_HGB_VERIFY")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
t0 = time.time()

mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))

tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
te = pd.read_parquet(os.path.join(BASE, "feature_test_v4.parquet"))

print(f"Train: {tr.shape}, Test: {te.shape}")
print(f"Load time: {time.time()-t0:.0f}s")

# ============================================================
# 2. PREPARE FEATURES
# ============================================================
print("\nPreparing features...")

y_full = tr["fraud"].values.astype(int)
X_train = tr[mfeat].copy()
X_test = te[mfeat].copy()

# Categorical encoding (same as HGB training)
for c in cats:
    X_train[c] = X_train[c].fillna("missing").astype("category").cat.codes
    X_test[c] = X_test[c].fillna("missing").astype("category").cat.codes

# NA handling (training-only stats)
num_carriers = [c for c in X_train.columns if c not in cats]
na_frac = X_train[num_carriers].isna().mean()
na_cols = [c for c in num_carriers if na_frac[c] > 0]
fills = X_train[num_carriers].mean().to_dict()

print(f"NA columns ({len(na_cols)}): {na_cols[:5]}...")

# Add NA indicators to train and test
for df in (X_train, X_test):
    for c in na_cols:
        df[c + "_na"] = df[c].isna().astype(np.int8)
        df[c] = df[c].fillna(fills[c])

# Identify low-cardinality categorical features
low_cat = {c for c in cats if X_train[c].nunique() <= 250}
cat_mask = np.array([c in low_cat for c in X_train.columns])
print(f"Low-card categorical features ({len(low_cat)}): {sorted(low_cat)}")
print(f"Total features: {X_train.shape[1]}")

# Cast to float32 for memory efficiency
for c in X_train.columns:
    X_train[c] = X_train[c].astype(np.float32)
    X_test[c] = X_test[c].astype(np.float32)

print(f"Memory usage: train ~{X_train.memory_usage(deep=True).sum()/1e9:.2f} GB")

# ============================================================
# 3. TRAIN HGB ON FULL DATA
# ============================================================
print("\nTraining HGB_tuned on FULL training data...")
print("Parameters: max_iter=1000, lr=0.03, leaves=127, min_leaf=60, l2=5.0")

m = HGB(
    max_iter=1000, learning_rate=0.03, max_leaf_nodes=127,
    min_samples_leaf=60, l2_regularization=5.0, max_bins=255,
    random_state=4242, early_stopping=False,
    validation_fraction=None,
    categorical_features=cat_mask.tolist()
)

t0 = time.time()
m.fit(X_train, y_full)
train_time = time.time() - t0
print(f"Training complete in {train_time:.0f}s")
print(f"Best iteration: {m.n_iter_}")

# Save model
import joblib
model_path = os.path.join(OUT, "HGB_tuned_fulldata_model.joblib")
joblib.dump(m, model_path)
print(f"Model saved: {model_path}")

# ============================================================
# 4. GENERATE TEST PREDICTIONS
# ============================================================
print("\nGenerating test predictions...")
t0 = time.time()
test_pred = m.predict_proba(X_test)[:, 1]
pred_time = time.time() - t0
print(f"Test predictions: N={len(test_pred)}, time={pred_time:.0f}s")
print(f"Prediction stats: min={test_pred.min():.6f}, max={test_pred.max():.6f}, mean={test_pred.mean():.6f}")

# Save HGB test predictions
hgb_test_path = os.path.join(OUT, "HGB_tuned_fulldata_test_pred.npy")
np.save(hgb_test_path, test_pred)
print(f"Saved: {hgb_test_path}")

# ============================================================
# 5. VERIFY FOLD-3 PREDICTIONS STILL CORRECT
# ============================================================
print("\nVerifying fold-3 model consistency...")
# Retrain on fold-3 training split (timestamp < 2026-07-01) for verification
tr_fold3 = tr.copy()
tr_fold3["ts"] = pd.to_datetime(tr_fold3["timestamp"])
f3_train_mask = tr_fold3["ts"] < "2026-07-01"
Xtr3 = X_train[f3_train_mask.values].copy()
ytr3 = y_full[f3_train_mask.values]

m3 = HGB(
    max_iter=1000, learning_rate=0.03, max_leaf_nodes=127,
    min_samples_leaf=60, l2_regularization=5.0, max_bins=255,
    random_state=4242, early_stopping=False,
    validation_fraction=None,
    categorical_features=cat_mask.tolist()
)
m3.fit(Xtr3, ytr3)

# Predict on fold-3 validation
Xva3 = X_train[~f3_train_mask.values].copy()
pred3 = m3.predict_proba(Xva3)[:, 1]
y3 = np.load(os.path.join(BASE, "Fold_3_v4_targets.npy"))
ap3 = ap(y3, pred3)
print(f"Fold-3 retrain AP: {ap3:.6f} (expected ~0.518915)")

# ============================================================
# 6. BLEND WITH CHAMPION
# ============================================================
print("\nBlending with champion test predictions...")

champion_test = np.load(os.path.join(BASE, "FINAL_0518226_SUBMISSION", "final_nested_0518226_test_pred.npy"))
print(f"Champion test pred: N={len(champion_test)}")
assert len(champion_test) == len(test_pred), "Length mismatch!"

# Load the other ensemble members for 3-way blend
acc_test = None
for p in [os.path.join(BASE, "EXP_B25_ACCELERATION_TUNED", "best_acceleration_global_test_pred.npy"),
          os.path.join(BASE, "EXP_B25_ACCELERATION", "acceleration_test_pred.npy")]:
    if os.path.exists(p):
        acc_test = np.load(p)
        print(f"Acceleration test pred: N={len(acc_test)}")
        break

print("\n--- Blend sweep (test predictions) ---")
results = []

# 2-way: G + HGBt
for w_pct in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    w = w_pct / 100.0
    bp = (1 - w) * champion_test + w * test_pred
    results.append({"blend": f"G_HGBt_w{w_pct}", "pred": bp})
    print(f"  G+HGBt@{w_pct:2d}%: N={len(bp)}, mean={bp.mean():.6f}")

# 3-way: G + HGBt + ACC
if acc_test is not None and len(acc_test) == len(test_pred):
    for wg, wh, wa in [(0.50, 0.40, 0.10), (0.55, 0.30, 0.15), (0.60, 0.25, 0.15)]:
        bp = wg * champion_test + wh * test_pred + wa * acc_test
        results.append({"blend": f"3way_G{int(wg*100)}_HGB{int(wh*100)}_ACC{int(wa*100)}", "pred": bp})
        print(f"  3way G={wg:.0%} HGB={wh:.0%} ACC={wa:.0%}: N={len(bp)}, mean={bp.mean():.6f}")

# ============================================================
# 7. SAVE ALL ARTIFACTS
# ============================================================
print("\n--- Saving artifacts ---")

# Save the recommended blend (G+HGBt@45%)
w_rec = 0.45
blend_rec = (1 - w_rec) * champion_test + w_rec * test_pred
rec_path = os.path.join(OUT, "HGB_blend_45pct_test_pred.npy")
np.save(rec_path, blend_rec)
print(f"Recommended blend (45% HGB): {rec_path}")

# Save the best 3-way blend
if acc_test is not None:
    blend_3way = 0.50 * champion_test + 0.40 * test_pred + 0.10 * acc_test
    path_3way = os.path.join(OUT, "HGB_3way_50_40_10_test_pred.npy")
    np.save(path_3way, blend_3way)
    print(f"3-way blend (50/40/10): {path_3way}")

# Save all blend results
blend_summary = {
    "hgb_individual_test_mean": float(test_pred.mean()),
    "champion_test_mean": float(champion_test.mean()),
    "recommended_blend": "G+HGBt@45%",
    "recommended_blend_mean": float(blend_rec.mean()),
    "training_rows": int(len(y_full)),
    "training_fraud_rate": float(y_full.mean()),
    "model_params": {
        "max_iter": 1000, "learning_rate": 0.03, "max_leaf_nodes": 127,
        "min_samples_leaf": 60, "l2_regularization": 5.0, "max_bins": 255,
        "random_state": 4242
    },
    "features": int(X_train.shape[1]),
    "na_indicator_features": len(na_cols),
    "train_time_s": float(train_time),
    "fold3_verification_ap": float(ap3)
}

with open(os.path.join(OUT, "fulldata_results.json"), "w") as f:
    json.dump(blend_summary, f, indent=2)
print(f"Saved: fulldata_results.json")

# ============================================================
# 8. VERIFY SAVED ARTIFACTS
# ============================================================
print("\n--- Verification ---")
for name, path in [("HGB test", hgb_test_path), ("Recommended blend", rec_path)]:
    p = np.load(path)
    assert len(p) == 262648, f"Wrong length for {name}: {len(p)}"
    assert np.all(np.isfinite(p)), f"Non-finite values in {name}"
    assert p.min() >= 0 and p.max() <= 1, f"Out of range in {name}"
    print(f"  {name}: OK (N={len(p)}, [{p.min():.4f}, {p.max():.4f}], mean={p.mean():.6f})")

print(f"\nAll artifacts saved to: {OUT}")
print(f"Total time: {time.time()-t0:.0f}s")
print("\nDONE")
