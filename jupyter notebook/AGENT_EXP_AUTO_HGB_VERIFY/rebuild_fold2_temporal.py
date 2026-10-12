import numpy as np, pandas as pd, os, pickle, time, warnings, json
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as ap

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_HGB_VERIFY")

# Load data with timestamps
tr_full = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
tr_full['ts'] = pd.to_datetime(tr_full['timestamp'])

mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))

# Fold-2 temporal split: train < 2026-06-16, val = 2026-06-16 to 2026-07-01
train_mask = tr_full['ts'] < '2026-06-16'
val_mask = (tr_full['ts'] >= '2026-06-16') & (tr_full['ts'] < '2026-07-01')
print(f"Fold-2 temporal: train={train_mask.sum()}, val={val_mask.sum()}")

y_full = tr_full['fraud'].values.astype(int)
ytr2 = y_full[train_mask.values]
yva2 = y_full[val_mask.values]

# Verify against stored targets
f2_targets = np.load(os.path.join(BASE, "EXP_B25_ACCELERATION_FOLD2", "fold2_targets.npy"))
assert np.array_equal(yva2.astype(np.int8), f2_targets.astype(np.int8)), "Fold-2 targets mismatch!"
print(f"Fold-2 targets verified: N={len(f2_targets)}, fraud_rate={f2_targets.mean():.5f}")

# Prepare features (same pipeline as hgb_v4.py)
X = tr_full[mfeat].copy()
for c in cats:
    X[c] = X[c].fillna("missing").astype("category").cat.codes

Xtr2 = X[train_mask.values].copy()
Xva2 = X[val_mask.values].copy()

# NA handling (training-only stats)
num_carriers = [c for c in Xtr2.columns if c not in cats]
na_frac = Xtr2[num_carriers].isna().mean()
na_cols = [c for c in num_carriers if na_frac[c] > 0]
fills = Xtr2[num_carriers].mean().to_dict()

for df in (Xtr2, Xva2):
    for c in na_cols:
        df[c + "_na"] = df[c].isna().astype(np.int8)
        df[c] = df[c].fillna(fills[c])

low_cat = {c for c in cats if Xtr2[c].nunique() <= 250}
cat_mask = np.array([c in low_cat for c in Xtr2.columns])
for c in Xtr2.columns:
    Xtr2[c] = Xtr2[c].astype(np.float32)
    Xva2[c] = Xva2[c].astype(np.float32)

print(f"Features: {Xtr2.shape[1]}, low-card cats: {sorted(low_cat)}")

# Train HGB on fold-2 temporal split
m = HGB(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100,
        l2_regularization=1.0, max_bins=255, random_state=4242,
        early_stopping=False, validation_fraction=None, categorical_features=cat_mask.tolist())
t0 = time.time()
m.fit(Xtr2, ytr2)
pred2 = m.predict_proba(Xva2)[:, 1]
elapsed = time.time() - t0

ap_f2 = ap(f2_targets, pred2)
print(f"\nHGB600 fold-2 (temporal): AP={ap_f2:.6f}  time={elapsed:.0f}s")

# Load fold-2 baselines
G2 = np.load(os.path.join(BASE, "EXP_B25_ACCELERATION_FOLD2", "fold2_matched_tuned_v4_baseline_pred.npy"))
A2 = np.load(os.path.join(BASE, "EXP_B25_ACCELERATION_FOLD2", "fold2_tuned_acceleration_pred.npy"))
ap_g2 = ap(f2_targets, G2)
ap_a2 = ap(f2_targets, A2)

print(f"\nFold-2 baselines:")
print(f"  G2 (tuned V4 baseline): AP={ap_g2:.6f}")
print(f"  A2 (tuned acceleration): AP={ap_a2:.6f}")
print(f"  HGB600 (temporal):       AP={ap_f2:.6f}")

gain_hgb_g2 = ap_f2 - ap_g2
print(f"\nHGB600 fold-2 gain vs G2: {gain_hgb_g2:+.6f}")
print(f"Stability gate (gain >= -0.001): {'PASS' if gain_hgb_g2 >= -0.001 else 'FAIL'}")

# Blend fold-2: G2 + HGB
print(f"\nFold-2 blend G2 + HGB:")
for w_pct in [10, 20, 30, 40, 50]:
    w = w_pct / 100.0
    bp = (1 - w) * G2 + w * pred2
    score = ap(f2_targets, bp)
    print(f"  G2+HGB@{w_pct}%: AP={score:.6f}  gain vs G2: {score - ap_g2:+.6f}")

# Save
np.save(os.path.join(OUT, "HGB600_fold2_temporal_pred.npy"), pred2)
results = {
    "HGB600_f2_temporal_ap": float(ap_f2),
    "G2_baseline_ap": float(ap_g2),
    "A2_baseline_ap": float(ap_a2),
    "HGB600_f2_gain_vs_G2": float(gain_hgb_g2),
    "HGB600_f2_stable": bool(gain_hgb_g2 >= -0.001),
    "f2_train_rows": int(train_mask.sum()),
    "f2_val_rows": int(val_mask.sum()),
    "runtime_s": float(elapsed)
}
with open(os.path.join(OUT, "fold2_temporal_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {os.path.join(OUT, 'fold2_temporal_results.json')}")
print("DONE")
