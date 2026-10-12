"""Quick HGB on mega features + final submission build."""
import numpy as np, pandas as pd, os, pickle, time, warnings, json
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as ap

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_MEGA_V2")

# Load
X_train = pd.read_parquet(os.path.join(OUT, "train_combined.parquet"))
X_test = pd.read_parquet(os.path.join(OUT, "test_combined.parquet"))
y = np.load(os.path.join(OUT, "y.npy"))
y_f3 = np.load(os.path.join(BASE, "Fold_3_v4_targets.npy"))
N_VAL = len(y_f3)

print(f"Train: {X_train.shape}, y: {y.mean():.4f}")

# Prepare
Xtr, Xva = X_train.iloc[:-N_VAL].copy(), X_train.iloc[-N_VAL:].copy()
ytr = y[:-N_VAL]

# Fill NaN
for c in Xtr.columns:
    Xtr[c] = Xtr[c].fillna(-999).astype(np.float32)
    Xva[c] = Xva[c].fillna(-999).astype(np.float32)

# Train HGB
print("Training HGB on mega features...")
t0 = time.time()
m = HGB(max_iter=800, learning_rate=0.03, max_leaf_nodes=127,
        min_samples_leaf=60, l2_regularization=5.0, max_bins=255,
        random_state=4242, early_stopping=False, validation_fraction=None)
m.fit(Xtr, ytr)
pred_va = m.predict_proba(Xva)[:, 1]
print(f"HGB mega fold-3 AP: {ap(y_f3, pred_va):.6f} ({time.time()-t0:.0f}s)")

# Full data
print("Training on full data...")
t0 = time.time()
for c in X_train.columns:
    X_train[c] = X_train[c].fillna(-999).astype(np.float32)
    X_test[c] = X_test[c].fillna(-999).astype(np.float32)

m_full = HGB(max_iter=m.n_iter_, learning_rate=0.03, max_leaf_nodes=127,
             min_samples_leaf=60, l2_regularization=5.0, max_bins=255,
             random_state=4242, early_stopping=False, validation_fraction=None)
m_full.fit(X_train, y)
pred_test = m_full.predict_proba(X_test)[:, 1]
print(f"HGB mega full-data: {time.time()-t0:.0f}s, test mean: {pred_test.mean():.6f}")

np.save(os.path.join(OUT, "hgb_mega_fold3_pred.npy"), pred_va)
np.save(os.path.join(OUT, "hgb_mega_test_pred.npy"), pred_test)

# ============================================================
# FINAL ENSEMBLE EVALUATION
# ============================================================
print("\n" + "="*60)
print("FINAL ENSEMBLE")
print("="*60)

# All fold-3 validation predictions
champ_f3 = np.load(os.path.join(BASE, "EXP_W2B_COST_SENSITIVE/best_extended_cost_global_fold3_pred.npy"))
lgb_f3 = np.load(os.path.join(OUT, "lgb_mega_fold3_pred.npy"))
cb_f3 = np.load(os.path.join(OUT, "cb_mega_fold3_pred.npy"))
hgb_f3 = pred_va

models_f3 = {"G": champ_f3, "LGB": lgb_f3, "CB": cb_f3, "HGB": hgb_f3}

print("Individual APs:")
for n, p in models_f3.items():
    print(f"  {n}: {ap(y_f3, p):.6f}")

# Blend search
best = (0, None)
for wg in np.arange(0.2, 0.6, 0.05):
    for wl in np.arange(0.1, 0.5, 0.05):
        for wh in np.arange(0.05, 0.4, 0.05):
            wc = 1 - wg - wl - wh
            if wc < 0.05: continue
            blend = wg*champ_f3 + wl*lgb_f3 + wc*cb_f3 + wh*hgb_f3
            s = ap(y_f3, blend)
            if s > best[0]:
                best = (s, {"G": round(wg,2), "LGB": round(wl,2), "CB": round(wc,2), "HGB": round(wh,2)})

print(f"\nBest 4-way blend: AP={best[0]:.6f} weights={best[1]}")
print(f"Gain vs champion: {best[0]-0.518226:+.6f}")

# ============================================================
# BUILD TEST SUBMISSIONS
# ============================================================
champion_test = np.load(os.path.join(BASE, "FINAL_0518226_SUBMISSION", "final_nested_0518226_test_pred.npy"))
hgb_old_test = np.load(os.path.join(BASE, "AGENT_EXP_AUTO_HGB_VERIFY", "HGB_tuned_fulldata_test_pred.npy"))
lgb_test = np.load(os.path.join(OUT, "lgb_mega_test_pred.npy"))
cb_test = np.load(os.path.join(OUT, "cb_mega_test_pred.npy"))

ss = pd.read_csv(r"sample_submission.csv")

# Use best weights for test
w = best[1]
blend_test = (w["G"]*champion_test + w["LGB"]*lgb_test + w["CB"]*cb_test + w["HGB"]*pred_test)

sub = ss.copy()
sub["fraud"] = blend_test
sub.to_csv(r"submission_file\mega_ensemble_submission.csv", index=False)
print(f"\nSaved: submission_file/mega_ensemble_submission.csv")
print(f"  Mean: {blend_test.mean():.6f}, N: {len(sub)}")

# Also save alternative: champion + HGB_old + LGB_mega (3-way, simpler)
blend2 = 0.4*champion_test + 0.3*hgb_old_test + 0.3*lgb_test
sub2 = ss.copy()
sub2["fraud"] = blend2
sub2.to_csv(r"submission_file\mega_3way_submission.csv", index=False)
print(f"Saved: submission_file/mega_3way_submission.csv")
print(f"  Mean: {blend2.mean():.6f}")

# Save all results
results = {
    "individual_ap": {n: float(ap(y_f3, p)) for n, p in models_f3.items()},
    "best_4way_blend": {"ap": float(best[0]), "weights": best[1], "gain": float(best[0]-0.518226)},
    "features": int(X_train.shape[1]),
    "submission_mean": float(blend_test.mean())
}
with open(os.path.join(OUT, "final_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDONE. Total features: {X_train.shape[1]}")
