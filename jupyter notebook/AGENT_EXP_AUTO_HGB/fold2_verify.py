import numpy as np, pandas as pd, os, pickle, time, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as ap
BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_HGB")

f2targets = np.load(os.path.join(BASE, "EXP_B25_ACCELERATION_FOLD2", "fold2_targets.npy"))
N2 = len(f2targets)
print("fold2 N2 =", N2)

mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
y = tr["fraud"].values.astype(int)
print("total", len(y), "N2", N2)

X = tr[mfeat].copy()
for c in cats:
    X[c] = X[c].fillna("missing").astype("category").cat.codes

# Fold 2: train = rows[:-N2], val = rows[-N2:]
Xtr, Xva = X.iloc[:-N2], X.iloc[-N2:]
ytr, yva = y[:-N2], y[-N2:]
print("ytr len", len(ytr), "yva rate", yva.mean())

na_cols = [c for c in X.columns if c not in cats and Xtr[c].isna().mean() > 0]
fills = Xtr[na_cols].mean().to_dict()
for df in (Xtr, Xva):
    for c in na_cols:
        df[c + "_na"] = df[c].isna().astype(np.int8)
        df[c] = df[c].fillna(fills[c])

low_cat = {c for c in cats if Xtr[c].nunique() <= 250}
cat_mask = np.array([c in low_cat for c in Xtr.columns], dtype=bool)
for c in Xtr.columns:
    Xtr[c] = Xtr[c].astype(np.float32); Xva[c] = Xva[c].astype(np.float32)

m = HGB(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100,
        l2_regularization=1.0, max_bins=255, random_state=4242,
        early_stopping=False, categorical_features=cat_mask.tolist())
t0 = time.time(); m.fit(Xtr, ytr)
pred = m.predict_proba(Xva)[:, 1]
np.save(os.path.join(OUT, "HGB1_fold2_correct_pred.npy"), pred)
s = ap(f2targets, pred)
print(f"HGB600 fold2 AP={s:.6f} time={time.time()-t0:.0f}s", flush=True)

# blend with fold-2 baselines
G2 = np.load(os.path.join(BASE, "EXP_B25_ACCELERATION_FOLD2", "fold2_matched_tuned_v4_baseline_pred.npy"))
A2 = np.load(os.path.join(BASE, "EXP_B25_ACCELERATION_FOLD2", "fold2_tuned_acceleration_pred.npy"))
print("\nfold-2 blend sweep:")
for name, base in [("G2", G2), ("A2", A2)]:
    for w in [0.0, 0.06, 0.1, 0.2, 0.3, 0.5]:
        bp = (1-w)*base + w*pred
        print(f"  {name}+HGB@{w:.2f} AP={ap(f2targets, bp):.6f}")

# 3-way
best3 = (0, None)
for wg in np.arange(0.0, 1.01, 0.05):
    for wh in np.arange(0.0, 1.01 - wg, 0.05):
        wa = 1 - wg - wh
        if wa < 0: continue
        bp = wg*G2 + wh*pred + wa*A2
        a = ap(f2targets, bp)
        if a > best3[0]: best3 = (a, (round(wg,2), round(wh,2), round(wa,2)))
print(f"\nfold-2 3-way G+HGB+A best={best3[0]:.6f} w={best3[1]}")
print("baseline tuned G2:", round(ap(f2targets, G2), 6))
print("baseline tuned A2:", round(ap(f2targets, A2), 6))