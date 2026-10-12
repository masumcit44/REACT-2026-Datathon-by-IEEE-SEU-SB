import numpy as np, pandas as pd, os, pickle, time, warnings, json
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as ap
BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_HGB")
N_VAL = 59465

mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
y = tr["fraud"].values.astype(int)
G = np.load(os.path.join(BASE, "EXP_W2B_COST_SENSITIVE", "best_extended_cost_global_fold3_pred.npy"))
ACC = np.load(os.path.join(BASE, "EXP_B25_ACCELERATION", "acceleration_fold3_global_pred.npy")) \
    if os.path.exists(os.path.join(BASE, "EXP_B25_ACCELERATION", "acceleration_fold3_global_pred.npy")) else None
yfold3 = np.load(os.path.join(BASE, "Fold_3_v4_targets.npy"))

X = tr[mfeat].copy()
for c in cats:
    X[c] = X[c].fillna("missing").astype("category").cat.codes
Xtr, Xva = X.iloc[:-N_VAL], X.iloc[-N_VAL:]
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
ytr = y[:-N_VAL]

m = HGB(max_iter=1000, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=60,
        l2_regularization=5.0, max_bins=255, random_state=4242,
        early_stopping=False, categorical_features=cat_mask.tolist())
t0 = time.time(); m.fit(Xtr, ytr)
pred = m.predict_proba(Xva)[:, 1]
s = ap(yfold3, pred)
np.save(os.path.join(OUT, "HGB_tuned_pred.npy"), pred)
rho = np.corrcoef(pred, G)[0, 1]
print(f"HGB_tuned(1000/0.03/127/60/l2=5) fold3 AP={s:.6f} corr(G)={rho:.4f} time={time.time()-t0:.0f}s", flush=True)

for tag, plist in [("G+HGBt", [G, pred])]:
    best = (0, None)
    for w in np.arange(0.0, 0.71, 0.05):
        bp = (1 - w) * plist[0] + w * plist[1]
        bap = ap(yfold3, bp)
        if bap > best[0]:
            best = (bap, round(w, 2))
    print(f"blend {tag} best={best[0]:.6f} w={best[1]} gate 0.519226", flush=True)

if ACC is not None:
    best3 = (0, None)
    for wg in np.arange(0.3, 0.75, 0.05):
        for wh in np.arange(0.05, 0.5, 0.05):
            wa = 1 - wg - wh
            if wa < 0: continue
            bp = wg * G + wh * pred + wa * ACC
            bap = ap(yfold3, bp)
            if bap > best3[0]:
                best3 = (bap, (round(wg, 2), round(wh, 2), round(wa, 2)))
    print(f"blend G+HGBt+ACC best={best3[0]:.6f} w={best3[1]}", flush=True)
    with open(os.path.join(OUT, "tuned_meta.json"), "w") as f:
        json.dump({"ap": s, "corr_G": rho, "blend_GHGBt": best[0], "blend_G3": best3[0], "w3": best3[1]}, f, indent=1)
else:
    with open(os.path.join(OUT, "tuned_meta.json"), "w") as f:
        json.dump({"ap": s, "corr_G": rho, "blend_GHGBt": best[0]}, f, indent=1)