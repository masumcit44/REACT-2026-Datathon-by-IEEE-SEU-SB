import numpy as np, pandas as pd, os, pickle, time, warnings, json
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as ap

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_HGB")
os.makedirs(OUT, exist_ok=True)
N_VAL = 59465

mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
y = tr["fraud"].values.astype(int)
ytr, yval = y[:-N_VAL], y[-N_VAL:]
G = np.load(os.path.join(BASE, "EXP_W2B_COST_SENSITIVE", "best_extended_cost_global_fold3_pred.npy"))

X = tr[mfeat].copy()
for c in cats:
    X[c] = X[c].fillna("missing").astype("category").cat.codes
Xtr, Xva = X.iloc[:-N_VAL], X.iloc[-N_VAL:]

num_carriers = [c for c in X.columns if c not in cats]
na_frac = (Xtr[num_carriers].isna().mean())
fills = Xtr[num_carriers].mean().to_dict()
# NaN mask features (only for columns with NaNs)
na_cols = [c for c in num_carriers if na_frac[c] > 0]
for df in (Xtr, Xva):
    for c in na_cols:
        df[c + "_na"] = df[c].isna().astype(np.int8)
        df[c] = df[c].fillna(fills[c])

cat_cards = {c: Xtr[c].nunique() for c in cats}
low_cat = {c for c in cats if cat_cards[c] <= 250}   # high-card -> ordinal numeric for HGB
print("HGB categorical(<=250):", sorted(low_cat), "numeric-high-card:", sorted(set(cats) - low_cat), flush=True)
cat_mask = np.array([c in low_cat for c in Xtr.columns])
for c in Xtr.columns:
    Xtr[c] = Xtr[c].astype(np.float32)
    Xva[c] = Xva[c].astype(np.float32)

m = HGB(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100,
        l2_regularization=1.0, max_bins=255, random_state=4242,
        early_stopping=False, validation_fraction=None, categorical_features=cat_mask.tolist())
t0 = time.time()
m.fit(Xtr, ytr)
pred = m.predict_proba(Xva)[:, 1]
s = ap(yval, pred)
np.save(os.path.join(OUT, "HGB1_pred.npy"), pred)
rho_c = np.corrcoef(pred, G)[0, 1]
print(f"HGB600(63/100) fold3 AP={s:.6f} corr(G)={rho_c:.4f} time={time.time()-t0:.0f}s", flush=True)

# blend sweep vs champion
best = (0, None, None, 0.0)
for w in np.arange(0.0, 0.56, 0.05):
    bp = (1 - w) * G + w * pred
    bap = ap(yval, bp)
    if bap > best[0]:
        best = (bap, round(w, 2), "G+HGB", w)
print(f"blend G+HGB best AP={best[0]:.6f} w={best[2]}={best[1]}  gate 0.519226+", flush=True)
with open(os.path.join(OUT, "HGB1_meta.json"), "w") as f:
    json.dump({"ap": s, "corr_G": rho_c, "blend_best": best[0], "blend_w": best[1], "n_feat": Xtr.shape[1], "na_cols": na_cols}, f, indent=1)