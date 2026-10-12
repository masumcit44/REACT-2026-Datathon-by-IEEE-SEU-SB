import numpy as np, pandas as pd, os, pickle, time, warnings, json
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as ap

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_HGB")

mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
te = pd.read_parquet(os.path.join(BASE, "feature_test_v4.parquet"))
y = tr["fraud"].values.astype(int)
N_VAL = 59465
y_f3 = np.load(os.path.join(BASE, "Fold_3_v4_targets.npy"))
assert N_VAL == len(y_f3)

print(f"train {tr.shape} test {te.shape}", flush=True)
t0 = time.time()

def prep(Xtr, Xte):
    for c in cats:
        Xtr[c] = Xtr[c].fillna("missing").astype("category").cat.codes
        Xte[c] = Xte[c].fillna("missing").astype("category").cat.codes
    num_carriers = [c for c in Xtr.columns if c not in cats]
    na_frac = Xtr[num_carriers].isna().mean()
    na_cols = [c for c in num_carriers if na_frac[c] > 0]
    fills = Xtr[num_carriers].mean().to_dict()
    for df in (Xtr, Xte):
        for c in na_cols:
            df[c + "_na"] = df[c].isna().astype(np.int8)
            df[c] = df[c].fillna(fills[c])
    low_cat = {c for c in cats if Xtr[c].nunique() <= 250}
    cat_mask = np.array([c in low_cat for c in Xtr.columns])
    for df in (Xtr, Xte):
        for c in Xtr.columns:
            df[c] = df[c].astype(np.float32)
    return Xtr, Xte, cat_mask, na_cols

X = tr[mfeat].copy()
Xtest = te[mfeat].copy()
X, Xtest, cat_mask, na_cols = prep(X, Xtest)
print(f"features {X.shape[1]} (na cols {len(na_cols)}), cat_mask sum {cat_mask.sum()}", flush=True)

ytr, yval = y[:-N_VAL], y[-N_VAL:]
Xtr, Xva = X.iloc[:-N_VAL], X.iloc[-N_VAL:]

P = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100,
        l2_regularization=1.0, max_bins=255, random_state=4242,
        early_stopping=False, validation_fraction=None, categorical_features=cat_mask.tolist())

m = HGB(**P)
m.fit(Xtr, ytr)
pred3 = m.predict_proba(Xva)[:, 1]
s = ap(yval, pred3)
print(f"FOLD-3 HGB600 AP = {s:.6f} (expect ~0.519930)", flush=True)
np.save(os.path.join(OUT, "HGB600_repro_fold3_pred.npy"), pred3)

t1 = time.time()
mfull = HGB(**P)
mfull.fit(X, y)
print(f"FULL fit done in {time.time()-t1:.0f}s n_iter={mfull.n_iter_}", flush=True)
pred_te = mfull.predict_proba(Xtest)[:, 1]
np.save(os.path.join(OUT, "HGB600_fulldata_test_pred.npy"), pred_te)
print(f"TEST pred saved N={len(pred_te)} mean={pred_te.mean():.6f} min={pred_te.min():.6f} max={pred_te.max():.6f}", flush=True)

import joblib
joblib.dump(mfull, os.path.join(OUT, "HGB600_fulldata_model.joblib"))

meta = {"fold3_ap": float(s), "features": int(X.shape[1]), "na_cols": na_cols,
        "params": P, "train_time_s": float(time.time()-t0), "test_pred_shape": pred_te.shape,
        "test_pred_mean": float(pred_te.mean())}
json.dump(meta, open(os.path.join(OUT, "HGB600_fulldata_meta.json"), "w"), indent=1, default=str)
print("DONE", flush=True)