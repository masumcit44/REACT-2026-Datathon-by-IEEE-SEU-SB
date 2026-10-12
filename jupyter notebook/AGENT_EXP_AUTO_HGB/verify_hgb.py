import numpy as np, pandas as pd, os, pickle, time, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as ap
BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_HGB")
N_VAL = 59465

yval = np.load(os.path.join(BASE, "Fold_3_v4_targets.npy"))
p1 = np.load(os.path.join(OUT, "HGB1_pred.npy"))
print("RELOAD HGB1 pred AP:", round(ap(yval, p1), 6))

mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
y = tr["fraud"].values.astype(int)

def build(lo, hi):
    X = tr[mfeat].copy()
    for c in cats:
        X[c] = X[c].fillna("missing").astype("category").cat.codes
    sl = slice(lo, hi)
    Xtr, Xva = X.iloc[lo:hi], X.iloc[hi:hi + N_VAL]
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
    return Xtr, Xva, cat_mask

def run(gtrain, predpath, seed, gval=None, name=""):
    m = HGB(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100,
            l2_regularization=1.0, max_bins=255, random_state=seed,
            early_stopping=False, categorical_features=cat_mask.tolist())
    t0 = time.time(); m.fit(Xtr, gtrain)
    pred = m.predict_proba(Xva)[:, 1]
    np.save(predpath, pred)
    s = ap(gval, pred) if gval is not None else None
    print(f"{name} AP={s} time={time.time()-t0:.0f}s", flush=True)
    return pred

# Fold 3 seed 4243 reproducibility
Xtr, Xva, cat_mask = build(0, len(y) - N_VAL)
yfold3 = y[-N_VAL:]; yf3train = y[:-N_VAL]
fit = None
run(yf3train, os.path.join(OUT, "HGB1_s2_pred.npy"), 4243, yfold3, "HGB600 seed4243")

# Fold 2 stability (val = rows[-2N:-N], train = rows[:-2N])
Xtr, Xva, cat_mask = build(0, len(y) - 2 * N_VAL)
yf2 = np.load(os.path.join(BASE, "Fold_2_v4_targets.npy")) if os.path.exists(os.path.join(BASE, "Fold_2_v4_targets.npy")) else y[-2 * N_VAL:-N_VAL]
run(y[:-2 * N_VAL], os.path.join(OUT, "HGB1_fold2_pred.npy"), 4243, yf2, "HGB600 fold2")