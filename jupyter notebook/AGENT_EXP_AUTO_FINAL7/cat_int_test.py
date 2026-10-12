import numpy as np, pandas as pd, os, pickle, time
import lightgbm as lgb
from sklearn.metrics import average_precision_score as ap

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_FINAL7")
N_VAL = 59465
mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
y = tr["fraud"].values.astype(int)
ytr, yval = y[:-N_VAL], y[-N_VAL:]

def run(label, cat_dtype):
    X = tr[mfeat].copy()
    for c in cats:
        X[c] = X[c].fillna("missing").astype("category")
        if cat_dtype == "int32":
            X[c] = X[c].cat.codes.astype("int32")
    for c in X.columns:
        if c not in cats:
            X[c] = X[c].astype(np.float32)
    cat_idx = [i for i, c in enumerate(mfeat) if c in cats]
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=100, reg_alpha=0.5, reg_lambda=5.0,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        min_data_per_group=100, cat_smooth=10, max_cat_threshold=64,
        random_state=4242, feature_fraction_seed=4243, bagging_seed=4244,
        n_jobs=8, verbosity=-1)
    model.fit(X.iloc[:-N_VAL], ytr, categorical_feature=cat_idx, feature_name=mfeat)
    pred = model.predict_proba(X.iloc[-N_VAL:])[:, 1]
    s = ap(yval, pred)
    np.save(os.path.join(OUT, f"A_{cat_dtype}_500it_pred.npy"), pred)
    print(f"[A @500it cats={cat_dtype}] AP={s:.6f} ({time.time()-t0:.1f}s)", flush=True)
    return s

run("A", "int32")