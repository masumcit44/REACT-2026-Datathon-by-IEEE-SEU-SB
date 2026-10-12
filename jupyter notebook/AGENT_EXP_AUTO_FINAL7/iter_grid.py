import numpy as np, pandas as pd, os, pickle, time
import lightgbm as lgb
from sklearn.metrics import average_precision_score as ap

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_FINAL7")
N_VAL = 59465
F7 = ["dev_dtype_cust", "dev_loc_cust", "num_dev_axes_cust", "is_new_in_last_30d",
      "dev_new_in_last_30d", "amt_pct_cust", "mch_growth_7d"]
mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
f7 = pd.read_parquet(os.path.join(OUT, "f7_train.parquet"))
y = tr["fraud"].values.astype(int)
ytr, yval = y[:-N_VAL], y[-N_VAL:]
assert np.array_equal(yval.astype(np.int8), np.load(os.path.join(BASE, "Fold_3_v4_targets.npy")).astype(np.int8))

def prep(feats):
    cols = [f for f in feats if f not in F7]
    X = tr[cols].copy()
    X = pd.concat([X, f7[[f for f in feats if f in F7]]], axis=1)
    for c in cats:
        if c in X:
            X[c] = X[c].fillna("missing").astype("category").cat.codes
    X = X.astype(np.float32)
    return X, [i for i, c in enumerate(feats) if c in cats]

def run(label, feats, n_est):
    out = os.path.join(OUT, f"{label}_{n_est}it_pred.npy")
    tag = os.path.join(OUT, f"{label}_{n_est}it_summary.json")
    if os.path.exists(out):
        import json
        return json.load(open(tag))["AP"]
    t0 = time.time()
    X, cat_idx = prep(feats)
    Xtr = X.iloc[:-N_VAL]
    model = lgb.LGBMClassifier(
        n_estimators=n_est, learning_rate=0.05, num_leaves=63,
        min_child_samples=100, reg_alpha=0.5, reg_lambda=5.0,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        min_data_per_group=100, cat_smooth=10, max_cat_threshold=64,
        random_state=4242, feature_fraction_seed=4243, bagging_seed=4244,
        n_jobs=8, verbosity=-1)
    model.fit(Xtr, ytr, categorical_feature=cat_idx, feature_name=feats)
    pred = model.predict_proba(X.iloc[-N_VAL:])[:, 1]
    s = ap(yval, pred)
    np.save(out, pred)
    import json
    json.dump({"AP": s, "iters": n_est}, open(tag, "w"))
    del X, model, pred
    print(f"[{label}] {n_est}it AP={s:.6f} ({time.time()-t0:.1f}s)", flush=True)
    return s

rows = []
for n in [600, 1200]:
    a = run("A", mfeat, n)
    b = run("B", mfeat + F7, n)
    rows.append((n, a, b, b - a))
print("RESULT (iters, A, B, gain):")
for r in rows: print(f"  n={r[0]}  A={r[1]:.6f}  B={r[2]:.6f}  gain={r[3]:+.6f}")