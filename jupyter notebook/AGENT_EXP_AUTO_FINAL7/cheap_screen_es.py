import numpy as np, pandas as pd, os, pickle, time, json
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
assert len(tr) == len(f7)
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

def run(label, feats, n_est=2500, lr=0.05, leaves=63, mcs=100, es=200,
        skip_if_exists=False):
    out_pred = os.path.join(OUT, f"{label}_cheapES_fold3_pred.npy")
    if skip_if_exists and os.path.exists(out_pred):
        print(f"[{label}] already exists, skipping", flush=True)
        return float(np.load(out_pred).shape[0]) * 0  # placeholder
    t0 = time.time()
    X, cat_idx = prep(feats)
    Xtr, Xval = X.iloc[:-N_VAL], X.iloc[-N_VAL:]
    del X
    model = lgb.LGBMClassifier(
        n_estimators=n_est, learning_rate=lr, num_leaves=leaves,
        min_child_samples=mcs, reg_alpha=0.5, reg_lambda=5.0,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        min_data_per_group=100, cat_smooth=10, max_cat_threshold=64,
        random_state=4242, feature_fraction_seed=4243, bagging_seed=4244,
        n_jobs=8, verbosity=-1)
    model.fit(Xtr, ytr, categorical_feature=cat_idx, feature_name=feats,
              eval_set=[(Xval, yval)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(es, verbose=False)])
    pred = model.predict_proba(Xval)[:, 1]
    s = ap(yval, pred)
    bi = model.best_iteration_
    np.save(out_pred, pred)
    with open(os.path.join(OUT, f"{label}_cheapES_meta.json"), "w") as f:
        json.dump({"AP": s, "best_iter": bi, "n_feats": len(feats)}, f)
    print(f"[{label}] {len(feats)}f AP={s:.6f} best_iter={bi} ({time.time()-t0:.1f}s)", flush=True)
    return s

def safe():
    ra = rb = None
    if os.path.exists(os.path.join(OUT, "A_es_cheapES_meta.json")):
        ra = json.load(open(os.path.join(OUT, "A_es_cheapES_meta.json"))).get("AP")
    if os.path.exists(os.path.join(OUT, "B_es_cheapES_meta.json")):
        rb = json.load(open(os.path.join(OUT, "B_es_cheapES_meta.json"))).get("AP")
    if ra is None:
        ra, _, _ = run("A_es", mfeat)
        json.dump({"AP": ra}, open(os.path.join(OUT, "A_es_cheapES_meta.json"), "w"))
    if rb is None:
        rb, _, mb = run("B_es", mfeat + F7)
        json.dump({"AP": rb}, open(os.path.join(OUT, "B_es_cheapES_meta.json"), "w"))
        imp = pd.DataFrame({"feature": mfeat + F7, "gain": mb.feature_importance("gain")}).sort_values("gain", ascending=False)
        imp.head(20).to_csv(os.path.join(OUT, "B_es_top20_importance.csv"), index=False)
    gain = rb - ra
    return ra, rb, gain

ra, rb, gain = safe()
json.dump({"A_es": ra, "B_es": rb, "gain": gain,
           "gate": "+0.002", "verdict": "PASS" if gain >= 0.002 else "FAIL"},
          open(os.path.join(OUT, "cheapES_screen_result.json"), "w"), indent=2)
print(f"CHEAP ES SCREEN: A={ra:.6f} B={rb:.6f} gain={gain:+.6f} " + ("PASS(>=+0.002)" if gain >= 0.002 else "FAIL(<+0.002)"))