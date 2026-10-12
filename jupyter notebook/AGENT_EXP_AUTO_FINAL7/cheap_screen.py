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
yval = y[-N_VAL:]
assert np.array_equal(yval.astype(np.int8),
                      np.load(os.path.join(BASE, "Fold_3_v4_targets.npy")).astype(np.int8)), "target alignment!"

def prep(feats):
    cols = [f for f in feats if f not in F7]
    X = tr[cols].copy()
    X = pd.concat([X, f7[[f for f in feats if f in F7]]], axis=1)
    for c in cats:
        if c in X:
            X[c] = X[c].fillna("missing").astype("category").cat.codes
    X = X.astype(np.float32)
    cat_idx = [i for i, c in enumerate(feats) if c in cats]
    return X, cat_idx

def run(label, feats, iters=500, lr=0.05, leaves=63, mcs=100):
    t0 = time.time()
    X, cat_idx = prep(feats)
    Xtr, Xval = X.iloc[:-N_VAL], X.iloc[-N_VAL:]
    del X
    model = lgb.LGBMClassifier(
        n_estimators=iters, learning_rate=lr, num_leaves=leaves,
        min_child_samples=mcs, reg_alpha=0.5, reg_lambda=5.0,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        min_data_per_group=100, cat_smooth=10, max_cat_threshold=64,
        random_state=4242, feature_fraction_seed=4243, bagging_seed=4244,
        n_jobs=8, verbosity=-1,
    )
    model.fit(Xtr, tr["fraud"].values[:-N_VAL], categorical_feature=cat_idx,
              feature_name=feats)
    pred = model.predict_proba(Xval)[:, 1]
    s = ap(yval, pred)
    np.save(os.path.join(OUT, f"{label}_cheap_fold3_pred.npy"), pred)
    model.booster_.save_model(os.path.join(OUT, f"{label}_cheap_model.txt"))
    with open(os.path.join(OUT, f"{label}_cheap_importance.csv"), "w") as f:
        imp = pd.DataFrame({"feature": feats,
                            "gain": model.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
        imp.to_csv(f, index=False)
    print(f"[{label}] {len(feats)}f AP={s:.6f} ({time.time()-t0:.1f}s)", flush=True)
    return s

res = {}
res["A_V4_101"] = run("A_v4_baseline", mfeat)
res["B_V4_F7_108"] = run("B_v4_f7", mfeat + F7)
gain = res["B_V4_F7_108"] - res["A_V4_101"]
verdict = "PASS (>= +0.002)" if gain >= 0.002 else "FAIL (< +0.002)"
json.dump({"A": res["A_V4_101"], "B": res["B_V4_F7_108"], "gain": gain,
           "gate": "+0.002", "verdict": verdict},
          open(os.path.join(OUT, "cheap_screen_result.json"), "w"), indent=2)
print(f"CHEAP SCREEN: A={res['A_V4_101']:.6f} B={res['B_V4_F7_108']:.6f} gain={gain:+.6f} VERDICT={verdict}")