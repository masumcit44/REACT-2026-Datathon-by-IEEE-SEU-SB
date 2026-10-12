import numpy as np, pandas as pd, os, pickle, time, json
import lightgbm as lgb
from sklearn.metrics import average_precision_score as ap

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_REPROD")
os.makedirs(OUT, exist_ok=True)
N_VAL = 59465

mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
y = tr["fraud"].values.astype(int)
ytr, yval = y[:-N_VAL], y[-N_VAL:]
assert np.array_equal(yval.astype(np.int8), np.load(os.path.join(BASE, "Fold_3_v4_targets.npy")).astype(np.int8))

X = tr[mfeat].copy()
for c in cats:
    X[c] = X[c].fillna("missing").astype("category").cat.codes
for c in X.columns:
    if c not in cats:
        X[c] = X[c].astype(np.float32)
cat_idx = [i for i, c in enumerate(mfeat) if c in cats]

model = lgb.LGBMClassifier(
    boosting_type="gbdt", objective="binary", num_leaves=63, max_depth=-1,
    learning_rate=0.05, n_estimators=600, subsample_for_bin=200000,
    min_split_gain=0.0, min_child_weight=0.001, min_child_samples=100,
    subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
    reg_alpha=0.5, reg_lambda=5.0, random_state=4242,
    importance_type="split", n_jobs=8, verbosity=-1)
t0 = time.time()
model.fit(X.iloc[:-N_VAL], ytr, categorical_feature=cat_idx)
pred = model.predict_proba(X.iloc[-N_VAL:])[:, 1]
s = ap(yval, pred)
np.save(os.path.join(OUT, "A_V4_probe600_pred.npy"), pred)
print(f"A_V4 (probe600 exact cfg) AP={s:.6f}  target 0.506852  ({time.time()-t0:.0f}s)", flush=True)