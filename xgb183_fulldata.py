import numpy as np, pandas as pd, os, pickle, time, warnings, json, subprocess, sys
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap

B = "jupyter notebook"
OUT = os.path.join(B, "AGENT_EXP_XGB")

print("install xgboost...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
import xgboost as xgb
print("xgboost", xgb.__version__)

os.makedirs(OUT, exist_ok=True)
t0 = time.time()
cats = pickle.load(open(os.path.join(B, "categorical_features_v4.pkl"), "rb"))
X = pd.read_parquet(os.path.join(B, "AGENT_EXP_MEGA_V2/train_combined.parquet"))
Xte = pd.read_parquet(os.path.join(B, "AGENT_EXP_MEGA_V2/test_combined.parquet"))
y = np.load(os.path.join(B, "AGENT_EXP_MEGA_V2/y.npy"))
y3 = np.load(os.path.join(B, "Fold_3_v4_targets.npy"))
N_VAL = len(y3)
print(f"train {X.shape} test {Xte.shape} y {y.shape} N_VAL {N_VAL}", flush=True)

na_frac = X.isna().mean()
na_cols = [c for c in X.columns if na_frac[c] > 0 and c not in cats]
fills = X[na_cols].mean().to_dict()
for df in (X, Xte):
    for c in na_cols:
        df[c + "_na"] = df[c].isna().astype(np.int8)
        df[c] = df[c].fillna(fills[c])
for df in (X, Xte):
    for c in X.columns:
        df[c] = df[c].astype("float32")
print(f"features {X.shape[1]} (na cols {len(na_cols)})", flush=True)

Xtr, Xva = X.iloc[:-N_VAL], X.iloc[-N_VAL:]
ytr, yva = y[:-N_VAL], y3

# xgb hist on all features (no native cat in xgb); scale_pos_weight for imbalance
frac_pos = y.mean()
w_pos = (1 - frac_pos) / frac_pos
print(f"pos_rate {frac_pos:.4f} scale_pos_weight {w_pos:.2f}", flush=True)

dtr = xgb.DMatrix(Xtr, label=ytr)
dva = xgb.DMatrix(Xva, label=yva)
dall = xgb.DMatrix(X, label=y)
dte = xgb.DMatrix(Xte.astype("float32"))

P = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
         eta=0.05, max_depth=11, min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
         scale_pos_weight=w_pos, gamma=0.0, lambda_=1.0, seed=4242, nthread=4)
m = xgb.train(P, dtr, num_boost_round=2000, evals=[(dva, "va")],
              early_stopping_rounds=100, verbose_eval=100)
print(f"best_iter {m.best_iteration}", flush=True)
pred3 = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
s3 = ap(yva, pred3)
print(f"XGB183 FOLD3 AP = {s3:.6f}", flush=True)
np.save(os.path.join(OUT, "xgb183_fold3_pred.npy"), pred3)

mfull = xgb.train(P, dall, num_boost_round=m.best_iteration + 1)
pred_te = mfull.predict(dte)
np.save(os.path.join(OUT, "xgb183_test_pred.npy"), pred_te)
print(f"XGB183 TEST pred N={len(pred_te)} mean={pred_te.mean():.6f}", flush=True)
print(f"XGB183 fold-3 AP={s3:.6f} time={time.time()-t0:.0f}s", flush=True)

meta = {"fold3_ap": float(s3), "best_iter": int(m.best_iteration), "features": int(X.shape[1]),
        "params": P, "test_pred_mean": float(pred_te.mean())}
json.dump(meta, open(os.path.join(OUT, "xgb183_meta.json"), "w"), indent=1, default=str)
print("DONE_XGB183", flush=True)