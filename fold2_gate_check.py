import numpy as np, pandas as pd, os, pickle, time, warnings, json, subprocess, sys
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap

# ============================================================
# FOLD-2 GATE CHECK FOR xgb_hgb600_blend
# Blend: XGB183 .57 + HGB600 .34 + mega_lgb .09 + mega_cb .002
# fold-3 AP = 0.527088
# gate: fold-2 AP >= 0.527088 - 0.001 = 0.526088
# ============================================================

B = "jupyter notebook"
MEGA = os.path.join(B, "AGENT_EXP_MEGA_V2")
HGB_DIR = os.path.join(B, "AGENT_EXP_AUTO_HGB")
XGB_DIR = os.path.join(B, "AGENT_EXP_XGB")

# Load data
print("Loading data...", flush=True)
X = pd.read_parquet(os.path.join(MEGA, "train_combined.parquet"))
y = np.load(os.path.join(MEGA, "y.npy"))
y3 = np.load(os.path.join(B, "Fold_3_v4_targets.npy"))
y2 = np.load(os.path.join(B, "Fold_2_v4_targets.npy"))
N3 = len(y3)
N2 = len(y2)
print(f"X: {X.shape}, y: {y.shape}, fold3: {N3}, fold2: {N2}", flush=True)

# Fold indices (sequential, time-ordered)
# fold-3 = last 59465 rows, fold-2 = rows -118930:-59465
idx_f3 = np.arange(len(y) - N3, len(y))
idx_f2 = np.arange(len(y) - N3 - N2, len(y) - N3)
idx_f1 = np.arange(0, len(y) - N3 - N2)
print(f"fold3 indices: {idx_f3[0]}-{idx_f3[-1]}, fold2: {idx_f2[0]}-{idx_f2[-1]}, fold1: 0-{idx_f1[-1]}", flush=True)

# For fold-2 gate: train on all data EXCEPT fold-2
# This means rows 0..idx_f2[0]-1  +  rows idx_f2[-1]+1..end
# Which is: fold-1 + everything before fold-1  +  fold-3
mask_f2_exclude = np.zeros(len(y), dtype=bool)
mask_f2_exclude[idx_f2] = True
Xtr_f2 = X.iloc[~mask_f2_exclude]
ytr_f2 = y[~mask_f2_exclude]
Xva_f2 = X.iloc[idx_f2]
yva_f2 = y2  # fold-2 targets

# For fold-3 verification: train on all data EXCEPT fold-3
mask_f3_exclude = np.zeros(len(y), dtype=bool)
mask_f3_exclude[idx_f3] = True
Xtr_f3 = X.iloc[~mask_f3_exclude]
ytr_f3 = y[~mask_f3_exclude]
Xva_f3 = X.iloc[idx_f3]
yva_f3 = y3

print(f"fold2 train: {Xtr_f2.shape}, val: {Xva_f2.shape}", flush=True)
print(f"fold3 train: {Xtr_f3.shape}, val: {Xva_f3.shape}", flush=True)

# NA handling (same as xgb183 script)
na_frac = Xtr_f2.isna().mean()
na_cols = [c for c in X.columns if na_frac[c] > 0]
fills = Xtr_f2[na_cols].mean().to_dict()
for df in (Xtr_f2, Xva_f2, Xtr_f3, Xva_f3):
    for c in na_cols:
        df[c + "_na"] = df[c].isna().astype(np.int8)
        df[c] = df[c].fillna(fills[c])
for df in (Xtr_f2, Xva_f2, Xtr_f3, Xva_f3):
    for c in df.columns:
        df[c] = df[c].astype("float32")

print(f"Features after NA indicators: {Xtr_f2.shape[1]}", flush=True)

# Load categorical features
cats = pickle.load(open(os.path.join(B, "categorical_features_v4.pkl"), "rb"))

results = {}
t0 = time.time()

# ============================================================
# MEMBER 1: XGB183
# ============================================================
print("\n=== XGB183 ===", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"], stderr=subprocess.DEVNULL)
import xgboost as xgb

frac_pos = ytr_f2.mean()
w_pos = (1 - frac_pos) / frac_pos
P = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
         eta=0.05, max_depth=11, min_child_weight=100, subsample=0.8, colsample_bytree=0.7,
         scale_pos_weight=w_pos, gamma=0.0, lambda_=1.0, seed=4242, nthread=4)

for fold_name, Xtr, ytr, Xva, yva in [("FOLD2", Xtr_f2, ytr_f2, Xva_f2, yva_f2),
                                        ("FOLD3", Xtr_f3, ytr_f3, Xva_f3, yva_f3)]:
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    m = xgb.train(P, dtr, num_boost_round=2000, evals=[(dva, "va")],
                  early_stopping_rounds=100, verbose_eval=200)
    pred = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
    score = ap(yva, pred)
    print(f"XGB183 {fold_name} AP = {score:.6f} (best_iter={m.best_iteration})", flush=True)
    results[f"xgb183_{fold_name}"] = pred
    np.save(f"gate_xgb183_{fold_name}_pred.npy", pred)
    if fold_name == "FOLD2":
        results["xgb183_f2_ap"] = score
    else:
        results["xgb183_f3_ap"] = score

# ============================================================
# MEMBER 2: HGB600
# ============================================================
print("\n=== HGB600 ===", flush=True)
from sklearn.ensemble import HistGradientBoostingClassifier

hgb_params = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                  min_samples_leaf=100, l2_regularization=1.0, max_bins=255,
                  random_state=4242, early_stopping=False, validation_fraction=None,
                  categorical_features="from_dtype")

for fold_name, Xtr, ytr, Xva, yva in [("FOLD2", Xtr_f2, ytr_f2, Xva_f2, yva_f2),
                                        ("FOLD3", Xtr_f3, ytr_f3, Xva_f3, yva_f3)]:
    t1 = time.time()
    clf = HistGradientBoostingClassifier(**hgb_params)
    clf.fit(Xtr, ytr)
    pred = clf.predict_proba(Xva)[:, 1]
    score = ap(yva, pred)
    print(f"HGB600 {fold_name} AP = {score:.6f} ({time.time()-t1:.0f}s)", flush=True)
    results[f"hgb600_{fold_name}"] = pred
    np.save(f"gate_hgb600_{fold_name}_pred.npy", pred)
    if fold_name == "FOLD2":
        results["hgb600_f2_ap"] = score
    else:
        results["hgb600_f3_ap"] = score

# ============================================================
# MEMBER 3: MEGA_LGB (183 features)
# ============================================================
print("\n=== MEGA_LGB ===", flush=True)
try:
    import lightgbm as lgb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], stderr=subprocess.DEVNULL)
    import lightgbm as lgb

lgb_params = dict(objective="binary", metric="average_precision", boosting_type="gbdt",
                  num_leaves=63, learning_rate=0.05, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=5, min_child_samples=100,
                  lambda_l2=1.0, seed=4242, verbose=-1, n_jobs=4)

for fold_name, Xtr, ytr, Xva, yva in [("FOLD2", Xtr_f2, ytr_f2, Xva_f2, yva_f2),
                                        ("FOLD3", Xtr_f3, ytr_f3, Xva_f3, yva_f3)]:
    t1 = time.time()
    dtrain = lgb.Dataset(Xtr, label=ytr, categorical_feature=[c for c in cats if c in Xtr.columns])
    dval = lgb.Dataset(Xva, label=yva, reference=dtrain)
    callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                     callbacks=callbacks)
    pred = model.predict(Xva)
    score = ap(yva, pred)
    print(f"MEGA_LGB {fold_name} AP = {score:.6f} ({time.time()-t1:.0f}s)", flush=True)
    results[f"lgb_{fold_name}"] = pred
    np.save(f"gate_lgb_{fold_name}_pred.npy", pred)
    if fold_name == "FOLD2":
        results["lgb_f2_ap"] = score
    else:
        results["lgb_f3_ap"] = score

# ============================================================
# MEMBER 4: MEGA_CB (183 features)
# ============================================================
print("\n=== MEGA_CB ===", flush=True)
try:
    import catboost as cb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"], stderr=subprocess.DEVNULL)
    import catboost as cb

cb_cat_idx = [i for i, c in enumerate(X.columns) if c in cats]

# CatBoost requires integer/string cat features, not float codes
for dfg in (Xtr_f2, Xva_f2, Xtr_f3, Xva_f3):
    for i in cb_cat_idx:
        col = X.columns[i]
        dfg[col] = dfg[col].round().astype(np.int64)

for fold_name, Xtr, ytr, Xva, yva in [("FOLD2", Xtr_f2, ytr_f2, Xva_f2, yva_f2),
                                        ("FOLD3", Xtr_f3, ytr_f3, Xva_f3, yva_f3)]:
    t1 = time.time()
    dtrain = cb.Pool(Xtr, label=ytr, cat_features=cb_cat_idx)
    dval = cb.Pool(Xva, label=yva, cat_features=cb_cat_idx)
    model = cb.CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=10, l2_leaf_reg=5,
        min_data_in_leaf=100, border_count=254, random_seed=4242,
        task_type="CPU", verbose=200, eval_metric="AveragePrecision",
        early_stopping_rounds=100
    )
    model.fit(dtrain, eval_set=dval, use_best_model=True)
    pred = model.predict_proba(Xva)[:, 1]
    score = ap(yva, pred)
    print(f"MEGA_CB {fold_name} AP = {score:.6f} ({time.time()-t1:.0f}s)", flush=True)
    results[f"cb_{fold_name}"] = pred
    np.save(f"gate_cb_{fold_name}_pred.npy", pred)
    if fold_name == "FOLD2":
        results["cb_f2_ap"] = score
    else:
        results["cb_f3_ap"] = score

# ============================================================
# BLEND COMPARISON
# ============================================================
print("\n=== BLEND ANALYSIS ===", flush=True)
W = {"xgb183": 0.57, "hgb600": 0.34, "lgb": 0.09, "cb": 0.002}
print(f"Weights: {W}", flush=True)

for fold_name in ["FOLD3", "FOLD2"]:
    preds = np.zeros_like(results[f"xgb183_{fold_name}"])
    for member, weight in W.items():
        preds += weight * results[f"{member}_{fold_name}"]
    score = ap(yva_f3 if fold_name == "FOLD3" else yva_f2, preds)
    print(f"BLEND {fold_name} AP = {score:.6f}", flush=True)
    results[f"blend_{fold_name}"] = preds
    results[f"blend_{fold_name}_ap"] = score

# Gate check
f3_ap = results["blend_FOLD3_ap"]
f2_ap = results["blend_FOLD2_ap"]
gate_threshold = f3_ap - 0.001
print(f"\n=== GATE CHECK ===", flush=True)
print(f"Fold-3 AP: {f3_ap:.6f}", flush=True)
print(f"Fold-2 AP: {f2_ap:.6f}", flush=True)
print(f"Gate threshold (f3 - 0.001): {gate_threshold:.6f}", flush=True)
if f2_ap >= gate_threshold:
    print("GATE: PASS", flush=True)
    results["gate"] = "PASS"
else:
    print("GATE: FAIL", flush=True)
    results["gate"] = "FAIL"

# Individual member fold-2 APs
print(f"\n=== INDIVIDUAL FOLD-2 APs ===", flush=True)
for member in ["xgb183", "hgb600", "lgb", "cb"]:
    print(f"  {member}: {results[f'{member}_f2_ap']:.6f}", flush=True)

# Save results
meta = {
    "blend_fold3_ap": float(f3_ap),
    "blend_fold2_ap": float(f2_ap),
    "gate_threshold": float(gate_threshold),
    "gate": results["gate"],
    "weights": W,
    "individual_fold2": {m: float(results[f"{m}_f2_ap"]) for m in ["xgb183", "hgb600", "lgb", "cb"]},
    "individual_fold3": {m: float(results[f"{m}_f3_ap"]) for m in ["xgb183", "hgb600", "lgb", "cb"]},
    "time_s": time.time() - t0,
}
json.dump(meta, open("fold2_gate_results.json", "w"), indent=2, default=str)

np.save("xgb_hgb600_blend_fold2_pred.npy", results["blend_FOLD2"])
np.save("xgb_hgb600_blend_fold3_pred.npy", results["blend_FOLD3"])

print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
print("DONE_FOLD2_GATE", flush=True)
