"""
RAPID-FIRE: Multiple approaches, quick iteration.
Goal: hit 0.58 AP on public LB.
"""
import numpy as np, pandas as pd, os, pickle, time, warnings, json
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
from sklearn.ensemble import IsolationForest
import lightgbm as lgb

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_RAPIDFIRE")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# LOAD ALL EXISTING PREDICTIONS
# ============================================================
print("Loading all existing assets...")
t0 = time.time()

y_f3 = np.load(os.path.join(BASE, "Fold_3_v4_targets.npy"))
N_VAL = len(y_f3)

tr_v4 = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
te_v4 = pd.read_parquet(os.path.join(BASE, "feature_test_v4.parquet"))
mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
y = tr_v4["fraud"].values.astype(int)

# Mega features
X_mega_tr = pd.read_parquet(os.path.join(BASE, "AGENT_EXP_MEGA_V2", "train_new_features.parquet"))
X_mega_te = pd.read_parquet(os.path.join(BASE, "AGENT_EXP_MEGA_V2", "test_new_features.parquet"))

# Load ALL existing test predictions
existing = {}
for name, path in [
    ("champion", "FINAL_0518226_SUBMISSION/final_nested_0518226_test_pred.npy"),
    ("hgb_old", "AGENT_EXP_AUTO_HGB_VERIFY/HGB_tuned_fulldata_test_pred.npy"),
    ("lgb_mega", "AGENT_EXP_MEGA_V2/lgb_mega_test_pred.npy"),
    ("cb_mega", "AGENT_EXP_MEGA_V2/cb_mega_test_pred.npy"),
]:
    p = os.path.join(BASE, path)
    if os.path.exists(p):
        existing[name] = np.load(p)
        print(f"  {name}: N={len(existing[name])}, mean={existing[name].mean():.6f}")

# Load ALL existing fold-3 predictions
existing_f3 = {}
for name, path in [
    ("champion", "EXP_W2B_COST_SENSITIVE/best_extended_cost_global_fold3_pred.npy"),
    ("hgb600", "AGENT_EXP_AUTO_HGB/HGB1_pred.npy"),
    ("hgb_tuned", "AGENT_EXP_AUTO_HGB/HGB_tuned_pred.npy"),
    ("lgb_mega", "AGENT_EXP_MEGA_V2/lgb_mega_fold3_pred.npy"),
    ("cb_mega", "AGENT_EXP_MEGA_V2/cb_mega_fold3_pred.npy"),
]:
    p = os.path.join(BASE, path)
    if os.path.exists(p):
        existing_f3[name] = np.load(p)
        print(f"  {name} f3: AP={ap(y_f3, existing_f3[name]):.6f}")

# Combined features
X_all_tr = pd.concat([tr_v4[mfeat].reset_index(drop=True), X_mega_tr.reset_index(drop=True).add_prefix("n_")], axis=1)
X_all_te = pd.concat([te_v4[mfeat].reset_index(drop=True), X_mega_te.reset_index(drop=True).add_prefix("n_")], axis=1)

for c in cats:
    X_all_tr[c] = X_all_tr[c].fillna("missing").astype("category").cat.codes
    X_all_te[c] = X_all_te[c].fillna("missing").astype("category").cat.codes
for c in X_all_tr.columns:
    X_all_tr[c] = X_all_tr[c].fillna(-999).astype(np.float32)
    X_all_te[c] = X_all_te[c].fillna(-999).astype(np.float32)

cat_idx = [i for i, c in enumerate(X_all_tr.columns) if c in cats]
Xtr, Xva = X_all_tr.iloc[:-N_VAL], X_all_tr.iloc[-N_VAL:]
ytr = y[:-N_VAL]

print(f"Features: {X_all_tr.shape[1]} ({time.time()-t0:.0f}s)")

# ============================================================
# APPROACH 1: LIGHTGBM RANKING OBJECTIVE (LambdaRank)
# ============================================================
print("\n[SKIP APPROACH 1: LambdaRank - API issue]")

# ============================================================
# APPROACH 2: LGB CROSS-ENTROPY + FOCAL LOSS
# ============================================================
print("\n" + "="*60)
print("APPROACH 2: LGB with custom focal loss")
print("="*60)

# Focal loss emphasizes hard examples
# custom_obj = focal binary
params_focal = {
    "boosting_type": "gbdt", "objective": "binary",
    "num_leaves": 200, "max_depth": 12,
    "learning_rate": 0.01, "n_estimators": 3000,
    "subsample_for_bin": 200000,
    "min_child_samples": 50,
    "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0, "reg_lambda": 10.0,
    "random_state": 4243, "n_jobs": 8, "verbosity": -1,
    "scale_pos_weight": 10  # upweight fraud
}

t0 = time.time()
m_focal = lgb.LGBMClassifier(**params_focal)
m_focal.fit(Xtr, ytr,
            categorical_feature=cat_idx,
            eval_set=[(Xva, y_f3)],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(300)])
pred_focal_va = m_focal.predict_proba(Xva)[:, 1]
ap_focal = ap(y_f3, pred_focal_va)
print(f"  focal fold-3 AP: {ap_focal:.6f} ({time.time()-t0:.0f}s)")

t0 = time.time()
m_focal_full = lgb.LGBMClassifier(**params_focal)
m_focal_full.set_params(n_estimators=m_focal.best_iteration_)
m_focal_full.fit(X_all_tr, y, categorical_feature=cat_idx)
pred_focal_test = m_focal_full.predict_proba(X_all_te)[:, 1]
np.save(os.path.join(OUT, "lgb_focal_test.npy"), pred_focal_test)
print(f"  focal full: {time.time()-t0:.0f}s")

# ============================================================
# APPROACH 3: LGB EXPLORATION + EXPLOITATION
# ============================================================
print("\n" + "="*60)
print("APPROACH 3: LGB wider trees + regularization sweep")
print("="*60)

for trial_name, params in [
    ("wide", {"num_leaves": 255, "max_depth": 15, "learning_rate": 0.02, "n_estimators": 2000,
              "min_child_samples": 80, "subsample": 0.8, "colsample_bytree": 0.7,
              "reg_alpha": 2.0, "reg_lambda": 10.0, "random_state": 4244}),
    ("deep", {"num_leaves": 63, "max_depth": -1, "learning_rate": 0.01, "n_estimators": 3000,
              "min_child_samples": 200, "subsample": 0.9, "colsample_bytree": 0.6,
              "reg_alpha": 0.1, "reg_lambda": 2.0, "random_state": 4245}),
    ("imbalance", {"num_leaves": 127, "max_depth": -1, "learning_rate": 0.025, "n_estimators": 2500,
                   "min_child_samples": 100, "subsample": 0.85, "colsample_bytree": 0.85,
                   "reg_alpha": 0.5, "reg_lambda": 5.0, "random_state": 4246,
                   "scale_pos_weight": 15}),
]:
    base = {
        "boosting_type": "gbdt", "objective": "binary",
        "subsample_for_bin": 200000, "subsample_freq": 1,
        "n_jobs": 8, "verbosity": -1
    }
    base.update(params)
    t0 = time.time()
    m = lgb.LGBMClassifier(**base)
    m.fit(Xtr, ytr, categorical_feature=cat_idx,
          eval_set=[(Xva, y_f3)],
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
    pred_va = m.predict_proba(Xva)[:, 1]
    s = ap(y_f3, pred_va)
    print(f"  {trial_name}: AP={s:.6f} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    m_full = lgb.LGBMClassifier(**base)
    m_full.set_params(n_estimators=m.best_iteration_)
    m_full.fit(X_all_tr, y, categorical_feature=cat_idx)
    pred_te = m_full.predict_proba(X_all_te)[:, 1]
    np.save(os.path.join(OUT, f"lgb_{trial_name}_test.npy"), pred_te)
    existing_f3[f"lgb_{trial_name}"] = pred_va
    existing[f"lgb_{trial_name}"] = pred_te
    print(f"    full: {time.time()-t0:.0f}s")

# ============================================================
# APPROACH 4: ISOLATION FOREST FEATURES + RETRAIN
# ============================================================
print("\n" + "="*60)
print("APPROACH 4: Isolation Forest anomaly features")
print("="*60)

num_feats = [c for c in mfeat if c not in cats]
X_num_tr = tr_v4[num_feats].fillna(0).astype(np.float32).values
X_num_te = te_v4[num_feats].fillna(0).astype(np.float32).values

iso = IsolationForest(n_estimators=300, max_samples=0.7, contamination=0.02, random_state=4242, n_jobs=-1)
iso.fit(X_num_tr[:100000])

iso_tr = -iso.score_samples(X_num_tr).reshape(-1, 1)
iso_te = -iso.score_samples(X_num_te).reshape(-1, 1)

# Add to features
X_iso_tr = np.hstack([X_all_tr.values, iso_tr])
X_iso_te = np.hstack([X_all_te.values, iso_te])
print(f"  IF features added: {X_iso_tr.shape[1]} total")

params_iso = {
    "boosting_type": "gbdt", "objective": "binary",
    "num_leaves": 127, "max_depth": -1,
    "learning_rate": 0.025, "n_estimators": 2500,
    "subsample_for_bin": 200000,
    "min_child_samples": 100, "subsample": 0.85, "colsample_bytree": 0.85,
    "reg_alpha": 0.5, "reg_lambda": 5.0,
    "random_state": 4242, "n_jobs": 8, "verbosity": -1
}

t0 = time.time()
m_iso = lgb.LGBMClassifier(**params_iso)
m_iso.fit(X_iso_tr[:-N_VAL], ytr,
          eval_set=[(X_iso_tr[-N_VAL:], y_f3)],
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
pred_iso_va = m_iso.predict_proba(X_iso_tr[-N_VAL:])[:, 1]
ap_iso = ap(y_f3, pred_iso_va)
print(f"  LGB+IF fold-3 AP: {ap_iso:.6f} ({time.time()-t0:.0f}s)")

t0 = time.time()
m_iso_full = lgb.LGBMClassifier(**params_iso)
m_iso_full.set_params(n_estimators=m_iso.best_iteration_)
m_iso_full.fit(X_iso_tr, y)
pred_iso_test = m_iso_full.predict_proba(X_iso_te)[:, 1]
np.save(os.path.join(OUT, "lgb_if_test.npy"), pred_iso_test)
existing_f3["lgb_if"] = pred_iso_va
existing["lgb_if"] = pred_iso_test
print(f"    full: {time.time()-t0:.0f}s")

# ============================================================
# APPROACH 5: TIME-DECAY WEIGHTED LGB
# ============================================================
print("\n" + "="*60)
print("APPROACH 5: Time-decay weighted LGB")
print("="*60)

# Load raw timestamps for time-decay
tr_raw_ts = pd.read_csv(r"train.csv", parse_dates=["timestamp"], usecols=["timestamp"])
ts_all = tr_raw_ts["timestamp"]
ts_min = ts_all.min()
max_days = (ts_all - ts_min).dt.total_seconds().max() / 86400
days_from_start = (ts_all - ts_min).dt.total_seconds() / 86400
train_days_vals = days_from_start.values[:len(ytr)]

# Exponential decay: half-life ~60 days
half_life = 60
decay = np.log(2) / half_life
time_weights = np.exp(decay * (max_days - days_from_start.values[:len(ytr)]))
time_weights = time_weights / time_weights.mean()  # normalize

t0 = time.time()
m_decay = lgb.LGBMClassifier(**{**params_iso, "random_state": 4247})
m_decay.fit(Xtr, ytr, sample_weight=time_weights,
            categorical_feature=cat_idx,
            eval_set=[(Xva, y_f3)],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
pred_decay_va = m_decay.predict_proba(Xva)[:, 1]
ap_decay = ap(y_f3, pred_decay_va)
print(f"  LGB+decay fold-3 AP: {ap_decay:.6f} ({time.time()-t0:.0f}s)")

t0 = time.time()
m_decay_full = lgb.LGBMClassifier(**{**params_iso, "random_state": 4247})
m_decay_full.set_params(n_estimators=m_decay.best_iteration_)
m_decay_full.fit(X_all_tr, y, sample_weight=time_weights, categorical_feature=cat_idx)
pred_decay_test = m_decay_full.predict_proba(X_all_te)[:, 1]
np.save(os.path.join(OUT, "lgb_decay_test.npy"), pred_decay_test)
existing_f3["lgb_decay"] = pred_decay_va
existing["lgb_decay"] = pred_decay_test
print(f"    full: {time.time()-t0:.0f}s")

# ============================================================
# APPROACH 6: HGB ON ALL FEATURES
# ============================================================
print("\n" + "="*60)
print("APPROACH 6: HGB on all features")
print("="*60)

from sklearn.ensemble import HistGradientBoostingClassifier as HGB

t0 = time.time()
m_hgb = HGB(max_iter=1000, learning_rate=0.03, max_leaf_nodes=127,
            min_samples_leaf=60, l2_regularization=5.0, max_bins=255,
            random_state=4242, early_stopping=False, validation_fraction=None)
m_hgb.fit(Xtr.values, ytr)
pred_hgb_va = m_hgb.predict_proba(Xva.values)[:, 1]
ap_hgb = ap(y_f3, pred_hgb_va)
print(f"  HGB fold-3 AP: {ap_hgb:.6f} ({time.time()-t0:.0f}s)")

t0 = time.time()
m_hgb_full = HGB(max_iter=m_hgb.n_iter_, learning_rate=0.03, max_leaf_nodes=127,
                 min_samples_leaf=60, l2_regularization=5.0, max_bins=255,
                 random_state=4242, early_stopping=False, validation_fraction=None)
m_hgb_full.fit(X_all_tr.values, y)
pred_hgb_test = m_hgb_full.predict_proba(X_all_te.values)[:, 1]
np.save(os.path.join(OUT, "hgb_all_test.npy"), pred_hgb_test)
existing_f3["hgb_all"] = pred_hgb_va
existing["hgb_all"] = pred_hgb_test
print(f"    full: {time.time()-t0:.0f}s")

# ============================================================
# APPROACH 7: CALIBRATED LGB (isotonic regression)
# ============================================================
print("\n" + "="*60)
print("APPROACH 7: Calibrated LGB ensemble")
print("="*60)

from sklearn.calibration import CalibratedClassifierCV

# Calibrate the best model
t0 = time.time()
m_cal = CalibratedClassifierCV(
    lgb.LGBMClassifier(**{**params_iso, "n_estimators": m_focal.best_iteration_, "random_state": 4248}),
    cv=5, method="isotonic"
)
m_cal.fit(Xtr.values, ytr)
pred_cal_va = m_cal.predict_proba(Xva.values)[:, 1]
ap_cal = ap(y_f3, pred_cal_va)
print(f"  Calibrated LGB fold-3 AP: {ap_cal:.6f} ({time.time()-t0:.0f}s)")

t0 = time.time()
m_cal_full = CalibratedClassifierCV(
    lgb.LGBMClassifier(**{**params_iso, "n_estimators": m_focal.best_iteration_, "random_state": 4248}),
    cv=5, method="isotonic"
)
m_cal_full.fit(X_all_tr.values, y)
pred_cal_test = m_cal_full.predict_proba(X_all_te.values)[:, 1]
np.save(os.path.join(OUT, "lgb_calibrated_test.npy"), pred_cal_test)
existing_f3["lgb_cal"] = pred_cal_va
existing["lgb_cal"] = pred_cal_test
print(f"    full: {time.time()-t0:.0f}s")

# ============================================================
# MEGA BLEND: Find best combination of ALL models
# ============================================================
print("\n" + "="*60)
print("MEGA BLEND SEARCH")
print("="*60)

print(f"\nAll {len(existing_f3)} fold-3 APs:")
for n, p in existing_f3.items():
    s = ap(y_f3, p)
    print(f"  {n}: {s:.6f}")

# Greedy forward selection
remaining = list(existing_f3.keys())
selected = []
best_score = 0

for step in range(min(6, len(remaining))):
    best_add = None
    for cand in remaining:
        trial = selected + [cand]
        if len(trial) == 1:
            score = ap(y_f3, existing_f3[trial[0]])
        else:
            # Simple average blend
            stacked = np.column_stack([existing_f3[m] for m in trial])
            # Grid search weights
            best_w = (0, None)
            from itertools import product
            # For speed, use equal weight
            blend = stacked.mean(axis=1)
            score = ap(y_f3, blend)
        if score > best_score:
            best_score = score
            best_add = cand

    if best_add:
        selected.append(best_add)
        remaining.remove(best_add)
        # Re-optimize weights for selected
        if len(selected) == 1:
            final_score = ap(y_f3, existing_f3[selected[0]])
        else:
            # Optimize weights
            best_ws = (0, None)
            n_mod = len(selected)
            for ws in product(np.arange(0.05, 0.95, 0.1), repeat=n_mod):
                if abs(sum(ws) - 1.0) > 0.05:
                    continue
                ws_norm = np.array(ws) / sum(ws)
                blend = sum(w * existing_f3[m] for w, m in zip(ws_norm, selected))
                s = ap(y_f3, blend)
                if s > best_ws[0]:
                    best_ws = (s, ws_norm.tolist())
            final_score = best_ws[0]
        print(f"  Step {step+1}: added {best_add}, selected={selected}, AP={final_score:.6f}")
    else:
        break

print(f"\nBest ensemble: {selected}")
print(f"Fold-3 AP: {best_score:.6f} (gain {best_score-0.518226:+.6f})")

# ============================================================
# BUILD FINAL SUBMISSIONS
# ============================================================
print("\n" + "="*60)
print("BUILDING SUBMISSIONS")
print("="*60)

ss = pd.read_csv(r"sample_submission.csv")

# Best ensemble test prediction (equal weight)
blend_test = sum(existing[m] for m in selected) / len(selected)
sub = ss.copy()
sub["fraud"] = blend_test
sub.to_csv(r"submission_file/rapidfire_best_ensemble.csv", index=False)
print(f"  rapidfire_best_ensemble: mean={blend_test.mean():.6f}")

# Also try all individual new models as solo submissions
for name in ["lgb_wide", "lgb_deep", "lgb_imbalance", "lgb_if", "lgb_decay", "lgb_cal", "lgb_focal", "hgb_all", "lgb_rank"]:
    if name in existing:
        sub = ss.copy()
        sub["fraud"] = existing[name]
        sub.to_csv(f"submission_file/rapid_{name}_submission.csv", index=False)
        print(f"  rapid_{name}: mean={existing[name].mean():.6f}")

# Save the champion + HGB_old + best new blend
for name_pair, w_new in [
    ("champ+hgb+best", {"champion": 0.3, "hgb_old": 0.3}),
]:
    blend = sum(w * existing[m] for m, w in w_new.items()) + (1-sum(w_new.values())) * blend_test
    sub = ss.copy()
    sub["fraud"] = blend
    sub.to_csv(f"submission_file/rapid_{name_pair}.csv", index=False)
    print(f"  rapid_{name_pair}: mean={blend.mean():.6f}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("DONE - ALL SUBMISSIONS READY")
