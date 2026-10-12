import numpy as np, pandas as pd, pickle, os, time, warnings, json, itertools
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
import lightgbm as lgb

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_RAPIDFIRE")
os.makedirs(OUT, exist_ok=True)

# LOAD
tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
te = pd.read_parquet(os.path.join(BASE, "feature_test_v4.parquet"))
mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
y = tr["fraud"].values.astype(int)
y_f3 = np.load(os.path.join(BASE, "Fold_3_v4_targets.npy"))
N = len(y_f3)

mt = pd.read_parquet(os.path.join(BASE, "AGENT_EXP_MEGA_V2", "train_new_features.parquet"))
me = pd.read_parquet(os.path.join(BASE, "AGENT_EXP_MEGA_V2", "test_new_features.parquet"))
mt.columns = [f"n_{c}" for c in mt.columns]
me.columns = [f"n_{c}" for c in me.columns]

Xtr_all = pd.concat([tr[mfeat].reset_index(drop=True), mt.reset_index(drop=True)], axis=1)
Xte_all = pd.concat([te[mfeat].reset_index(drop=True), me.reset_index(drop=True)], axis=1)
for c in cats:
    Xtr_all[c] = Xtr_all[c].fillna("missing").astype("category").cat.codes
    Xte_all[c] = Xte_all[c].fillna("missing").astype("category").cat.codes
for c in Xtr_all.columns:
    Xtr_all[c] = Xtr_all[c].fillna(-999).astype(np.float32)
    Xte_all[c] = Xte_all[c].fillna(-999).astype(np.float32)

cat_idx = [i for i, c in enumerate(Xtr_all.columns) if c in cats]
Xtr, Xva = Xtr_all.iloc[:-N], Xtr_all.iloc[-N:]
ytr = y[:-N]
print(f"Features: {Xtr_all.shape[1]}")

champ = np.load(BASE + r"/EXP_W2B_COST_SENSITIVE/best_extended_cost_global_fold3_pred.npy")
hgb1 = np.load(BASE + r"/AGENT_EXP_AUTO_HGB/HGB1_pred.npy")
hgbt = np.load(BASE + r"/AGENT_EXP_AUTO_HGB/HGB_tuned_pred.npy")

# MODEL 1: Deep + imbalance
print("\nModel 1: LGB deep imbalance...")
t0 = time.time()
m1 = lgb.LGBMClassifier(
    boosting_type="gbdt", objective="binary", num_leaves=200, max_depth=15,
    learning_rate=0.015, n_estimators=3000, subsample_for_bin=200000,
    min_child_samples=80, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=2.0, reg_lambda=10.0, scale_pos_weight=15,
    random_state=4242, n_jobs=8, verbosity=-1)
m1.fit(Xtr, ytr, categorical_feature=cat_idx,
       eval_set=[(Xva, y_f3)],
       callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)])
p1 = m1.predict_proba(Xva)[:, 1]
print(f"  AP={ap(y_f3, p1):.6f} iters={m1.best_iteration_} ({time.time()-t0:.0f}s)")
m1f = lgb.LGBMClassifier(**m1.get_params())
m1f.set_params(n_estimators=m1.best_iteration_)
m1f.fit(Xtr_all, y, categorical_feature=cat_idx)
p1t = m1f.predict_proba(Xte_all)[:, 1]
np.save(os.path.join(OUT, "lgb_deep_imp_test.npy"), p1t)
np.save(os.path.join(OUT, "lgb_deep_imp_f3.npy"), p1)

# MODEL 2: Patience + low lr
print("\nModel 2: LGB patience...")
t0 = time.time()
m2 = lgb.LGBMClassifier(
    boosting_type="gbdt", objective="binary", num_leaves=127, max_depth=-1,
    learning_rate=0.01, n_estimators=4000, subsample_for_bin=200000,
    min_child_samples=100, subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.5, reg_lambda=5.0, scale_pos_weight=8,
    random_state=4243, n_jobs=8, verbosity=-1)
m2.fit(Xtr, ytr, categorical_feature=cat_idx,
       eval_set=[(Xva, y_f3)],
       callbacks=[lgb.early_stopping(250), lgb.log_evaluation(500)])
p2 = m2.predict_proba(Xva)[:, 1]
print(f"  AP={ap(y_f3, p2):.6f} iters={m2.best_iteration_} ({time.time()-t0:.0f}s)")
m2f = lgb.LGBMClassifier(**m2.get_params())
m2f.set_params(n_estimators=m2.best_iteration_)
m2f.fit(Xtr_all, y, categorical_feature=cat_idx)
p2t = m2f.predict_proba(Xte_all)[:, 1]
np.save(os.path.join(OUT, "lgb_patience_test.npy"), p2t)
np.save(os.path.join(OUT, "lgb_patience_f3.npy"), p2)

# MODEL 3: GOSS
print("\nModel 3: LGB goss...")
t0 = time.time()
m3 = lgb.LGBMClassifier(
    boosting_type="goss", objective="binary", num_leaves=200, max_depth=12,
    learning_rate=0.02, n_estimators=2500, subsample_for_bin=200000,
    min_child_samples=80, subsample=1.0, colsample_bytree=1.0,
    reg_alpha=1.0, reg_lambda=8.0, scale_pos_weight=12,
    random_state=4244, n_jobs=8, verbosity=-1)
m3.fit(Xtr, ytr, categorical_feature=cat_idx,
       eval_set=[(Xva, y_f3)],
       callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)])
p3 = m3.predict_proba(Xva)[:, 1]
print(f"  AP={ap(y_f3, p3):.6f} iters={m3.best_iteration_} ({time.time()-t0:.0f}s)")
m3f = lgb.LGBMClassifier(**m3.get_params())
m3f.set_params(n_estimators=m3.best_iteration_)
m3f.fit(Xtr_all, y, categorical_feature=cat_idx)
p3t = m3f.predict_proba(Xte_all)[:, 1]
np.save(os.path.join(OUT, "lgb_goss_test.npy"), p3t)
np.save(os.path.join(OUT, "lgb_goss_f3.npy"), p3)

# MODEL 4: Hyper极端 - very high leaves + strong regularization
print("\nModel 4: LGB extreme...")
t0 = time.time()
m4 = lgb.LGBMClassifier(
    boosting_type="gbdt", objective="binary", num_leaves=31, max_depth=8,
    learning_rate=0.03, n_estimators=2000, subsample_for_bin=200000,
    min_child_samples=200, subsample=0.7, colsample_bytree=0.6,
    reg_alpha=5.0, reg_lambda=20.0, scale_pos_weight=10,
    min_split_gain=0.1,
    random_state=4245, n_jobs=8, verbosity=-1)
m4.fit(Xtr, ytr, categorical_feature=cat_idx,
       eval_set=[(Xva, y_f3)],
       callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)])
p4 = m4.predict_proba(Xva)[:, 1]
print(f"  AP={ap(y_f3, p4):.6f} iters={m4.best_iteration_} ({time.time()-t0:.0f}s)")
m4f = lgb.LGBMClassifier(**m4.get_params())
m4f.set_params(n_estimators=m4.best_iteration_)
m4f.fit(Xtr_all, y, categorical_feature=cat_idx)
p4t = m4f.predict_proba(Xte_all)[:, 1]
np.save(os.path.join(OUT, "lgb_extreme_test.npy"), p4t)
np.save(os.path.join(OUT, "lgb_extreme_f3.npy"), p4)

# MODEL 5: Binary + custom scale
print("\nModel 5: LGB alt seed...")
t0 = time.time()
m5 = lgb.LGBMClassifier(
    boosting_type="gbdt", objective="binary", num_leaves=180, max_depth=12,
    learning_rate=0.02, n_estimators=2500, subsample_for_bin=200000,
    min_child_samples=60, subsample=0.75, colsample_bytree=0.65,
    reg_alpha=3.0, reg_lambda=15.0, scale_pos_weight=20,
    random_state=9999, n_jobs=8, verbosity=-1)
m5.fit(Xtr, ytr, categorical_feature=cat_idx,
       eval_set=[(Xva, y_f3)],
       callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)])
p5 = m5.predict_proba(Xva)[:, 1]
print(f"  AP={ap(y_f3, p5):.6f} iters={m5.best_iteration_} ({time.time()-t0:.0f}s)")
m5f = lgb.LGBMClassifier(**m5.get_params())
m5f.set_params(n_estimators=m5.best_iteration_)
m5f.fit(Xtr_all, y, categorical_feature=cat_idx)
p5t = m5f.predict_proba(Xte_all)[:, 1]
np.save(os.path.join(OUT, "lgb_altseed_test.npy"), p5t)
np.save(os.path.join(OUT, "lgb_altseed_f3.npy"), p5)

# ============================================================
# BLEND
# ============================================================
print("\n" + "="*60)
print("ALL MODEL APS")
print("="*60)

all_f3 = {"champ": champ, "hgb600": hgb1, "hgbt": hgbt,
          "deep_imp": p1, "patience": p2, "goss": p3, "extreme": p4, "altseed": p5}
for n, p in all_f3.items():
    print(f"  {n}: {ap(y_f3, p):.6f}")

# Best 4-way
best = (0, None, None)
keys = list(all_f3.keys())
for combo in itertools.combinations(keys, 4):
    for w1 in np.arange(0.1, 0.5, 0.05):
        for w2 in np.arange(0.1, 0.4, 0.05):
            for w3 in np.arange(0.1, 0.3, 0.05):
                w4 = 1 - w1 - w2 - w3
                if w4 < 0.05: continue
                blend = w1*all_f3[combo[0]] + w2*all_f3[combo[1]] + w3*all_f3[combo[2]] + w4*all_f3[combo[3]]
                s = ap(y_f3, blend)
                if s > best[0]:
                    best = (s, list(combo), [round(w1,2), round(w2,2), round(w3,2), round(w4,2)])

print(f"\nBest 4-way: AP={best[0]:.6f} gain={best[0]-0.518226:+.6f}")
print(f"  models={best[1]} weights={best[2]}")

# Best 5-way
best5 = (0, None, None)
for combo in itertools.combinations(keys, 5):
    for w1 in np.arange(0.1, 0.35, 0.05):
        for w2 in np.arange(0.1, 0.3, 0.05):
            for w3 in np.arange(0.1, 0.25, 0.05):
                for w4 in np.arange(0.1, 0.2, 0.05):
                    w5 = 1 - w1 - w2 - w3 - w4
                    if w5 < 0.05: continue
                    blend = (w1*all_f3[combo[0]] + w2*all_f3[combo[1]] + w3*all_f3[combo[2]]
                           + w4*all_f3[combo[3]] + w5*all_f3[combo[4]])
                    s = ap(y_f3, blend)
                    if s > best5[0]:
                        best5 = (s, list(combo), [round(w1,2), round(w2,2), round(w3,2), round(w4,2), round(w5,2)])

print(f"Best 5-way: AP={best5[0]:.6f} gain={best5[0]-0.518226:+.6f}")
print(f"  models={best5[1]} weights={best5[2]}")

# ============================================================
# SAVE SUBMISSIONS
# ============================================================
print("\nSaving submissions...")
ss = pd.read_csv(r"sample_submission.csv")
all_t = {
    "champ": np.load(BASE + r"/FINAL_0518226_SUBMISSION/final_nested_0518226_test_pred.npy"),
    "hgb600": np.load(BASE + r"/AGENT_EXP_AUTO_HGB_VERIFY/HGB_tuned_fulldata_test_pred.npy"),
    "hgbt": np.load(BASE + r"/AGENT_EXP_AUTO_HGB_VERIFY/HGB_tuned_fulldata_test_pred.npy"),
    "deep_imp": p1t, "patience": p2t, "goss": p3t, "extreme": p4t, "altseed": p5t
}

# Best 4-way
w = best[2]; m = best[1]
blend4 = sum(ww*all_t[mm] for ww, mm in zip(w, m))
sub = ss.copy(); sub["fraud"] = blend4
sub.to_csv(r"submission_file/rapidfire_best4.csv", index=False)
print(f"  rapidfire_best4: mean={blend4.mean():.6f}")

# Best 5-way
w5 = best5[2]; m5_list = best5[1]
blend5 = sum(ww*all_t[mm] for ww, mm in zip(w5, m5_list))
sub = ss.copy(); sub["fraud"] = blend5
sub.to_csv(r"submission_file/rapidfire_best5.csv", index=False)
print(f"  rapidfire_best5: mean={blend5.mean():.6f}")

# Equal all
blend_eq = sum(all_t.values()) / len(all_t)
sub = ss.copy(); sub["fraud"] = blend_eq
sub.to_csv(r"submission_file/rapidfire_equal_all.csv", index=False)
print(f"  rapidfire_equal: mean={blend_eq.mean():.6f}")

# Solo new models
for name in ["deep_imp", "patience", "goss", "extreme", "altseed"]:
    sub = ss.copy(); sub["fraud"] = all_t[name]
    sub.to_csv(f"submission_file/rapid_{name}.csv", index=False)
    print(f"  rapid_{name}: mean={all_t[name].mean():.6f}")

print("\nDONE")
