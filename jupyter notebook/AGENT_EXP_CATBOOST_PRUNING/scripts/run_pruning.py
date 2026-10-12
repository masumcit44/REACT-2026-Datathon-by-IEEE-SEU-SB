import numpy as np
import pandas as pd
import os
import json
import time
import pickle
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import average_precision_score as ap

BASE = r"D:\Desktop\React datathon"
NB = os.path.join(BASE, "jupyter notebook")
OUT = os.path.join(NB, "AGENT_EXP_CATBOOST_PRUNING")
os.makedirs(OUT, exist_ok=True)

SEED = 2026
ITERATIONS = 450
LEARNING_RATE = 0.1
DEPTH = 6
L2 = 3.0
EVAL_FREQ = 25
MAX_CTR_COMPLEXITY = 1

tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))
y_all = tr["fraud"].astype(int).values
N_VAL = 59465
y_val_all = y_all[-N_VAL:]

mfeat = pickle.load(open(os.path.join(NB, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(NB, "categorical_features_v4.pkl"), "rb"))
removed = ["customer_high_velocity_24h", "very_rapid_customer_switch_on_device",
           "rapid_customer_switch_on_device", "device_burst_6h", "device_burst_1h",
           "shared_device_recent_burst", "large_amount_and_customer_burst"]

pruned_features = [f for f in mfeat if f not in set(removed)]

def prep(df, feats):
    X = df[feats].copy()
    nan_counts = {c: int(X[c].isna().sum()) for c in feats if X[c].isna().sum() > 0}
    cat_feats = [c for c in feats if c in cats]
    for c in cat_feats:
        X.loc[:, c] = X[c].fillna("MISSING").astype(str)
    return X, cat_feats, nan_counts

results = {}
for label, feats in [("UNPRUNED_101F", mfeat), ("PRUNED_94F", pruned_features)]:
    t0 = time.time()
    X, cat_feats, nan_counts = prep(tr, feats)
    Xtr, Xval = X.iloc[:-N_VAL], X.iloc[-N_VAL:]
    ytr, yval = y_all[:-N_VAL], y_all[-N_VAL:]
    print(f"[{label}] features={len(feats)} train={len(Xtr)} val={len(Xval)} "
          f"cat_feats={cat_feats}", flush=True)
    if nan_counts:
        print(f"[{label}] NaN filled: {nan_counts}", flush=True)
    assert np.array_equal(yval.astype(np.int8), np.load(os.path.join(NB, "Fold_3_v4_targets.npy")).astype(np.int8)), "val target alignment broken!"

    pool_tr = Pool(Xtr, ytr, cat_features=cat_feats)
    pool_val = Pool(Xval, yval, cat_features=cat_feats)

    model = CatBoostClassifier(
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        l2_leaf_reg=L2,
        loss_function="Logloss",
        eval_metric="PRAUC",
        max_ctr_complexity=MAX_CTR_COMPLEXITY,
        random_seed=SEED,
        od_type="Iter",
        od_wait=120,
        use_best_model=True,
        thread_count=8,
        verbose=50,
    )
    model.fit(pool_tr, eval_set=pool_val)
    best_iter = model.get_best_iteration()
    pred = model.predict_proba(pool_val)[:, 1]
    score = ap(yval, pred)
    elapsed = (time.time() - t0) / 60.0

    # eval curve
    curve = model.eval_metrics(pool_val, metrics=["PRAUC"], eval_period=EVAL_FREQ)
    ap_curve = curve["PRAUC"]

    # importance
    imp = pd.DataFrame({
        "feature": feats,
        "importance_pvchange": np.array(model.get_feature_importance(pool_tr, type="PredictionValuesChange")).ravel(),
        "importance_loss": np.array(model.get_feature_importance(pool_tr, type="LossFunctionChange")).ravel(),
    }).sort_values("importance_pvchange", ascending=False)

    tag = "unpruned" if label == "UNPRUNED_101F" else "pruned"
    np.save(os.path.join(OUT, f"catboost_{tag}_fold3_pred.npy"), pred)
    np.save(os.path.join(OUT, "fold3_targets.npy"), yval)
    model.save_model(os.path.join(OUT, f"catboost_{tag}_model.cbm"))
    imp.to_csv(os.path.join(OUT, f"catboost_{tag}_importance.csv"), index=False)
    pd.DataFrame({
        "iteration": list(range(len(ap_curve))),
        "eval_iteration": np.arange(0, len(ap_curve)) * EVAL_FREQ,
        "AP": ap_curve,
    }).to_csv(os.path.join(OUT, f"catboost_{tag}_checkpoints.csv"), index=False)
    results[label] = {
        "label": label, "feature_count": len(feats), "best_iteration": best_iter,
        "fold3_AP": float(score), "runtime_minutes": round(elapsed, 3),
        "nan_filled": nan_counts, "seed": SEED,
        "iterations": ITERATIONS, "learning_rate": LEARNING_RATE, "depth": DEPTH, "l2_leaf_reg": L2,
    }
    print(f"[{label}] FINAL AP={score:.6f} best_iter={best_iter} runtime={elapsed:.2f} min", flush=True)

summary_df = pd.DataFrame(results.values())
summary_df.to_csv(os.path.join(OUT, "pruning_summary.csv"), index=False)

pruned_ap = results["PRUNED_94F"]["fold3_AP"]
unpruned_ap = results["UNPRUNED_101F"]["fold3_AP"]
gain = pruned_ap - unpruned_ap
final = {
    "unpruned_fold3_AP": unpruned_ap,
    "pruned_fold3_AP": pruned_ap,
    "pruning_gain": gain,
    "decision": "REJECT PRUNING (gain below +0.001)" if gain < 0.001 else "PASS PRUNING GATE (gain >= +0.001)",
    "blend_baseline_global_AP": 0.5182256022283107,
}
json.dump(final, open(os.path.join(OUT, "pruning_final_result.json"), "w"), indent=2)
print(json.dumps(final, indent=2), flush=True)
print("CATBOOST PRUNING FINAL RESULT", flush=True)