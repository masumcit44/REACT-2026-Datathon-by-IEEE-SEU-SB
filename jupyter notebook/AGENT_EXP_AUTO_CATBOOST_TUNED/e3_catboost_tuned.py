import numpy as np, pandas as pd, os, time, pickle, json
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score as ap

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_CATBOOST_TUNED")
os.makedirs(OUT, exist_ok=True)
N_VAL = 59465

mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
b25_cols = json.load(open(os.path.join(OUT, "b25_meta.json")))["cols"]
b25 = np.load(os.path.join(OUT, "b25_fold3_train.npy"))
y_all = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"), columns=["fraud"])["fraud"].values.astype(int)
yval = y_all[-N_VAL:]
assert np.array_equal(yval.astype(np.int8), np.load(os.path.join(BASE, "Fold_3_v4_targets.npy")).astype(np.int8))

tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"))

def run(label, feats, iters=2000, lr=0.03, depth=8, od=120):
    out_pred = os.path.join(OUT, f"catboost_{label}_fold3_pred.npy")
    meta_f = os.path.join(OUT, f"catboost_{label}_meta.json")
    if os.path.exists(out_pred):
        return [round(x, 6) for x in json.load(open(meta_f)).values()]
    t0 = time.time()
    X = tr[[f for f in feats if f not in b25_cols]].copy()
    b25_idx = {c: j for j, c in enumerate(b25_cols)}
    for c in feats:
        if c in b25_cols:
            X[c] = b25[:, b25_idx[c]]
    for c in cats:
        if c in X:
            X[c] = X[c].fillna("missing").astype(str)
    numeric = [c for c in feats if c not in cats]
    for c in numeric:
        X[c] = X[c].astype(np.float32)
    cat_feats = [c for c in feats if c in cats]
    Xtr, Xval = X.iloc[:-N_VAL], X.iloc[-N_VAL:]
    del X
    import gc; gc.collect()
    model = CatBoostClassifier(
        iterations=iters, learning_rate=lr, depth=depth, l2_leaf_reg=3.0,
        loss_function="Logloss", eval_metric="PRAUC",
        max_ctr_complexity=1, random_seed=2026, od_type="Iter", od_wait=od,
        use_best_model=True, thread_count=6, verbose=100,
        used_ram_limit=2048 * 1024 * 1024)
    pool_tr = Pool(Xtr, y_all[:-N_VAL], cat_features=[Xtr.columns.get_loc(c) for c in cat_feats])
    pool_val = Pool(Xval, yval, cat_features=[Xval.columns.get_loc(c) for c in cat_feats])
    model.fit(pool_tr, eval_set=pool_val)
    bi = model.get_best_iteration()
    pred = model.predict_proba(pool_val)[:, 1]
    s = ap(yval, pred)
    np.save(out_pred, pred)
    model.save_model(os.path.join(OUT, f"catboost_{label}_model.cbm"))
    imp = pd.DataFrame({"feature": feats,
                        "gain": np.array(model.get_feature_importance(pool_tr, type="PredictionValuesChange")).ravel()}).sort_values("gain", ascending=False)
    imp.to_csv(os.path.join(OUT, f"catboost_{label}_importance.csv"), index=False)
    json.dump({"AP": round(s, 6), "best_iter": bi, "features": len(feats)}, open(meta_f, "w"), indent=1)
    print(f"[CATBOOST {label}] {len(feats)}f AP={s:.6f} best_iter={bi} ({time.time()-t0:.1f}s)", flush=True)
    del model, pool_tr, pool_val, Xtr, Xval; gc.collect()
    return [round(s, 6), bi]

removed = ["customer_high_velocity_24h", "very_rapid_customer_switch_on_device",
           "rapid_customer_switch_on_device", "device_burst_6h", "device_burst_1h",
           "shared_device_recent_burst", "large_amount_and_customer_burst"]
pruned = [f for f in mfeat if f not in set(removed)] + b25_cols
full = mfeat + b25_cols
run("FULL110_tuned", full)
run("PRUNED103_tuned", pruned)
print("E3 DONE")