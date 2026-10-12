import numpy as np, pandas as pd, os, time, pickle, json, gc
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score as ap
BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_CATBOOST_TUNED")
N_VAL = 59465
mfeat = pickle.load(open(os.path.join(BASE, "model_features_v4.pkl"), "rb"))
cats = pickle.load(open(os.path.join(BASE, "categorical_features_v4.pkl"), "rb"))
b25_cols = json.load(open(os.path.join(OUT, "b25_meta.json")))["cols"]
b25 = np.load(os.path.join(OUT, "b25_fold3_train.npy"))
y_all = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"), columns=["fraud"])["fraud"].values.astype(int)
yval = y_all[-N_VAL:]

def build(feats):
    tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"),
                         columns=[f for f in feats if f not in b25_cols])
    b25_idx = {c: j for j, c in enumerate(b25_cols)}
    for c in feats:
        if c in b25_cols:
            tr[c] = b25[:, b25_idx[c]]
    for c in cats:
        if c in tr:
            tr[c] = tr[c].fillna("missing").astype(str)
    for c in feats:
        if c not in cats:
            tr[c] = tr[c].astype(np.float32)
    cat_cols = [c for c in feats if c in cats]
    return tr, cat_cols

def run(label, feats, iters, lr, depth, od, ram=None, verbose=20, seed=2026):
    X, cat_cols = build(feats)
    Xtr = X.iloc[:-N_VAL].reset_index(drop=True)
    Xval = X.iloc[-N_VAL:].reset_index(drop=True)
    del X; gc.collect()
    kw = {}
    if ram: kw["used_ram_limit"] = ram
    model = CatBoostClassifier(
        iterations=iters, learning_rate=lr, depth=depth, l2_leaf_reg=3.0,
        loss_function="Logloss", eval_metric="PRAUC",
        max_ctr_complexity=1, random_seed=seed, od_type="Iter", od_wait=od,
        use_best_model=True, thread_count=6, verbose=verbose, **kw)
    pool_tr = Pool(Xtr, y_all[:-N_VAL], cat_features=[Xtr.columns.get_loc(c) for c in cat_cols])
    pool_val = Pool(Xval, yval, cat_features=[Xval.columns.get_loc(c) for c in cat_cols])
    t0 = time.time()
    model.fit(pool_tr, eval_set=pool_val)
    print("tree_count", model.tree_count_, "best_iter", model.get_best_iteration(), "elapsed", round(time.time()-t0,1))
    pred = model.predict_proba(pool_val)[:, 1]
    s = ap(yval, pred)
    np.save(os.path.join(OUT, f"{label}_fold3_pred.npy"), pred)
    json.dump({"AP": round(s, 6), "best_iter": model.get_best_iteration(),
               "trees": model.tree_count_}, open(os.path.join(OUT, f"{label}_meta.json"), "w"))
    print(f"[{label}] {len(feats)}f AP={s:.6f}", flush=True)
    del model, pool_tr, pool_val, Xtr, Xval; gc.collect()

removed = ["customer_high_velocity_24h", "very_rapid_customer_switch_on_device",
           "rapid_customer_switch_on_device", "device_burst_6h", "device_burst_1h",
           "shared_device_recent_burst", "large_amount_and_customer_burst"]
if not os.path.exists(os.path.join(OUT, "known_good_101_fold3_pred.npy")):
    run("known_good_101", mfeat, iters=450, lr=0.1, depth=6, od=120, ram=3 * 1024**3)
# ram test uses bytes: 3GB
run("full110_known", mfeat + b25_cols, iters=450, lr=0.1, depth=6, od=120, ram=3 * 1024**3)
print("DONE")