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
    return tr, [c for c in feats if c in cats]

def run(label, feats, iters=450, lr=0.1, depth=6):
    X, cat_cols = build(feats)
    Xtr = X.iloc[:-N_VAL].reset_index(drop=True)
    Xval = X.iloc[-N_VAL:].reset_index(drop=True)
    del X; gc.collect()
    model = CatBoostClassifier(
        iterations=iters, learning_rate=lr, depth=depth, l2_leaf_reg=3.0,
        loss_function="Logloss", eval_metric="Logloss",
        max_ctr_complexity=1, random_seed=2026,
        thread_count=6, verbose=200, used_ram_limit=3 * 1024**3)
    pool_tr = Pool(Xtr, y_all[:-N_VAL], cat_features=[Xtr.columns.get_loc(c) for c in cat_cols])
    pool_val = Pool(Xval, yval, cat_features=[Xval.columns.get_loc(c) for c in cat_cols])
    t0 = time.time()
    model.fit(pool_tr, eval_set=pool_val)
    pred = model.predict_proba(pool_val)[:, 1]
    s = ap(yval, pred)
    np.save(os.path.join(OUT, f"{label}_fold3_pred.npy"), pred)
    model.save_model(os.path.join(OUT, f"{label}_model.cbm"))
    json.dump({"AP": round(s, 6), "trees": model.tree_count_, "params": [iters, lr, depth]},
              open(os.path.join(OUT, f"{label}_meta.json"), "w"))
    print(f"[CB {label}] {len(feats)}f AP={s:.6f} trees={model.tree_count_} ({time.time()-t0:.0f}s)", flush=True)
    del model, pool_tr, pool_val, Xtr, Xval; gc.collect()

removed = ["customer_high_velocity_24h", "very_rapid_customer_switch_on_device",
           "rapid_customer_switch_on_device", "device_burst_6h", "device_burst_1h",
           "shared_device_recent_burst", "large_amount_and_customer_burst"]
pruned = [f for f in mfeat if f not in set(removed)] + b25_cols
full = mfeat + b25_cols
run("full110_fixed", full)
run("pruned103_fixed", pruned)
print("E3 FIXED DONE")