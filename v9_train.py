"""V9 STEP 3 - residual training with honest protocol + ablations.

Train rows: all rows EXCEPT fold-3 (i.e. rows 0:672477) -> predict fold-3.
Labels: y on train-prefix only. Fold-3 labels used ONLY for scoring.
OOF V7 input: PROVEN LEAKING -> EXCLUDED (v7 fold-3 preds trained on fold-3
excluded-leave-out; using them as a feature would leak fold-3 info).
Hence R4 == R3 by construction; reported explicitly.

Features per ablation:
  R1: GV4-lite only (34)
  R2: night-anomaly only (9)
  R3: GV4-lite + night (34)
  R0: V7 only (reference)
Model: spec LightGBM (no early stopping on fold-3; fixed rounds).
"""
import os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
import lightgbm as lgb
OUT = os.path.join(ROOT, "v9_out")
PARAMS = dict(objective="binary", metric="binary_logloss", n_estimators=500,
    learning_rate=0.025, num_leaves=15, max_depth=5, min_child_samples=150,
    feature_fraction=0.80, bagging_fraction=0.85, bagging_freq=1,
    reg_alpha=1.0, reg_lambda=12.0, verbosity=-1, random_state=20260915, n_jobs=-1)
def main():
    t0 = time.time()
    _, _, y, _, y3 = C.load_core(verbose=False)
    v7f3 = np.load(os.path.join(OUT, "_v7_f3.npy"))
    F_tr = np.load(os.path.join(OUT, "graph_v4_train.npy"))
    names = [l.strip() for l in open(os.path.join(OUT, "v9_graph_v4_feature_list.txt")) if l.strip()]
    gv_idx = list(range(24))
    night_idx = list(range(24, 33))
    assert len(names) == 34, len(names)
    tr_idx = np.arange(0, C.FOLD3_START)
    va_idx = C.IDX_F3
    ytr = y[tr_idx]
    cfgs = {"R1_gv4": gv_idx, "R2_night": night_idx, "R3_full": list(range(33))}
    FVA = F_tr[va_idx]
    res = {}
    imps = {}
    for tag, cols in cfgs.items():
        Xtr, Xva = F_tr[tr_idx][:, cols], FVA[:, cols]
        fn = [names[c] for c in cols]
        dtr = lgb.Dataset(Xtr, label=ytr)
        m = lgb.train(PARAMS, dtr, num_boost_round=500)
        p = m.predict(Xva)
        res[tag] = p
        imps[tag] = sorted(zip(fn, m.feature_importance(importance_type="gain")), key=lambda z: -z[1])[:12]
        print(f"[v9-tr] {tag} k={len(cols)} fold3 AP={C.ap(y3, p):.6f} corr_v7={pd.Series(p).corr(pd.Series(v7f3)):.4f} {time.time()-t0:.0f}s", flush=True)
        m.save_model(os.path.join(OUT, f"v9_model_{tag}.txt"))
        np.save(os.path.join(OUT, f"v9_fold3_{tag}.npy"), p)
    # July blocks
    tr_dates = pd.to_datetime(pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["timestamp"])["timestamp"])
    day = tr_dates.iloc[va_idx].dt.day.to_numpy()
    jb = {"early": (day <= 5), "mid": (day >= 6) & (day <= 10), "late": (day >= 11)}
    rows = []
    for tag, p in res.items():
        r = {"cfg": tag, "standalone_ap": round(float(C.ap(y3, p)), 6),
             "corr_v7": round(float(pd.Series(p).corr(pd.Series(v7f3))), 5)}
        for b, m in jb.items():
            r[f"july_{b}"] = round(float(C.ap(y3[m], p[m])), 6)
        r["worst_july"] = min(r["july_early"], r["july_mid"], r["july_late"])
        rows.append(r)
    r0 = {"cfg": "R0_v7", "standalone_ap": round(float(C.ap(y3, v7f3)), 6), "corr_v7": 1.0,
          "july_early": round(float(C.ap(y3[jb['early']], v7f3[jb['early']])), 6),
          "july_mid": round(float(C.ap(y3[jb['mid']], v7f3[jb['mid']])), 6),
          "july_late": round(float(C.ap(y3[jb['late']], v7f3[jb['late']])), 6)}
    r0["worst_july"] = min(r0["july_early"], r0["july_mid"], r0["july_late"])
    rows.append(r0)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "v9_ablation_scores.csv"), index=False)
    print(pd.DataFrame(rows).to_string(), flush=True)
    C.jdump({k: [[n, float(v)] for n, v in vv] for k, vv in imps.items()}, os.path.join(OUT, "v9_feature_importance.json"))
    print("DONE_TRAIN", flush=True)
if __name__ == "__main__":
    main()
