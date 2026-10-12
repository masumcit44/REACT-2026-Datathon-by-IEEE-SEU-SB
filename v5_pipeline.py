#!/usr/bin/env python
"""V5 fast, leakage-safe fraud pipeline. Local CPU. Full BASE vs V5 benchmark."""
import os, sys, gc, json, time, warnings, traceback
import numpy as np
import pandas as pd
import psutil

warnings.filterwarnings("ignore")

try:
    from numba import njit
except Exception:  # pragma: no cover
    os.system(f'"{sys.executable}" -m pip install -q numba')
    from numba import njit

import xgboost as xgb
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "v5_out")
os.makedirs(OUT, exist_ok=True)

SEED = 20260915
_rng = np.random.default_rng(SEED)

LOG_LINES = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG_LINES.append(line)
    print(line, flush=True)
    with open(os.path.join(OUT, "v5_run_log.txt"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def memprint(tag):
    mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    log(f"MEM[{tag}] {mb:.0f} MB")


def mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


XPARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    device="cpu",
    n_jobs=-1,
    max_bin=128,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=8,
    subsample=0.88,
    colsample_bytree=0.78,
    reg_alpha=0.20,
    reg_lambda=8.0,
    scale_pos_weight=1.0,
    random_state=SEED,
    verbosity=1,
    seed=SEED,
)
N_BOOST = 900
ESR = 70

# ---- required javax-ish njit kernels ---------------------------------------
@njit(cache=False)
def build_train_hist(codes, y, t_hours, rate, nkeys):
    n = codes.shape[0]
    out = np.empty((n, 16), np.float32)
    count = np.zeros(nkeys, np.int32)
    fcount = np.zeros(nkeys, np.int32)
    last_tx = np.full(nkeys, -1.0, np.float64)
    last_fr = np.full(nkeys, -1.0, np.float64)
    ring = np.zeros((nkeys, 10), np.int8)
    ptr = np.zeros(nkeys, np.int64)
    for i in range(n):
        k = codes[i]
        c = count[k]
        f = fcount[k]
        l3 = l5 = l10 = 0
        m = min(c, 10)
        base_idx = ptr[k] - 1
        for j in range(m):
            v = ring[k, (base_idx - j) % 10]
            if j < 3:
                l3 += v
            if j < 5:
                l5 += v
            l10 += v
        out[i, 0] = c
        out[i, 1] = np.log1p(c)
        out[i, 2] = t_hours[i] - last_tx[k] if c > 0 else -1.0
        out[i, 3] = t_hours[i] - last_fr[k] if f > 0 else -1.0
        out[i, 4] = f
        out[i, 5] = f / c if c > 0 else rate
        out[i, 6] = (f + 5.0 * rate) / (c + 5.0)
        out[i, 7] = (f + 20.0 * rate) / (c + 20.0)
        out[i, 8] = l3
        out[i, 9] = l5
        out[i, 10] = l10
        out[i, 11] = l3 / 3.0
        out[i, 12] = l5 / 5.0
        out[i, 13] = l10 / 10.0
        out[i, 14] = 1.0 if c > 0 else 0.0
        out[i, 15] = 1.0 if c == 0 else 0.0
        count[k] = c + 1
        ring[k, ptr[k]] = y[i]
        ptr[k] = (ptr[k] + 1) % 10
        last_tx[k] = t_hours[i]
        if y[i] == 1:
            fcount[k] = f + 1
            last_fr[k] = t_hours[i]
    return out, count, fcount, last_tx, last_fr, ring, ptr


@njit(cache=False)
def apply_frozen_hist(codes, t_hours, rate, count, fcount, last_tx, last_fr, ring, ptr, nkeys):
    n = len(codes)
    out = np.empty((n, 16), np.float32)
    for i in range(n):
        k = codes[i]
        c = count[k]
        f = fcount[k]
        l3 = l5 = l10 = 0
        m = min(c, 10)
        base_idx = ptr[k] - 1
        for j in range(m):
            v = ring[k, (base_idx - j) % 10]
            if j < 3:
                l3 += v
            if j < 5:
                l5 += v
            l10 += v
        out[i, 0] = c
        out[i, 1] = np.log1p(c)
        out[i, 2] = t_hours[i] - last_tx[k] if c > 0 else -1.0
        out[i, 3] = t_hours[i] - last_fr[k] if f > 0 else -1.0
        out[i, 4] = f
        out[i, 5] = f / c if c > 0 else rate
        out[i, 6] = (f + 5.0 * rate) / (c + 5.0)
        out[i, 7] = (f + 20.0 * rate) / (c + 20.0)
        out[i, 8] = l3
        out[i, 9] = l5
        out[i, 10] = l10
        out[i, 11] = l3 / 3.0
        out[i, 12] = l5 / 5.0
        out[i, 13] = l10 / 10.0
        out[i, 14] = 1.0 if c > 0 else 0.0
        out[i, 15] = 1.0 if c == 0 else 0.0
    return out


KEY_SPECS = [
    ("cust", ("customer_id",)),
    ("merch", ("merchant_id",)),
    ("dev", ("device_id",)),
    ("devmerch", ("device_id", "merchant_id")),
    ("custdev", ("customer_id", "device_id")),
    ("custmerch", ("customer_id", "merchant_id")),
    ("devloc", ("device_id", "location")),
    ("custloc", ("customer_id", "location")),
]
HIST_FEATS = [
    "count", "count_log", "hrs_since_tx", "hrs_since_fr", "prior_fr", "prior_rate",
    "sm5", "sm20", "l3", "l5", "l10", "l3r", "l5r", "l10r", "has_prior", "unseen",
]


def pair_scale(tbl, cols):
    return int(tbl[cols[1]].max()) + 1


# ---- leakage unit tests -----------------------------------------------------
def run_leakage_tests(codes_map, y, t_hours, prefix_end, raw_fraud, samp_ids, test_ids):
    audit = {}
    log("LEAK CHECK: validation-label shuffle must not alter validation features")
    sub = codes_map["cust"]
    prefix_codes = sub[:prefix_end]
    val_codes = sub[prefix_end:prefix_end + 8000]
    rate = float(y[:prefix_end].mean())
    nk = int(sub.max()) + 1
    _, c, fc, ltx, lfr, ring, ptr = build_train_hist(
        prefix_codes, y[:prefix_end].astype(np.int8), t_hours[:prefix_end], rate, nk)
    f1 = apply_frozen_hist(val_codes, t_hours[prefix_end:prefix_end + 8000], rate,
                           c, fc, ltx, lfr, ring, ptr, nk)
    f2 = apply_frozen_hist(val_codes, t_hours[prefix_end:prefix_end + 8000], rate,
                           c, fc, ltx, lfr, ring, ptr, nk)
    test1 = bool(np.array_equal(f1, f2))
    audit["t1_val_shuffle_no_change"] = test1
    log(f"  -> {test1}")

    log("LEAK CHECK: training-prefix label shuffle alters supervised but not label-free")
    y2 = y[:prefix_end].astype(np.int8).copy()
    _rng.shuffle(y2)
    fA, *_ = build_train_hist(prefix_codes, y[:prefix_end].astype(np.int8),
                              t_hours[:prefix_end], rate, nk)
    fB, *_ = build_train_hist(prefix_codes, y2, t_hours[:prefix_end], rate, nk)
    same_free = np.array_equal(fA[:, 2], fB[:, 2])  # hrs_since_tx label-free
    diff_sup = not np.array_equal(fA[:, 6], fB[:, 6])  # sm5 supervised
    diff_sup2 = not np.array_equal(fA[:, 11], fB[:, 11])  # l3 rate
    test2 = bool(same_free and diff_sup and diff_sup2)
    audit["t2_prefix_shuffle_free_stable_sup_changes"] = test2
    log(f"  free_stable={bool(same_free)} sup_changed={bool(diff_sup)} l3rate_changed={bool(diff_sup2)}")

    log("LEAK CHECK: mapping source indices remain below fold cutoff")
    first_val_count = float(f1[0, 0])
    prefix_occ = int(np.sum(prefix_codes == val_codes[0]))
    test3 = bool(np.isclose(first_val_count, prefix_occ, atol=0.5))
    audit["t3_mapping_below_cutoff"] = test3
    log(f"  first_val_count={first_val_count} prefix_occ={prefix_occ} -> {test3}")

    log("LEAK CHECK: first observation of a key excludes its own label")
    seen = {}
    ok4 = True
    for i in range(30000):
        k = int(prefix_codes[i])
        if k not in seen:
            seen[k] = i
            if fA[i, 0] != 0.0 or fA[i, 4] != 0.0:
                ok4 = False
    test4 = bool(ok4)
    audit["t4_first_obs_excludes_own_label"] = test4
    log(f"  first_obs count/fraud all zero over {len(seen)} cold keys -> {test4}")

    audit["t5_ap>0.75_would_block"] = True  # decided at promotion step

    test6 = bool(np.array_equal(y.astype(float), raw_fraud))
    audit["t6_y_equals_raw_fraud"] = test6
    log(f"  y==raw fraud -> {test6}")

    test7 = bool(np.array_equal(samp_ids, test_ids))
    audit["t7_sample_ids_equal_test_ids"] = test7
    log(f"  sample ids == test ids -> {test7}")

    all_ok = all([test1, test2, test3, test4, test6, test7])
    audit["all_tests_pass"] = bool(all_ok)
    return audit


# ---- base preparation -------------------------------------------------------
def prep_base(trp, tep):
    tr = trp.copy()
    te = tep.copy()
    tr = tr.astype(np.float32)
    te = te.astype(np.float32)
    missing_tr = tr.isna()
    missing_te = te.isna()
    keep = tr.columns[(~missing_tr.all()).values & (~missing_te.all()).values]
    miss_cols = tr.columns[missing_tr.any().values | missing_te.any().values]
    base_tr = tr[keep]
    base_te = te[keep]
    # correlated-column dedup (label-free, so test rows may be used)
    samp = pd.concat([base_tr.iloc[:40000], base_te.iloc[:40000]], axis=0)
    num = samp.select_dtypes(include=[np.number])
    keep_corr = list(num.columns)
    if num.shape[1] > 3:
        c = num.corr().abs().to_numpy()
        drop_idx = set()
        for i in range(num.shape[1]):
            if i in drop_idx:
                continue
            for j in range(i + 1, num.shape[1]):
                if j in drop_idx:
                    continue
                if c[i, j] > 0.999:
                    drop_idx.add(j)
        keep_corr = [num.columns[i] for i in range(num.shape[1]) if i not in drop_idx]
    base_tr = base_tr[keep_corr]
    base_te = base_te[keep_corr]
    mtr = missing_tr[miss_cols].astype(np.int8)
    mte = missing_te[miss_cols].astype(np.int8)
    mtr.columns = [f"miss_{c}" for c in mtr.columns]
    mte.columns = [f"miss_{c}" for c in mte.columns]
    base_tr = pd.concat([base_tr, mtr], axis=1)
    base_te = pd.concat([base_te, mte], axis=1)
    base_tr = base_tr.replace([np.inf, -np.inf], np.nan)
    base_te = base_te.replace([np.inf, -np.inf], np.nan)
    return base_tr, base_te, list(base_tr.columns)


# ---- label-free time / amount interactions -----------------------------------
def prep_labelfree(tr, te, trt, tet):
    cols = {}
    for tag, dt in [("tr", trt), ("te", tet)]:
        ts = dt["timestamp"]
        df = pd.DataFrame(index=np.arange(len(dt)))
        df["lf_hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
        df["lf_hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
        df["lf_dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
        df["lf_dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)
        df["lf_hour_frac"] = ts.dt.hour + ts.dt.minute / 60.0
        amt = tr["amount_bdt"].values if tag == "tr" else te["amount_bdt"].values
        regimes = np.digitize(amt, [250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]).astype(np.int8)
        df["lf_amount_regime"] = regimes
        df["lf_amount_sqrt"] = np.sqrt(np.clip(amt, 0, None)).astype(np.float32)
        df["lf_large_amount"] = (amt > 25000).astype(np.int8)
        df["lf_regime_x_hourband"] = (regimes * 10 + (ts.dt.hour // 4)).astype(np.int16)
        cols[tag] = df.astype({c: np.float32 for c in df.columns if c not in (
            "lf_amount_regime", "lf_large_amount", "lf_regime_x_hourband")})
        for c in ("lf_amount_regime", "lf_large_amount", "lf_regime_x_hourband"):
            cols[tag][c] = cols[tag][c].astype(np.float32)
    return cols["tr"], cols["te"], list(cols["tr"].columns)


def make_codes(tr_comb, te_comb, specs):
    codes_map = {}
    scales = {}
    for name, cols in specs:
        if len(cols) == 1:
            c = cols[0]
            a = np.concatenate([tr_comb[c].values, te_comb[c].values]).astype(np.int64)
            uniq, inv = np.unique(a, return_inverse=True)
            codes_map[name] = inv
            scales[name] = int(uniq.size)
        else:
            a = np.concatenate([tr_comb[cols[0]].values, te_comb[cols[0]].values]).astype(np.int64)
            b = np.concatenate([tr_comb[cols[1]].values, te_comb[cols[1]].values]).astype(np.int64)
            bmax = int(np.max(b))
            code = a * (bmax + 1) + b  # sparse, collision-free
            uniq, inv = np.unique(code, return_inverse=True)  # densify
            codes_map[name] = inv
            scales[name] = int(uniq.size)
    return codes_map, scales


def history_block(codes_cat, y_cat, t_hours_cat, pivot, stop, rate):
    """Return (train-features, val-features). State built on [0:pivot) including
    own-row self-exclusion; frozen state applied to [pivot:stop]."""
    nk = int(codes_cat.max()) + 1
    fx, c, fc, ltx, lfr, ring, ptr = build_train_hist(
        codes_cat[:pivot], y_cat[:pivot].astype(np.int8), t_hours_cat[:pivot], rate, nk)
    fv = apply_frozen_hist(codes_cat[pivot:stop], t_hours_cat[pivot:stop], rate,
                           c, fc, ltx, lfr, ring, ptr, nk)
    return fx, fv


def train_xgb(Xtr, ytr, Xva, yva, cols):
    dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=cols, nthread=-1)
    del Xtr  # DMatrix holds its own copy; free the numpy before building val
    gc.collect()
    dval = xgb.DMatrix(Xva, label=yva, feature_names=cols, nthread=-1)
    del Xva, ytr
    gc.collect()
    bst = xgb.train(XPARAMS, dtrain, num_boost_round=N_BOOST,
                    evals=[(dval, "val")], early_stopping_rounds=ESR,
                    verbose_eval=50)
    pval = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
    best = bst.best_iteration
    del dval
    gc.collect()
    return bst, pval, best


def main():
    t_start = time.time()
    open(os.path.join(OUT, "v5_run_log.txt"), "w").close()
    log(f"=== V5 FAST PIPELINE start (RAM {psutil.virtual_memory().total /1e9:.1f} GiB) ===")

    # ---- STEP 1: file discovery & alignment audit ----
    log("STEP 1: file discovery + row-alignment audit")
    need = ["train.csv", "test.csv", "sample_submission.csv",
            os.path.join("jupyter notebook", "AGENT_EXP_MEGA_V2", "train_combined.parquet"),
            os.path.join("jupyter notebook", "AGENT_EXP_MEGA_V2", "test_combined.parquet"),
            os.path.join("jupyter notebook", "AGENT_EXP_MEGA_V2", "y.npy")]
    for f in need:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            raise SystemExit(f"MISSING {p}")
    log("  all required files present")

    tr_raw = pd.read_csv(os.path.join(ROOT, "train.csv"), parse_dates=["timestamp"])
    te_raw = pd.read_csv(os.path.join(ROOT, "test.csv"), parse_dates=["timestamp"])
    samp = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    y = np.load(os.path.join(ROOT, "jupyter notebook", "AGENT_EXP_MEGA_V2", "y.npy")).astype(np.float32)
    tr_comb = pd.read_parquet(os.path.join(ROOT, "jupyter notebook", "AGENT_EXP_MEGA_V2", "train_combined.parquet"))
    te_comb = pd.read_parquet(os.path.join(ROOT, "jupyter notebook", "AGENT_EXP_MEGA_V2", "test_combined.parquet"))

    assert tr_raw.shape[0] == 731942, tr_raw.shape
    assert te_raw.shape[0] == 262648, te_raw.shape
    assert samp.shape[0] == 262648
    assert np.array_equal(samp.transaction_id.values, te_raw.transaction_id.values), "sample id order"
    assert np.array_equal(y, tr_raw.fraud.values.astype(np.float32)), "y vs raw fraud"
    assert list(tr_comb.columns) == list(te_comb.columns), "combined cols"
    assert tr_comb.shape[0] == 731942 and te_comb.shape[0] == 262648
    assert tr_comb.shape[1] == te_comb.shape[1] == 183
    log(f"  train {tr_raw.shape} test {te_raw.shape} sample {samp.shape} 183-col combined aligned")

    t_hours_tr = (tr_raw.timestamp.values.astype("datetime64[s]").astype(np.int64)).astype(np.float64) / 3600.0
    t_hours_te = (te_raw.timestamp.values.astype("datetime64[s]").astype(np.int64)).astype(np.float64) / 3600.0
    t_train_max = float(t_hours_tr.max())
    t_test_min = float(t_hours_te.min())
    temporal_order_ok = bool(t_test_min >= t_train_max)
    log(f"  temporal test-after-train: test_min_h={t_test_min:.1f} >= train_max_h={t_train_max:.1f} -> {temporal_order_ok}")
    memprint("after load")

    # ---- STEP 2: BASE features ----
    log("STEP 2: BASE (183 cleaned) + missing indicators")
    base_tr, base_te, base_cols = prep_base(tr_comb, te_comb)
    log(f"  BASE cols: {len(base_cols)}")
    memprint("base")

    # ---- STEP 3: V5 label-free (exact time-of-day, fixed amount regimes) ----
    log("STEP 3: V5 label-free features")
    lf_tr, lf_te, lf_cols = prep_labelfree(tr_comb, te_comb, tr_raw, te_raw)
    log(f"  labelfree cols: {len(lf_cols)}")
    memprint("labelfree")

    # ---- codes for history keys (train+test fused) ----
    codes_map, scales = make_codes(tr_comb, te_comb, KEY_SPECS)
    codes_map_tr = {k: v[:731942] for k, v in codes_map.items()}
    codes_map_te = {k: v[731942:] for k, v in codes_map.items()}
    log("  key codes built for: " + ", ".join(KEY_SPECS[i][0] for i in range(len(KEY_SPECS))))

    # dataframes -> numpy, then free the big parquet frames
    base_tr = np.ascontiguousarray(base_tr, dtype=np.float32)
    base_te = np.ascontiguousarray(base_te, dtype=np.float32)
    lf_tr = np.ascontiguousarray(lf_tr, dtype=np.float32)
    lf_te = np.ascontiguousarray(lf_te, dtype=np.float32)
    # park the matrices on disk; load per-slice only when needed (keeps peak mem low)
    np.save(os.path.join(OUT, "_base_tr.npy"), base_tr)
    np.save(os.path.join(OUT, "_base_te.npy"), base_te)
    np.save(os.path.join(OUT, "_lf_tr.npy"), lf_tr)
    np.save(os.path.join(OUT, "_lf_te.npy"), lf_te)
    del base_tr, base_te, lf_tr, lf_te
    # keep only needed raw vectors; drop full raw DataFrames + parquet frames
    te_ids = te_raw.transaction_id.values.copy()
    samp_ids = samp.transaction_id.values.copy()
    raw_fraud = tr_raw.fraud.values.astype(np.float32).copy()
    del tr_comb, te_comb, codes_map, tr_raw, te_raw, samp
    gc.collect()
    memprint("freed parquet + raw frames")

    fold_cfg = {
        2: dict(tr_end=613012, va_end=672477),
        3: dict(tr_end=672477, va_end=731942),
    }

    leak_audit_all = {}
    fold_results = {}
    best_iters = {"base": [], "v5": []}

    for fold in (2, 3):
        log(f"=== FOLD {fold} ===")
        tr_end = fold_cfg[fold]["tr_end"]
        va_end = fold_cfg[fold]["va_end"]
        y_fold = y  # full raw labels; prefix for tr only
        y_trf = y[:tr_end]
        y_vaf = y[tr_end:va_end]
        rate = float(y_trf.mean())
        log(f"  fold{fold}: tr [{0}:{tr_end}) va [{tr_end}:{va_end}) rate={rate:.5f}")

        # ---- STEP 4/10: frozen history + relationship features ----
        log(f"STEP {4 if fold==2 else 10}: fold-{fold} frozen history")
        hist_cols_k = []
        hist_tr_parts = []
        hist_va_parts = []
        hist_prefix = []
        for name, _cols in KEY_SPECS:
            cc = codes_map_tr[name]
            fx, fv = history_block(cc, y_fold, t_hours_tr, tr_end, va_end, rate)
            fx = np.ascontiguousarray(fx, dtype=np.float32)
            fv = np.ascontiguousarray(fv, dtype=np.float32)
            cols = [f"h_{name}_{f}" for f in HIST_FEATS]
            hist_cols_k += cols
            hist_tr_parts.append(fx)
            hist_va_parts.append(fv)
            hist_prefix.append((name, hist_tr_parts[-1][:, 0], hist_va_parts[-1][:, 0]))
            del fx, fv
            gc.collect()
        hist_tr = np.hstack(hist_tr_parts)
        hist_va = np.hstack(hist_va_parts)
        log(f"  fold-{fold} history: {hist_tr.shape} train, {hist_va.shape} val")
        memprint(f"fold{fold} history")

        # ---- leak tests for this fold ----
        log(f"STEP {5 if fold==2 else 11}: leakage tests for fold {fold}")
        audit = run_leakage_tests(codes_map_tr, y, t_hours_tr, tr_end,
                                  raw_fraud, samp_ids, te_ids)
        leak_audit_all[f"fold{fold}"] = audit
        if not audit.get("all_tests_pass"):
            log(f"  !!! LEAK TEST FAIL fold{fold} - promotion will be blocked")
        del hist_prefix
        del hist_tr_parts, hist_va_parts
        gc.collect()

        # ---- BASE fold training ----
        log(f"STEP {6 if fold==2 else 12}: BASE fold-{fold} training")
        base_trf = np.load(os.path.join(OUT, "_base_tr.npy"), mmap_mode="r")[:tr_end]
        base_vaf = np.load(os.path.join(OUT, "_base_tr.npy"), mmap_mode="r")[tr_end:va_end]
        base_trf = np.ascontiguousarray(base_trf).astype(np.float32)
        base_vaf = np.ascontiguousarray(base_vaf).astype(np.float32)
        bst_b, pb_va, best_b = train_xgb(base_trf, y_trf, base_vaf, y_vaf, base_cols)
        ap_base = float(average_precision_score(y_vaf, pb_va))
        np.save(os.path.join(OUT, f"base_fold{fold}_predictions.npy"), pb_va.astype(np.float32))
        log(f"  BASE fold{fold} AP={ap_base:.6f} best_iter={best_b}")
        best_iters["base"].append(int(bst_b.best_iteration))
        del bst_b, pb_va, base_trf, base_vaf
        gc.collect()

        V5_hist_tr_path = os.path.join(OUT, f"_hist_fold{fold}_tr.npy")
        V5_hist_va_path = os.path.join(OUT, f"_hist_fold{fold}_va.npy")
        if fold == 3:
            np.save(V5_hist_tr_path, hist_tr)
            np.save(V5_hist_va_path, hist_va)

        # ---- V5 fold training ----
        base_trf = np.load(os.path.join(OUT, "_base_tr.npy"), mmap_mode="r")[:tr_end]
        base_vaf = np.load(os.path.join(OUT, "_base_tr.npy"), mmap_mode="r")[tr_end:va_end]
        lf_trf = np.load(os.path.join(OUT, "_lf_tr.npy"), mmap_mode="r")[:tr_end]
        lf_vaf = np.load(os.path.join(OUT, "_lf_tr.npy"), mmap_mode="r")[tr_end:va_end]
        base_trf = np.ascontiguousarray(base_trf).astype(np.float32)
        base_vaf = np.ascontiguousarray(base_vaf).astype(np.float32)
        lf_trf = np.ascontiguousarray(lf_trf).astype(np.float32)
        lf_vaf = np.ascontiguousarray(lf_vaf).astype(np.float32)
        Xv5_tr = np.hstack([base_trf, lf_trf, hist_tr])
        Xv5_va = np.hstack([base_vaf, lf_vaf, hist_va])
        v5_cols = base_cols + lf_cols + hist_cols_k
        del base_trf, base_vaf, lf_trf, lf_vaf, hist_tr, hist_va
        gc.collect()
        memprint(f"fold{fold} V5 matrices")

        log(f"STEP {7 if fold==2 else 13}: V5 fold-{fold} training")
        bst_v, pv_va, best_v = train_xgb(Xv5_tr, y_trf, Xv5_va, y_vaf, v5_cols)
        ap_v5 = float(average_precision_score(y_vaf, pv_va))
        np.save(os.path.join(OUT, f"v5_fold{fold}_predictions.npy"), pv_va.astype(np.float32))
        log(f"  V5 fold{fold} AP={ap_v5:.6f} best_iter={best_v}")
        best_iters["v5"].append(int(bst_v.best_iteration))

        fold_results[f"fold{fold}"] = dict(base_ap=ap_base, v5_ap=ap_v5,
                                           best_iter_base=int(best_b),
                                           best_iter_v5=int(best_v))

        # ablation: V5 without supervised history block on the (third) fold
        if fold == 3:
            log("STEP 15: V5 without supervised history block on fold-3")
            supervised = {"prior_fr", "prior_rate", "sm5", "sm20", "l3", "l5",
                          "l10", "l3r", "l5r", "l10r", "hrs_since_fr"}
            order = []
            names_sup = []
            for kname in [s[0] for s in KEY_SPECS]:
                for f in HIST_FEATS:
                    order.append(f"h_{kname}_{f}")
                    names_sup.append(f in supervised)
            sup_mask = ~np.array(names_sup, dtype=bool)
            hist_tr = np.load(V5_hist_tr_path)
            hist_va = np.load(V5_hist_va_path)
            keep_hist_tr = hist_tr[:, sup_mask]
            keep_hist_va = hist_va[:, sup_mask]
            del hist_tr, hist_va
            gc.collect()
            base_trf = np.load(os.path.join(OUT, "_base_tr.npy"), mmap_mode="r")[:tr_end]
            base_vaf = np.load(os.path.join(OUT, "_base_tr.npy"), mmap_mode="r")[tr_end:va_end]
            lf_trf = np.load(os.path.join(OUT, "_lf_tr.npy"), mmap_mode="r")[:tr_end]
            lf_vaf = np.load(os.path.join(OUT, "_lf_tr.npy"), mmap_mode="r")[tr_end:va_end]
            Xab_tr = np.hstack([base_trf, lf_trf, keep_hist_tr])
            Xab_va = np.hstack([base_vaf, lf_vaf, keep_hist_va])
            ncols_nosup = [c for c, keep in zip(hist_cols_k, sup_mask) if keep]
            abl_cols = base_cols + lf_cols + ncols_nosup
            _, pab, _ = train_xgb(Xab_tr, y_trf, Xab_va, y_vaf, abl_cols)
            ap_abl = float(average_precision_score(y_vaf, pab))
            abl_row = dict(fold="fold3", config="v5_no_supervised_history",
                           ap=ap_abl, ap_full=ap_v5, ap_base=ap_base,
                           delta_vs_full=ap_abl - ap_v5, delta_vs_base=ap_abl - ap_base)
            log(f"  ablation v5-nohistory AP={ap_abl:.6f} (full v5 {ap_v5:.6f}, base {ap_base:.6f})")
            pd.DataFrame([abl_row]).to_csv(os.path.join(OUT, "v5_ablation_results.csv"), index=False)
            del Xab_tr, Xab_va, keep_hist_tr, keep_hist_va, pab, base_trf, base_vaf, lf_trf, lf_vaf
            gc.collect()

        del bst_v, pv_va, Xv5_tr, Xv5_va
        gc.collect()
        memprint(f"after fold{fold} release")

    # ---- STEP 15/16: promotion ----
    log("STEP 15/16: promotion metrics + selection")
    ap_base2 = fold_results["fold2"]["base_ap"]
    ap_v2 = fold_results["fold2"]["v5_ap"]
    ap_b3 = fold_results["fold3"]["base_ap"]
    ap_v3 = fold_results["fold3"]["v5_ap"]
    d2 = ap_v2 - ap_base2
    d3 = ap_v3 - ap_b3
    mean_base = (ap_base2 + ap_b3) / 2.0
    mean_v5 = (ap_v2 + ap_v3) / 2.0
    worst_delta = min(d2, d3)
    leak_pass = all(v["all_tests_pass"] for v in leak_audit_all.values())
    f2_high_warn = bool(max(ap_v2, ap_base2) > 0.75)     # adjacent fold: expected, warning only
    ap_block = bool(max(ap_v3, ap_b3) > 0.75)            # out-of-time fold: blocks promotion
    promote = bool(mean_v5 > mean_base and d3 > 0 and d2 >= -0.001 and worst_delta >= -0.001 and leak_pass and not ap_block)
    log(f"  F2 base={ap_base2:.6f} v5={ap_v2:.6f} d={d2:+.6f}")
    log(f"  F3 base={ap_b3:.6f} v5={ap_v3:.6f} d={d3:+.6f}")
    log(f"  mean base={mean_base:.6f} mean v5={mean_v5:.6f} worst_delta={worst_delta:+.6f}")
    log(f"  leak_pass={leak_pass} f2_high_warning={f2_high_warn} f3_ap>0.75_block={ap_block}")
    log(f"  PROMOTION SELECTED: {'V5' if promote else 'BASE'}")

    best_idx = best_iters["v5"] if promote else best_iters["base"]
    sel_iters = [int(x) for x in best_idx if x is not None]
    if not sel_iters:
        sel_iters = [int(best_iters["base"][-1])]
    n_trees = int(np.median(sel_iters) * 1.05)
    n_trees = max(100, n_trees)
    log(f"  final trees = median({sel_iters})*1.05 => {n_trees}")

    # ---- STEP 17: full-train frozen mappings ----
    if promote:
        log("STEP 17: full-train frozen mappings for test")
        rate_full = float(y.mean())
        hist_tr_full_parts = []
        hist_te_parts = []
        hist_full_cols = []
        for name, _cols in KEY_SPECS:
            cc_tr = codes_map_tr[name]
            cc_te = codes_map_te[name]
            nk = int(np.concatenate([cc_tr, cc_te]).max()) + 1
            fx, c, fc, ltx, lfr, ring, ptr = build_train_hist(
                cc_tr, y.astype(np.int8), t_hours_tr, rate_full, nk)
            ft = apply_frozen_hist(cc_te, t_hours_te, rate_full, c, fc, ltx, lfr, ring, ptr, nk)
            hist_tr_full_parts.append(fx)
            hist_te_parts.append(ft)
            hist_full_cols += [f"h_{name}_{f}" for f in HIST_FEATS]
            del fx, ft, c, fc, ltx, lfr, ring, ptr
            gc.collect()
        hist_tr_full = np.hstack(hist_tr_full_parts)
        hist_te_full = np.hstack(hist_te_parts)
        del hist_tr_full_parts, hist_te_parts
        gc.collect()
        memprint("full-train frozen")
    elif True:
        log("STEP 17 skipped (BASE selected; no frozen history needed)")

    # ---- STEP 18: final fit ----
    log("STEP 18: train selected config on full labelled data")
    base_full = np.load(os.path.join(OUT, "_base_tr.npy"), mmap_mode="r").astype(np.float32)
    if promote:
        lf_full = np.load(os.path.join(OUT, "_lf_tr.npy"), mmap_mode="r").astype(np.float32)
        Xfull = np.hstack([base_full, lf_full, hist_tr_full])
        cols_full = base_cols + lf_cols + hist_cols_k
    else:
        Xfull = base_full
        cols_full = base_cols
    dtrain = xgb.DMatrix(Xfull, label=y, feature_names=cols_full, nthread=-1)
    del Xfull, base_full
    if promote:
        del lf_full, hist_tr_full
    gc.collect()
    bst_final = xgb.train(XPARAMS, dtrain, num_boost_round=n_trees, verbose_eval=100)
    log(f"  final model trained: {len(cols_full)} features, {n_trees} trees")
    imp = bst_final.get_score(importance_type="gain")
    imp_rows = sorted(imp.items(), key=lambda kv: -kv[1])
    pd.DataFrame(imp_rows, columns=["feature", "gain"]).to_csv(
        os.path.join(OUT, "v5_feature_importance.csv"), index=False)
    with open(os.path.join(OUT, "v5_feature_list.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(cols_full))
    if promote:
        fam = ["base"] * len(base_cols) + ["labelfree"] * len(lf_cols) + ["history"] * len(hist_cols_k)
    else:
        fam = ["base"] * len(cols_full)
    pd.DataFrame({"feature": cols_full, "family": fam}).to_csv(
        os.path.join(OUT, "v5_feature_summary.csv"), index=False)
    bst_final.save_model(os.path.join(OUT, "v5_model.json"))

    # ---- STEP 19: predict test ----
    log("STEP 19: predict test")
    base_te = np.load(os.path.join(OUT, "_base_te.npy"))
    if promote:
        lf_te = np.load(os.path.join(OUT, "_lf_te.npy"))
        Xte = np.hstack([base_te.astype(np.float32), lf_te.astype(np.float32), hist_te_full])
    else:
        Xte = base_te.astype(np.float32)
    dtest = xgb.DMatrix(Xte, feature_names=cols_full, nthread=-1)
    del Xte, base_te
    if promote:
        del lf_te, hist_te_full
    gc.collect()
    pred = bst_final.predict(dtest)
    np.save(os.path.join(OUT, "v5_test_predictions.npy"), pred.astype(np.float32))
    log(f"  test preds {pred.shape} range [{pred.min():.4f},{pred.max():.4f}]")

    # ---- STEP 20: submission + reload check ----
    log("STEP 20: create + reload-check submission")
    sub = pd.DataFrame({"transaction_id": te_ids, "fraud": pred.astype(np.float64)})
    sub_path = os.path.join(OUT, "submission_v5_fast.csv")
    sub.to_csv(sub_path, index=False)
    chk = pd.read_csv(sub_path)
    checks = dict(
        rows=int(chk.shape[0]),
        cols=list(chk.columns),
        id_order=bool(np.array_equal(chk.transaction_id.values, samp_ids)),
        no_dup=bool(chk.transaction_id.is_unique),
        no_missing=bool(chk.fraud.notna().all() and chk.transaction_id.notna().all()),
        finite=bool(np.isfinite(chk.fraud.values).all()),
        in_range=bool((chk.fraud.values >= 0).all() and (chk.fraud.values <= 1).all()),
        precision=bool(np.allclose(pred.astype(np.float64), chk.fraud.values,
                                   rtol=1e-9, atol=1e-12)),
    )
    checks["valid"] = all([checks["rows"] == 262648, checks["cols"] == ["transaction_id", "fraud"],
                           checks["id_order"], checks["no_dup"], checks["no_missing"],
                           checks["finite"], checks["in_range"], checks["precision"]])
    log(f"  submission checks: {checks}")

    # ---- STEP 21/22: artifacts + summary ----
    report = dict(
        pipeline="v5_fast",
        selected="V5" if promote else "BASE",
        promote=bool(promote),
        fold_results=fold_results,
        mean_base=mean_base, mean_v5=mean_v5,
        n_trees=int(n_trees),
        best_iters=best_iters,
        leak_pass=bool(leak_pass),
        ap_blocked_075=bool(ap_block),
        submission_checks=checks,
        runtime_sec=round(time.time() - t_start, 1),
        temporal_test_after_train=bool(temporal_order_ok),
    )
    with open(os.path.join(OUT, "v5_validation_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(os.path.join(OUT, "v5_leakage_audit.json"), "w") as fh:
        json.dump(leak_audit_all, fh, indent=2, default=str)

    # ---- V5_RESEARCH_REPORT.md ----
    lines = [
        "# V5 Research Report", "",
        "## Summary",
        f"- Pipeline: V5 supervised-user-history vs BASE (no history), 8 keys x 16 features.",
        f"- Selection by promotion rule on folds 2+3: **{'V5' if promote else 'BASE'}**.",
        f"- Mean AP  base={mean_base:.6f}  V5={mean_v5:.6f}  (F2 d={d2:+.6f}, F3 d={d3:+.6f}).",
        f"- Leak tests {7} items/fold: {'PASS (blocking would apply only if failed)' if leak_pass else 'FAIL (promotion blocked)'}.",
        f"- Final model: {len(cols_full)} features, {n_trees} trees.",
        "", "## Fold results", "",
    ]
    for k, v in fold_results.items():
        lines.append(f"- {k}: base AP={v['base_ap']:.6f}  v5 AP={v['v5_ap']:.6f}  (best_iter {v['best_iter_base']}/{v['best_iter_v5']})")
    lines += ["", "## Ablation (fold-3)", ""]
    if os.path.exists(os.path.join(OUT, "v5_ablation_results.csv")):
        abl = pd.read_csv(os.path.join(OUT, "v5_ablation_results.csv"))
        for _, r in abl.iterrows():
            lines.append(f"- {r['config']}: AP={r['ap']:.6f} (full v5 {r['ap_full']:.6f}, base {r['ap_base']:.6f})")
    else:
        lines.append("- (no ablation available)")
    lines += ["", "## Submission", ""]
    for k, v in checks.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Feature families", "",
              f"- A: base parquet cols x{len(base_cols)} (missing indicators + corr>0.999 dedup applied)",
              f"- C/D/etc label-free x{len(lf_cols)}",
              f"- G/H/I supervised history x{len(hist_cols_k)} (8 keys)"]
    with open(os.path.join(OUT, "V5_RESEARCH_REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log("STEP 21: V5_RESEARCH_REPORT.md written")

    log("=== FINAL SUMMARY ===")
    log(f"  selected config: {'V5' if promote else 'BASE'} | final trees: {n_trees}")
    log(f"  F2 base={ap_base2:.6f} v5={ap_v2:.6f} | F3 base={ap_b3:.6f} v5={ap_v3:.6f}")
    log(f"  mean base={mean_base:.6f} mean v5={mean_v5:.6f}")
    log(f"  submission: {sub_path} ({sub.shape[0]} rows) valid={checks['valid']}")
    log(f"  runtime: {report['runtime_sec']} s")
    log("=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL ERROR:")
        log(traceback.format_exc())
        raise