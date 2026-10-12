"""V10 STEP 2 PART 2 - device amount behavior, purchase behavior,
behavior-change features (all strictly past-only, label-free).

Generates features for:
- Device amount behavior (v10_dev_amount_*)
- Customer purchase behavior (v10_cust_purchase_*)
- Device purchase behavior (v10_dev_purchase_*)
- Behavior-change features (v10_change_*)

Uses chronological split: train rows [0:731942), test rows [731942:994590).
No labels are used. No future data leaks into any feature.
"""
import os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
OUT = os.path.join(ROOT, "v10_out")
os.makedirs(OUT, exist_ok=True)

def main():
    t0 = time.time()
    cols = ["customer_id", "device_id", "merchant_id", "merchant_category",
            "device_type", "payment_method", "transaction_type", "timestamp",
            "amount_bdt"]
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=cols)
    te = pd.read_csv(os.path.join(ROOT, "test.csv"), usecols=cols)
    raw = pd.concat([tr, te], ignore_index=True)
    n_tr = len(tr)
    N = len(raw)

    amt = raw["amount_bdt"].to_numpy(float)
    loga = np.log1p(np.maximum(amt, 0)).astype(np.float32)
    cust = pd.factorize(raw["customer_id"])[0].astype(np.int64)
    dev = pd.factorize(raw["device_id"])[0].astype(np.int64)
    cat = pd.factorize(raw["merchant_category"])[0].astype(np.int64)
    pay = pd.factorize(raw["payment_method"])[0].astype(np.int64)
    ttyp = (raw["transaction_type"].to_numpy() == "purchase").astype(np.int8)
    merch = pd.factorize(raw["merchant_id"])[0].astype(np.int64)

    F = {}

    # ---- DEVICE AMOUNT BEHAVIOR ----
    print("[v10] device amount behavior...", flush=True)
    s_d = pd.Series(dev)
    g_d = pd.Series(amt).groupby(s_d)
    F["v10_dev_prior_amount_count"] = g_d.cumcount().to_numpy().astype(np.float32)
    F["v10_dev_prior_amount_mean"] = g_d.transform(lambda s: s.shift(1).expanding().mean()).fillna(0).to_numpy(np.float32)
    F["v10_dev_prior_amount_std"] = g_d.transform(lambda s: s.shift(1).expanding().std()).fillna(0).to_numpy(np.float32)
    F["v10_dev_prior_amount_max"] = g_d.transform(lambda s: s.shift(1).expanding().max()).fillna(0).to_numpy(np.float32)
    F["v10_dev_prior_amount_median_approx"] = g_d.transform(lambda s: s.shift(1).expanding().median()).fillna(0).to_numpy(np.float32)
    mn = F["v10_dev_prior_amount_mean"]; sd = np.maximum(F["v10_dev_prior_amount_std"], 1e-6)
    F["v10_dev_amount_zscore"] = np.clip((amt - mn) / sd, -10, 10).astype(np.float32)
    F["v10_dev_amount_log_ratio"] = (loga - np.log1p(np.maximum(mn, 0))).astype(np.float32)
    F["v10_dev_amount_vs_previous"] = np.zeros(N, np.float32)
    prev_amt = g_d.transform(lambda s: s.shift(1)).fillna(0).to_numpy(float)
    F["v10_dev_amount_vs_previous"] = np.clip(amt / np.maximum(prev_amt, 1), 0, 50).astype(np.float32)
    for w, nm in [(5, "recent_5"), (10, "recent_10"), (20, "recent_20")]:
        rm = g_d.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()).fillna(0).to_numpy(float)
        F[f"v10_dev_amount_vs_{nm}_mean"] = np.clip(amt / np.maximum(rm, 1), 0, 50).astype(np.float32)
    print(f"[v10] device amt done ({time.time()-t0:.0f}s)", flush=True)

    # ---- CUSTOMER PURCHASE BEHAVIOR ----
    print("[v10] customer purchase behavior...", flush=True)
    s_c = pd.Series(cust)
    is_pur = ttyp.astype(bool)
    pur_amt = np.where(is_pur, amt, np.nan)
    s_pur = pd.Series(pur_amt)
    g_pur_c = s_pur.groupby(s_c)
    pur_count = is_pur.astype(int)
    g_pur_cnt = pd.Series(pur_count).groupby(s_c)
    F["v10_cust_purchase_count"] = g_pur_cnt.cumsum().to_numpy().astype(np.float32)
    F["v10_cust_prior_purchase_mean"] = g_pur_c.transform(lambda s: s.shift(1).expanding().mean()).fillna(0).to_numpy(np.float32)
    F["v10_cust_prior_purchase_count"] = g_pur_c.transform(lambda s: s.shift(1).expanding().count()).fillna(0).to_numpy(np.float32)
    F["v10_cust_purchase_freq"] = np.clip(F["v10_cust_purchase_count"] / np.maximum(pd.Series(cust).groupby(s_c).cumcount().to_numpy().astype(float), 1), 0, 5).astype(np.float32)
    print(f"[v10] cust purchase done ({time.time()-t0:.0f}s)", flush=True)

    # ---- DEVICE PURCHASE BEHAVIOR ----
    print("[v10] device purchase behavior...", flush=True)
    g_pur_d = s_pur.groupby(s_d)
    F["v10_dev_prior_purchase_mean"] = g_pur_d.transform(lambda s: s.shift(1).expanding().mean()).fillna(0).to_numpy(np.float32)
    F["v10_dev_purchase_freq"] = np.clip(pd.Series(pur_count).groupby(s_d).cumsum().to_numpy().astype(float) / np.maximum(pd.Series(dev).groupby(s_d).cumcount().to_numpy().astype(float), 1), 0, 5).astype(np.float32)
    print(f"[v10] dev purchase done ({time.time()-t0:.0f}s)", flush=True)

    # ---- BEHAVIOR-CHANGE FEATURES ----
    print("[v10] behavior-change features...", flush=True)
    cat_changed = (cat != pd.Series(cat).shift(1).fillna(-1).to_numpy()).astype(int)
    pay_changed = (pay != pd.Series(pay).shift(1).fillna(-1)).to_numpy().astype(int)
    s_cd = pd.Series(list(zip(cust.tolist(), cat.tolist())))
    cat_change_cust = (s_cd != s_cd.shift(1).fillna(-1)).astype(int)
    s_cp = pd.Series(list(zip(cust.tolist(), pay.tolist())))
    pay_change_cust = (s_cp != s_cp.shift(1).fillna(-1)).astype(int)
    s_dm = pd.Series(list(zip(dev.tolist(), merch.tolist())))
    dm_change = (s_dm != s_dm.shift(1).fillna(-1)).astype(int)
    F["v10_merchant_category_changed"] = cat_changed.astype(np.float32)
    F["v10_payment_method_changed"] = pay_changed.astype(np.float32)
    F["v10_cust_cat_changed"] = cat_change_cust.astype(np.float32)
    F["v10_cust_pay_changed"] = pay_change_cust.astype(np.float32)
    F["v10_dev_merchant_changed"] = dm_change.astype(np.float32)
    print(f"[v10] behavior change done ({time.time()-t0:.0f}s)", flush=True)

    # Save all Part 2 features
    all_names = sorted(F.keys())
    part2_train = np.column_stack([F[k][:n_tr] for k in all_names]).astype(np.float32)
    part2_test = np.column_stack([F[k][n_tr:] for k in all_names]).astype(np.float32)
    np.save(os.path.join(OUT, "_f2_train.npy"), part2_train)
    np.save(os.path.join(OUT, "_f2_test.npy"), part2_test)
    C.jdump({"names": all_names}, os.path.join(OUT, "_f2.json"))
    print(f"[v10] PART2 done k={len(all_names)}", flush=True)
    print("DONE_PART2", flush=True)

if __name__ == "__main__":
    main()