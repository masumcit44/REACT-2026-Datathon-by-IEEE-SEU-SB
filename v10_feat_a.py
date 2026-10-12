"""V10 STEP 2 - repeat-behavior feature block (past-only, vectorized).

History excludes current row (shift(1)/cumcount/prior-window only).
No labels. Chronological train+test stream.
"""
import os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
OUT = os.path.join(ROOT, "v10_out")
SAFE_BASE = ["amount_bdt", "amount_log", "hour", "day_of_week", "day_of_month",
    "account_age_days", "is_weekend", "is_business_hour", "n_amount", "n_amount_log",
    "n_amount_sqrt", "n_account_age_days", "n_account_age_log", "n_is_new_account",
    "n_is_very_new", "n_merchant_category_code", "n_device_type_code", "n_location_code",
    "n_payment_method_code", "n_transaction_type_code"]
def main():
    t0 = time.time()
    cols = ["customer_id", "device_id", "merchant_id", "merchant_category", "device_type",
        "payment_method", "transaction_type", "timestamp", "amount_bdt"]
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=cols)
    te = pd.read_csv(os.path.join(ROOT, "test.csv"), usecols=cols)
    raw = pd.concat([tr, te], ignore_index=True)
    n_tr = len(tr)
    N = len(raw)
    amt = raw["amount_bdt"].to_numpy(float)
    loga = np.log1p(np.maximum(amt, 0)).astype(np.float32)
    hr = pd.to_datetime(raw["timestamp"]).dt.hour.to_numpy()
    ts = (pd.to_datetime(raw["timestamp"]).astype("int64").to_numpy() // 10**9).astype(float)
    cust = pd.factorize(raw["customer_id"])[0].astype(np.int64)
    dev = pd.factorize(raw["device_id"])[0].astype(np.int64)
    merch = pd.factorize(raw["merchant_id"])[0].astype(np.int64)
    cat = pd.factorize(raw["merchant_category"])[0].astype(np.int64)
    pay = pd.factorize(raw["payment_method"])[0].astype(np.int64)
    dtyp = pd.factorize(raw["device_type"])[0].astype(np.int64)
    ttyp = (raw["transaction_type"].to_numpy() == "purchase").astype(np.int8)
    day = ((hr >= 6) & (hr <= 22)).astype(np.int8)
    F = {}
    # ---- customer amount behavior (expanding shift-1 stats, vectorized) ----
    print("[v10-fe] customer amount...", flush=True)
    s_amt = pd.Series(amt)
    s_c = pd.Series(cust)
    g = s_amt.groupby(s_c)
    F["v10_customer_prior_amount_count"] = g.cumcount().to_numpy().astype(np.float32)
    F["v10_customer_prior_amount_mean"] = g.transform(lambda s: s.shift(1).expanding().mean()).fillna(0).to_numpy(np.float32)
    F["v10_customer_prior_amount_std"] = g.transform(lambda s: s.shift(1).expanding().std()).fillna(0).to_numpy(np.float32)
    F["v10_customer_prior_amount_max"] = g.transform(lambda s: s.shift(1).expanding().max()).fillna(0).to_numpy(np.float32)
    F["v10_customer_prior_amount_median_approx"] = g.transform(lambda s: s.shift(1).expanding().median()).fillna(0).to_numpy(np.float32)
    mn = F["v10_customer_prior_amount_mean"]; sd = np.maximum(F["v10_customer_prior_amount_std"], 1e-6)
    F["v10_customer_amount_zscore"] = np.clip((amt - mn) / sd, -10, 10).astype(np.float32)
    F["v10_customer_amount_log_ratio"] = (loga - np.log1p(np.maximum(mn, 0))).astype(np.float32)
    F["v10_customer_amount_vs_previous"] = np.zeros(N, np.float32)
    prev_amt = g.transform(lambda s: s.shift(1)).fillna(0).to_numpy(float)
    F["v10_customer_amount_vs_previous"] = np.clip(amt / np.maximum(prev_amt, 1), 0, 50).astype(np.float32)
    for w, nm in [(5, "recent_5"), (10, "recent_10"), (20, "recent_20")]:
        rm = g.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()).fillna(0).to_numpy(float)
        F[f"v10_customer_amount_vs_{nm}_mean"] = np.clip(amt / np.maximum(rm, 1), 0, 50).astype(np.float32)
    F["v10_customer_amount_percentile_approx"] = np.zeros(N, np.float32)
    # per-customer rank of current vs prior: use expanding rank via groupby apply on codes is slow; approx with zscore CDF
    from scipy.stats import norm as _n
    try:
        F["v10_customer_amount_percentile_approx"] = _n.cdf(F["v10_customer_amount_zscore"]).astype(np.float32)
    except Exception:
        F["v10_customer_amount_percentile_approx"] = np.clip((F["v10_customer_amount_zscore"] + 3) / 6, 0, 1).astype(np.float32)
    # same-type / same-category / same-hour-band means (past-only)
    key_t = pd.Series(list(zip(cust.tolist(), ttyp.tolist())))
    F["v10_customer_amount_vs_same_transaction_type_mean"] = np.clip(amt / np.maximum(pd.Series(amt).groupby(key_t).transform(lambda s: s.shift(1).expanding().mean()).fillna(0).to_numpy(float), 1), 0, 50).astype(np.float32)
    key_c = pd.Series(list(zip(cust.tolist(), cat.tolist())))
    F["v10_customer_amount_vs_same_merchant_category_mean"] = np.clip(amt / np.maximum(pd.Series(amt).groupby(key_c).transform(lambda s: s.shift(1).expanding().mean()).fillna(0).to_numpy(float), 1), 0, 50).astype(np.float32)
    hb = (hr // 6).astype(int)
    key_h = pd.Series(list(zip(cust.tolist(), hb.tolist())))
    F["v10_customer_amount_vs_same_hour_band_mean"] = np.clip(amt / np.maximum(pd.Series(amt).groupby(key_h).transform(lambda s: s.shift(1).expanding().mean()).fillna(0).to_numpy(float), 1), 0, 50).astype(np.float32)
    print(f"[v10-fe] cust amt done {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(OUT, "_f1_train.npy"), np.column_stack([F[k][:n_tr] for k in sorted(F)]).astype(np.float32))
    np.save(os.path.join(OUT, "_f1_test.npy"), np.column_stack([F[k][n_tr:] for k in sorted(F)]).astype(np.float32))
    C.jdump({"names": sorted(F)}, os.path.join(OUT, "_f1.json"))
    print(f"[v10-fe] PART1 k={len(F)}", flush=True)
    print("DONE_F1", flush=True)
if __name__ == "__main__":
    main()
