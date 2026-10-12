import numpy as np, pandas as pd, os, time, json

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_FINAL7")
os.makedirs(OUT, exist_ok=True)
COLS = ["customer_id", "merchant_id", "device_id", "device_type", "location",
        "amount_bdt", "timestamp"]
DAY_NS = 86400_000_000_000

def build(src):
    t0 = time.time()
    df = pd.read_parquet(src, columns=COLS).sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].values.astype("datetime64[ns]")
    ts_i = ts.astype("int64")
    n = len(df)
    out = {}

    def axis_count(axis_col):
        g = pd.DataFrame({"cust": df["customer_id"].values, "x": df[axis_col].values})
        first = (~g.duplicated(keep="first").values).astype(np.int64)
        cum = pd.Series(first).groupby(g["cust"].values).cumsum().values
        return cum - first

    out["dev_dtype_cust"] = axis_count("device_type")
    out["dev_loc_cust"] = axis_count("location")
    out["num_dev_axes_cust"] = axis_count("device_id")

    df["_dev_first"] = df.groupby("device_id")["timestamp"].transform("min")
    df["_cust_first"] = df.groupby("customer_id")["timestamp"].transform("min")
    days_dev = ((ts - df["_dev_first"].values.astype("datetime64[ns]")).astype("timedelta64[s]").astype(np.float64)) / 86400.0
    days_cust = ((ts - df["_cust_first"].values.astype("datetime64[ns]")).astype("timedelta64[s]").astype(np.float64)) / 86400.0
    out["dev_new_in_last_30d"] = (days_dev <= 30.0).astype(np.float64)
    out["is_new_in_last_30d"] = (days_cust <= 30.0).astype(np.float64)

    # exact past-only amount percentile per customer via g x g matrix per group
    amt = df["amount_bdt"].values.astype(np.float64)
    res = np.full(n, np.nan)
    gi = df.groupby("customer_id").indices
    t1 = time.time(); done = 0
    for c, idx in gi.items():
        idx = np.asarray(idx); g = len(idx)
        a = amt[idx]
        ge = (a[:, None] <= a[None, :])
        # keep only previous rows k < r (time order) -> upper triangle above first diagonal
        lower = np.triu(np.ones((g, g), dtype=bool), k=1)
        prev_le = (ge & lower).sum(axis=0)
        cnt = np.arange(g, dtype=np.float64)
        res[idx] = np.divide(prev_le, cnt, out=np.zeros(g), where=cnt > 0)
        done += 1
        if done % 5000 == 0:
            print(f"  amt_pct {done}/{len(gi)} groups t={time.time()-t1:.0f}s", flush=True)
    out["amt_pct_cust"] = res
    print(f"  amt_pct done in {time.time()-t1:.1f}s", flush=True)

    # merchant growth: windowed counts per merchant (O(g log g))
    res_m = np.zeros(n)
    gi2 = df.groupby("merchant_id").indices
    for m, idx in gi2.items():
        idx = np.asarray(idx); t = ts_i[idx]; pos = np.arange(len(idx))
        c7 = pos - np.searchsorted(t, t - 7 * DAY_NS, "left")
        c28 = pos - np.searchsorted(t, t - 28 * DAY_NS, "left")
        daily = c28 / 28.0
        res_m[idx] = np.where(c28 > 0, np.clip(c7 / np.maximum(daily, 1e-9), 0, 100.0), 0.0)
    out["mch_growth_7d"] = res_m
    print(f"  merchant growth done in {time.time()-t0:.1f}s total", flush=True)

    f = pd.DataFrame(out).round(6)
    print(f"built {f.shape[1]} features on {n} rows in {time.time()-t0:.0f}s", flush=True)
    print(json.dumps({c: {"nan": int(f[c].isna().sum()), "min": float(f[c].min()), "max": float(f[c].max())} for c in f}), flush=True)
    return f

tr = build(os.path.join(BASE, "feature_train_v4.parquet"))
te = build(os.path.join(BASE, "feature_test_v4.parquet"))
tr.to_parquet(os.path.join(OUT, "f7_train.parquet"))
te.to_parquet(os.path.join(OUT, "f7_test.parquet"))
print("F7 BUILD OK - parquets saved")