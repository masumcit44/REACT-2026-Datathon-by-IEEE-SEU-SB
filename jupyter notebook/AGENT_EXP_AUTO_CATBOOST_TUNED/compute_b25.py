import numpy as np, pandas as pd, os, time, pickle, json
# compute the 9 b25 acceleration features (past-only windowed ratios) on FOLD-3 train split and save
BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_CATBOOST_TUNED_110")  # documented? no; E3 output
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_CATBOOST_TUNED")
os.makedirs(OUT, exist_ok=True)
tr = pd.read_parquet(os.path.join(BASE, "feature_train_v4.parquet"),
                     columns=["timestamp", "customer_id", "device_id", "merchant_id", "fraud"])
y = tr["fraud"].values.astype(int)
ts_i = tr["timestamp"].values.astype("datetime64[ns]").astype("int64")
n = len(tr)
H1, H6, H24 = 3600_000_000_000, 21600_000_000_000, 86400_000_000_000

def windowed(df, key):
    res = {}
    g = df.groupby(key).indices
    c1 = np.zeros(n); c6 = np.zeros(n); c24 = np.zeros(n)
    for k, idx in g.items():
        idx = np.asarray(idx); t = ts_i[idx]; pos = np.arange(len(idx))
        c1[idx] = pos - np.searchsorted(t, t - H1, "left")
        c6[idx] = pos - np.searchsorted(t, t - H6, "left")
        c24[idx] = pos - np.searchsorted(t, t - H24, "left")
    return c1, c6, c24

for ent, key in [("customer", "customer_id"), ("device", "device_id"), ("merchant", "merchant_id")]:
    c1, c6, c24 = windowed(tr, key)
    tr[f"b25_{ent}_1h_to_24h_ratio"] = np.where(c24 > 0, c1 / c24, np.nan)
    tr[f"b25_{ent}_6h_to_24h_ratio"] = np.where(c24 > 0, c6 / c24, np.nan)
    tr[f"b25_{ent}_1h_to_6h_ratio"] = np.where(c6 > 0, c1 / c6, np.nan)
    del c1, c6, c24

b25_cols = [f"b25_{e}_{a}_to_{b}_ratio" for e in ["customer", "device", "merchant"] for a, b in [("1h", "24h"), ("6h", "24h"), ("1h", "6h")]]
b25 = tr[b25_cols].astype(np.float32)
np.save(os.path.join(OUT, "b25_fold3_train.npy"), b25.values)
json.dump({"cols": b25_cols, "nan": int(b25.isna().sum().sum())}, open(os.path.join(OUT, "b25_meta.json"), "w"))
print("b25 computed", b25.shape, "nan", int(b25.isna().sum().sum()))