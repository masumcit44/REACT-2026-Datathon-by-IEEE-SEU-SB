"""V9 STEP 2b - Graph V4 Lite part B: cd/dm edges, customer-location (past-only)."""
import os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
OUT = os.path.join(ROOT, "v9_out")
def main():
    t0 = time.time()
    cols = ["customer_id", "device_id", "merchant_id", "location", "timestamp"]
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=cols)
    te = pd.read_csv(os.path.join(ROOT, "test.csv"), usecols=cols)
    raw = pd.concat([tr, te], ignore_index=True)
    n_tr = len(tr)
    ts = (pd.to_datetime(raw["timestamp"]).astype("int64").to_numpy() // 10**9).astype(np.float64)
    cust = pd.factorize(raw["customer_id"])[0].astype(np.int64)
    dev = pd.factorize(raw["device_id"])[0].astype(np.int64)
    merch = pd.factorize(raw["merchant_id"])[0].astype(np.int64)
    loc = pd.factorize(raw["location"])[0].astype(np.int64)
    N = len(raw)
    F = {}
    F["cd_t_since"] = np.full(N, -1, dtype=np.float32)
    F["cd_cnt_24h"] = np.zeros(N, dtype=np.float32)
    F["cd_cnt_7d"] = np.zeros(N, dtype=np.float32)
    F["cd_decay"] = np.zeros(N, dtype=np.float32)
    F["cd_new"] = np.ones(N, dtype=np.float32)
    F["dm_t_since"] = np.full(N, -1, dtype=np.float32)
    F["dm_cnt_24h"] = np.zeros(N, dtype=np.float32)
    F["dm_cnt_7d"] = np.zeros(N, dtype=np.float32)
    F["dm_decay"] = np.zeros(N, dtype=np.float32)
    F["dm_new"] = np.ones(N, dtype=np.float32)
    # CD edges grouped by (cust, dev)
    print("[v9-gv4b] cd edges...", flush=True)
    keys = pd.Series(list(zip(cust.tolist(), dev.tolist())))
    for key, idx in keys.groupby(keys).indices.items():
        idx = np.asarray(idx)
        o = idx[np.argsort(ts[idx], kind="mergesort")]
        t = ts[o]
        F["cd_new"][o[0]] = 1.0
        for k in range(1, len(o)):
            F["cd_new"][o[k]] = 0.0
            F["cd_t_since"][o[k]] = np.log1p(t[k] - t[k - 1]).astype(np.float32)
        for k, pos in enumerate(o):
            lo24 = np.searchsorted(t, t[k] - 86400, side="left")
            lo7 = np.searchsorted(t, t[k] - 604800, side="left")
            F["cd_cnt_24h"][pos] = k - lo24
            F["cd_cnt_7d"][pos] = k - lo7
            if k > 0:
                dt = t[k] - t[max(lo7, 0):k]
                F["cd_decay"][pos] = float(np.exp(-dt / 86400.0).sum())
    print(f"[v9-gv4b] cd done {time.time()-t0:.0f}s", flush=True)
    print("[v9-gv4b] dm edges...", flush=True)
    keys2 = pd.Series(list(zip(dev.tolist(), merch.tolist())))
    for key, idx in keys2.groupby(keys2).indices.items():
        idx = np.asarray(idx)
        o = idx[np.argsort(ts[idx], kind="mergesort")]
        t = ts[o]
        for k in range(1, len(o)):
            F["dm_new"][o[k]] = 0.0
            F["dm_t_since"][o[k]] = np.log1p(t[k] - t[k - 1]).astype(np.float32)
        for k, pos in enumerate(o):
            lo24 = np.searchsorted(t, t[k] - 86400, side="left")
            lo7 = np.searchsorted(t, t[k] - 604800, side="left")
            F["dm_cnt_24h"][pos] = k - lo24
            F["dm_cnt_7d"][pos] = k - lo7
            if k > 0:
                dt = t[k] - t[max(lo7, 0):k]
                F["dm_decay"][pos] = float(np.exp(-dt / 86400.0).sum())
    print(f"[v9-gv4b] dm done {time.time()-t0:.0f}s", flush=True)
    # customer-location: time since same loc, unique locs 24h, switches 6h
    print("[v9-gv4b] customer-location...", flush=True)
    F["cl_t_since"] = np.full(N, -1, dtype=np.float32)
    F["cl_uniq_24h"] = np.zeros(N, dtype=np.float32)
    F["cl_sw_6h"] = np.zeros(N, dtype=np.float32)
    for c, idx in pd.Series(cust).groupby(pd.Series(cust)).indices.items():
        idx = np.asarray(idx)
        o = idx[np.argsort(ts[idx], kind="mergesort")]
        t = ts[o]; l = loc[o]
        last = {}
        for k, pos in enumerate(o):
            if l[k] in last:
                F["cl_t_since"][pos] = np.log1p(t[k] - last[l[k]]).astype(np.float32)
            last[l[k]] = t[k]
            lo24 = np.searchsorted(t, t[k] - 86400, side="left")
            lo6 = np.searchsorted(t, t[k] - 21600, side="left")
            F["cl_uniq_24h"][pos] = len(np.unique(l[lo24:k])) if k > lo24 else 0
            seg = l[lo6:k + 1]
            F["cl_sw_6h"][pos] = int((np.diff(seg) != 0).sum()) if len(seg) > 1 else 0
    # shares (past-only cumulative)
    print("[v9-gv4b] shares...", flush=True)
    F["cd_dev_share"] = np.zeros(N, dtype=np.float32)
    F["cd_cust_share"] = np.zeros(N, dtype=np.float32)
    cum_cd = {}
    cum_c = {}
    cum_d = {}
    # vectorized cumulative counts via groupby cumcount on pair keys
    s_cd = pd.Series(list(zip(cust.tolist(), dev.tolist())))
    s_c = pd.Series(cust); s_d = pd.Series(dev)
    cumcd = s_cd.groupby(s_cd).cumcount().to_numpy() + 1
    cumc = s_c.groupby(s_c).cumcount().to_numpy() + 1
    cumd = s_d.groupby(s_d).cumcount().to_numpy() + 1
    F["cd_dev_share"] = ((cumcd - 1) / np.maximum(cumd - 1, 1)).astype(np.float32)
    F["cd_cust_share"] = ((cumcd - 1) / np.maximum(cumc - 1, 1)).astype(np.float32)
    F["cl_rapid"] = (F["cl_sw_6h"] >= 3).astype(np.float32)
    names = list(F.keys())
    G = np.column_stack([F[n] for n in names]).astype(np.float32)
    assert np.isfinite(G).all()
    np.save(os.path.join(OUT, "_gv4_part2_train.npy"), G[:n_tr])
    np.save(os.path.join(OUT, "_gv4_part2_test.npy"), G[n_tr:])
    C.jdump({"names": names}, os.path.join(OUT, "_gv4_part2.json"))
    print(f"[v9-gv4b] PART2 done k={len(names)} {time.time()-t0:.0f}s", flush=True)
    print("DONE_GV4B", flush=True)
if __name__ == "__main__":
    main()
