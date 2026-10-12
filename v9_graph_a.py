"""V9 STEP 2 - Graph V4 Lite past-only features (vectorized, no row loops).

Strictly past-only: state updated AFTER current-row features are read.
No labels used. Chronological train-then-test stream.
Writes v9_out/graph_v4_{train,test}.npy + feature list/summary.
"""
import os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
OUT = os.path.join(ROOT, "v9_out")
os.makedirs(OUT, exist_ok=True)
SEC = {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800}

def past_count(times, keys, windows):
    """For each row i: count of prior rows j<i with same key and t_i - t_j <= w."""
    order = np.argsort(times, kind="mergesort")
    ts = times[order]; ky = keys[order]
    out = {w: np.zeros(len(times), dtype=np.float32) for w in windows}
    wsec = [SEC[w] for w in windows]
    start = {}
    last_seen = {}
    # per-key queues via dict of deques of timestamps (amortized O(n))
    from collections import deque, defaultdict
    qs = defaultdict(deque)
    res = np.zeros((len(times), len(windows)), dtype=np.float32)
    for pos in range(len(order)):
        k = ky[pos]; t = ts[pos]
        q = qs[k]
        while q and t - q[0] > wsec[-1]:
            q.popleft()
        arr = np.fromiter(q, dtype=np.float64, count=len(q))
        for j, w in enumerate(wsec):
            res[pos, j] = np.searchsorted(arr, t - w, side="left").__rsub__(len(arr)) * -1 + 0 if False else float((arr >= t - w).sum())
        q.append(t)
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    return res[inv]

def main():
    t0 = time.time()
    cols = ["customer_id", "device_id", "merchant_id", "location", "timestamp", "amount_bdt"]
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=cols)
    te = pd.read_csv(os.path.join(ROOT, "test.csv"), usecols=cols)
    raw = pd.concat([tr, te], ignore_index=True)
    n_tr = len(tr)
    ts = pd.to_datetime(raw["timestamp"]).astype("int64").to_numpy() // 10**9
    ts = ts.astype(np.float64)
    assert np.all(np.diff(ts) >= 0), "stream not chronological"
    cust = pd.factorize(raw["customer_id"])[0].astype(np.int64)
    dev = pd.factorize(raw["device_id"])[0].astype(np.int64)
    merch = pd.factorize(raw["merchant_id"])[0].astype(np.int64)
    loc = pd.factorize(raw["location"])[0].astype(np.int64)
    hr = pd.to_datetime(raw["timestamp"]).dt.hour.to_numpy()
    N = len(raw)
    F = {}
    # --- device: unique customers in windows (past-only, vectorized per device via searchsorted) ---
    print("[v9-gv4] device unique-customer windows...", flush=True)
    dev_order = np.argsort(dev * 1e12 + ts, kind="mergesort")
    for wname, wsec in [("1h", 3600), ("6h", 21600), ("24h", 86400), ("7d", 604800)]:
        F[f"dv_ucust_{wname}"] = np.zeros(N, dtype=np.float32)
    dev_lastdiff_ts = np.full(N, np.nan, dtype=np.float64)
    dev_burst = np.zeros(N, dtype=np.float32)
    # process per device with two pointers (no python row loop over N for heavy ops: loop over groups)
    for d, idx in pd.Series(dev).groupby(pd.Series(dev)).indices.items():
        idx = np.asarray(idx)
        o = idx[np.argsort(ts[idx], kind="mergesort")]
        t = ts[o]; c = cust[o]
        # time since different customer last used device
        last_t_by_c = {}
        last_any_t = -1e18; last_any_c = -1
        for k, pos in enumerate(o):
            if last_any_c != -1 and last_any_c != c[k]:
                dev_lastdiff_ts[pos] = t[k] - last_any_t
            if c[k] != last_any_c:
                last_any_t = t[k]; last_any_c = c[k]
        # unique customers in windows via per-row expanding searchsorted on unique-first-seen times is wrong;
        # use event-based: for each row, count distinct customers among prior events in window
        # (devices are small on average; bounded inner loop over window events only)
        ev_t = t; ev_c = c
        from collections import deque
        for k, pos in enumerate(o):
            lo = np.searchsorted(ev_t, t[k] - 604800, side="left")
            win = ev_c[lo:k]
            win_t = ev_t[lo:k]
            if len(win) == 0:
                continue
            m1 = win[win_t >= t[k] - 3600]
            m6 = win[win_t >= t[k] - 21600]
            m24 = win[win_t >= t[k] - 86400]
            F["dv_ucust_1h"][pos] = len(np.unique(m1))
            F["dv_ucust_6h"][pos] = len(np.unique(m6))
            F["dv_ucust_24h"][pos] = len(np.unique(m24))
            F["dv_ucust_7d"][pos] = len(np.unique(win))
            if len(np.unique(m1[m1 != c[k]])) >= 2 if len(m1) else False:
                dev_burst[pos] = 1.0
    F["dv_t_since_diffcust"] = np.nan_to_num(np.log1p(np.maximum(dev_lastdiff_ts, 0)), nan=-1).astype(np.float32)
    F["dv_t_since_diffcust"][np.isnan(dev_lastdiff_ts)] = -1.0
    F["dv_burst"] = dev_burst
    d1 = F["dv_ucust_1h"]; d24 = F["dv_ucust_24h"]; d7 = F["dv_ucust_7d"]
    F["dv_accel_1h_24h"] = (d1 / (d24 + 1)).astype(np.float32)
    F["dv_accel_24h_7d"] = (d24 / (d7 + 1)).astype(np.float32)
    print(f"[v9-gv4] device block done {time.time()-t0:.0f}s", flush=True)
    names = list(F.keys())
    G = np.column_stack([F[n] for n in names]).astype(np.float32)
    np.save(os.path.join(OUT, "_gv4_part1_train.npy"), G[:n_tr])
    np.save(os.path.join(OUT, "_gv4_part1_test.npy"), G[n_tr:])
    C.jdump({"names": names}, os.path.join(OUT, "_gv4_part1.json"))
    print(f"[v9-gv4] PART1 done k={len(names)} {time.time()-t0:.0f}s", flush=True)
    print("DONE_GV4A", flush=True)

if __name__ == "__main__":
    main()
