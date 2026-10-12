"""V9 STEP 1b - error cluster analysis over honest fold-3 segments."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
OUT = os.path.join(ROOT, "v9_out")
def main():
    _, _, y, _, y3 = C.load_core(verbose=False)
    ranks = np.load(os.path.join(OUT, "_rank_f3.npy"))
    v7f3 = np.load(os.path.join(OUT, "_v7_f3.npy"))
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"),
        usecols=["customer_id", "device_id", "merchant_category", "payment_method", "transaction_type", "timestamp", "amount_bdt"])
    f3 = tr.iloc[C.IDX_F3].reset_index(drop=True).copy()
    f3["y"] = y3; f3["v7"] = v7f3; f3["rank"] = ranks
    f3["ts"] = pd.to_datetime(f3["timestamp"])
    f3["hour"] = f3["ts"].dt.hour; f3["dow"] = f3["ts"].dt.dayofweek; f3["day"] = f3["ts"].dt.day
    q = np.quantile(f3["amount_bdt"].to_numpy(), [0.33, 0.66])
    f3["amt_b"] = pd.cut(f3["amount_bdt"], [-1, q[0], q[1], 1e18], labels=["low", "mid", "high"])
    f3["h_b"] = f3["hour"].map(lambda h: "h00" if h == 0 else ("h01_04" if 1 <= h <= 4 else ("h05" if h == 5 else "rest")))
    f3["wknd_night"] = ((f3["dow"] >= 5) & (f3["hour"].between(1, 4))).astype(int)
    f3["july_b"] = pd.cut(f3["day"], [0, 5, 10, 16], labels=["jul1_5", "jul6_10", "jul11_15"])
    cumc = pd.Series(tr["customer_id"]).groupby(pd.Series(tr["customer_id"])).cumcount().to_numpy()
    cumd = pd.Series(tr["device_id"]).groupby(pd.Series(tr["device_id"])).cumcount().to_numpy()
    f3["cust_prior"] = cumc[C.IDX_F3]; f3["dev_prior"] = cumd[C.IDX_F3]
    G = np.load(os.path.join(ROOT, "v7_out", "graph_v3_train.npy"))[C.IDX_F3]
    import json as J
    gi = {n: j for j, n in enumerate(J.load(open(os.path.join(ROOT, "v7_out", "graph_v3_meta.json")))["names"])}
    f3["cd_pairs"] = G[:, gi["g_cd_pairs"]]
    f3["new_edge"] = (f3["cd_pairs"] == 0).astype(int)
    rows = []
    def add(name, mask):
        d = f3[mask]
        if len(d) < 200:
            return
        r10 = d[d["rank"].between(0.01, 0.10)]
        rows.append({"segment": name, "n": len(d), "fraud": int(d["y"].sum()),
            "rate": round(float(d["y"].mean()), 5), "v7_ap": round(float(C.ap(d["y"], d["v7"])), 5),
            "v7_mean": round(float(d["v7"].mean()), 5),
            "fn_out1": int(((d["y"] == 1) & (d["rank"] >= 0.01)).sum()),
            "fraud_1_10": int((r10["y"] == 1).sum())})
    for b in sorted(f3["h_b"].unique()): add(f"hour:{b}", f3["h_b"] == b)
    add("weekend_night:1", f3["wknd_night"] == 1)
    for b in ["low", "mid", "high"]: add(f"amt:{b}", f3["amt_b"] == b)
    for b in ["jul1_5", "jul6_10", "jul11_15"]: add(f"july:{b}", f3["july_b"] == b)
    add("cust_prior:0-2", f3["cust_prior"] <= 2); add("cust_prior:3+", f3["cust_prior"] > 2)
    add("dev_prior:0-5", f3["dev_prior"] <= 5); add("dev_prior:6+", f3["dev_prior"] > 5)
    add("new_edge:1", f3["new_edge"] == 1); add("new_edge:0", f3["new_edge"] == 0)
    for c in ["transaction_type", "merchant_category", "payment_method"]:
        for v in f3[c].dropna().unique(): add(f"{c}:{v}", f3[c] == v)
    sdf = pd.DataFrame(rows).sort_values("fn_out1", ascending=False)
    sdf.to_csv(os.path.join(OUT, "v9_error_clusters.csv"), index=False)
    print(sdf.head(22).to_string(), flush=True)
    print("DONE_B", flush=True)
if __name__ == "__main__":
    main()
