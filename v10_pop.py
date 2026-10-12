"""V10 STEP 1 - research population: repeat/daytime/purchase flags (label-free)."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
OUT = os.path.join(ROOT, "v10_out")
def main():
    _, _, y, _, y3 = C.load_core(verbose=False)
    v7 = np.load(os.path.join(ROOT, "v8_out", "v8_reproduced_fold3.npy")).astype(float).ravel()
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"),
        usecols=["customer_id", "device_id", "merchant_id", "transaction_type", "timestamp"])
    o = np.argsort(-v7, kind="mergesort")
    rk = np.empty(len(v7)); rk[o] = (np.arange(len(v7)) + 1) / len(v7)
    ts = pd.to_datetime(tr["timestamp"])
    hr = ts.dt.hour.to_numpy()
    cc = pd.Series(tr["customer_id"]).groupby(pd.Series(tr["customer_id"])).cumcount().to_numpy()
    dc = pd.Series(tr["device_id"]).groupby(pd.Series(tr["device_id"])).cumcount().to_numpy()
    cd = pd.Series(list(zip(tr["customer_id"], tr["device_id"]))).groupby(pd.Series(list(zip(tr["customer_id"], tr["device_id"])))).cumcount().to_numpy()
    cm = pd.Series(list(zip(tr["customer_id"], tr["merchant_id"]))).groupby(pd.Series(list(zip(tr["customer_id"], tr["merchant_id"])))).cumcount().to_numpy()
    f3 = pd.DataFrame({"y": y3, "v7": v7, "rank": rk})
    f3["rep_c"] = (cc[C.IDX_F3] > 0).astype(int)
    f3["rep_d"] = (dc[C.IDX_F3] > 0).astype(int)
    f3["rep_cd"] = (cd[C.IDX_F3] > 0).astype(int)
    f3["rep_cm"] = (cm[C.IDX_F3] > 0).astype(int)
    f3["day"] = ((hr[C.IDX_F3] >= 6) & (hr[C.IDX_F3] <= 22)).astype(int)
    f3["pur"] = (tr["transaction_type"].to_numpy()[C.IDX_F3] == "purchase").astype(int)
    f3["rep_day_pur"] = ((f3["rep_c"] == 1) | (f3["rep_d"] == 1)) & (f3["day"] == 1) & (f3["pur"] == 1)
    f3["rep_c_day_pur"] = (f3["rep_c"] == 1) & (f3["day"] == 1) & (f3["pur"] == 1)
    f3["rep_d_day_pur"] = (f3["rep_d"] == 1) & (f3["day"] == 1) & (f3["pur"] == 1)
    fn_total = int(((f3["y"] == 1) & (f3["rank"] >= 0.01)).sum())
    rows = []
    groups = {"repeat_customer": "rep_c", "repeat_device": "rep_d", "repeat_cd": "rep_cd",
        "repeat_cm": "rep_cm", "daytime": "day", "purchase": "pur",
        "repeat_daytime_purchase": "rep_day_pur", "repeat_customer_daytime_purchase": "rep_c_day_pur",
        "repeat_device_daytime_purchase": "rep_d_day_pur"}
    for nm, col in groups.items():
        m = f3[col] == 1
        d = f3[m]
        fn = int(((d["y"] == 1) & (d["rank"] >= 0.01)).sum())
        cap = {}
        for q in [0.01, 0.02, 0.05, 0.10]:
            thr = np.quantile(v7, 1 - q)
            cap[f"cap{int(q*100)}"] = int(((f3["y"] == 1) & (f3["v7"] >= thr) & m).sum())
        rows.append({"group": nm, "n": int(m.sum()), "fraud": int(d["y"].sum()),
            "prev": round(float(d["y"].mean()), 5), "v7_ap": round(float(C.ap(d["y"], d["v7"])), 5),
            "v7_med_rank": round(float(d["rank"].median()), 4), "fn": fn,
            "fn_share": round(fn / max(fn_total, 1), 4), **cap})
    sdf = pd.DataFrame(rows).sort_values("fn", ascending=False)
    sdf.to_csv(os.path.join(OUT, "v10_population_analysis.csv"), index=False)
    print(sdf.to_string(), flush=True)
    print(f"[v10] total FN outside top1% = {fn_total}", flush=True)
    np.save(os.path.join(OUT, "_tmp_rank.npy"), rk)
    np.save(os.path.join(OUT, "_tmp_v7.npy"), v7)
    print("DONE_POP", flush=True)
if __name__ == "__main__":
    main()
