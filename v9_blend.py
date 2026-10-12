"""V9 STEP 4 - fixed residual blends + gates + preservation (no tuning)."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
OUT = os.path.join(ROOT, "v9_out")
def rank01(v):
    s = pd.Series(np.asarray(v)); return (s.rank(method="average").to_numpy() / max(len(v) - 1, 1))
def ap(y, p): return C.ap(np.asarray(y), np.asarray(p))
def band_ap(y, v7, cand, lo, hi):
    o = np.argsort(-v7, kind="mergesort"); r = np.empty(len(v7)); r[o] = (np.arange(len(v7)) + 1) / len(v7)
    m = (r >= lo) & (r < hi)
    return float(C.ap(y[m], cand[m])), int(y[m].sum())
def main():
    _, _, y, _, y3 = C.load_core(verbose=False)
    v7 = np.load(os.path.join(OUT, "_v7_f3.npy"))
    ranks = np.load(os.path.join(OUT, "_rank_f3.npy"))
    res = {t: np.load(os.path.join(OUT, f"v9_fold3_{t}.npy")) for t in ["R1_gv4", "R2_night", "R3_full"]}
    day = pd.to_datetime(pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["timestamp"])["timestamp"]).iloc[C.IDX_F3].dt.day.to_numpy()
    jb = {"early": day <= 5, "mid": (day >= 6) & (day <= 10), "late": day >= 11}
    base = {b: ap(y3[m], v7[m]) for b, m in jb.items()}
    base_worst = min(base.values())
    base_f3 = ap(y3, v7)
    b15, _ = band_ap(y3, v7, v7, 0.01, 0.05); b110, _ = band_ap(y3, v7, v7, 0.01, 0.10)
    print(f"[v9-bl] V7 f3={base_f3:.6f} july={base} band1-5={b15:.5f} band1-10={b110:.5f}", flush=True)
    brows, jrows, prows = [], [], []
    for tag, p in res.items():
        r = rank01(p); rv = rank01(v7)
        for w in [0.025, 0.05, 0.10, 0.15]:
            for mode in ["prob", "rank"]:
                c = (1 - w) * v7 + w * p if mode == "prob" else (1 - w) * rv + w * r
                f3 = ap(y3, c)
                jd = {b: ap(y3[m], c[m]) - base[b] for b, m in jb.items()}
                worst = min(ap(y3[m], c[m]) for b, m in jb.items())
                c15, _ = band_ap(y3, v7, c, 0.01, 0.05); c110, _ = band_ap(y3, v7, c, 0.01, 0.10)
                corr = float(pd.Series(c).corr(pd.Series(v7)))
                gate = (f3 - base_f3 >= 0.0005 and all(v > -0.001 for v in jd.values())
                        and (worst - base_worst) >= -0.0005 and (c15 > b15 or c110 > b110) and corr < 0.995)
                brows.append({"cfg": tag, "mode": mode, "w": w, "f3": round(f3, 6), "gain": round(f3 - base_f3, 6),
                    "d_early": round(jd["early"], 6), "d_mid": round(jd["mid"], 6), "d_late": round(jd["late"], 6),
                    "worst": round(worst, 6), "d_worst": round(worst - base_worst, 6),
                    "band1_5": round(c15, 5), "band1_10": round(c110, 5),
                    "corr": round(corr, 5), "pass": bool(gate)})
                print(f" {tag} {mode} w={w}: f3={f3:.6f} gain={f3-base_f3:+.6f} d=({jd['early']:+.5f},{jd['mid']:+.5f},{jd['late']:+.5f}) corr={corr:.4f} pass={gate}", flush=True)
        jrows.append({"cfg": tag, "base": base_f3, **{f"july_{b}": round(ap(y3[m], res[tag][m]), 6) for b, m in jb.items()}})
    pd.DataFrame(brows).to_csv(os.path.join(OUT, "v9_blend_scores.csv"), index=False)
    pd.DataFrame(jrows).to_csv(os.path.join(OUT, "v9_july_stability.csv"), index=False)
    # preservation on best passing else best gain
    pdf = pd.DataFrame(brows)
    ok = pdf[pdf["pass"]]
    sel = ok.sort_values("gain", ascending=False).iloc[0] if len(ok) else pdf.sort_values("gain", ascending=False).iloc[0]
    print(f"[v9-bl] selected: {sel['cfg']} {sel['mode']} w={sel['w']} pass={sel['pass']}", flush=True)
    p = res[sel["cfg"]]
    c = (1 - sel["w"]) * v7 + sel["w"] * p if sel["mode"] == "prob" else (1 - sel["w"]) * rank01(v7) + sel["w"] * rank01(p)
    for q in [0.005, 0.01, 0.02]:
        tv = np.quantile(v7, 1 - q); cv = np.quantile(c, 1 - q)
        tm = v7 >= tv; cm = c >= cv
        tp_t = int(((y3 == 1) & tm).sum()); tp_c = int(((y3 == 1) & cm).sum())
        prows.append({"q": q, "overlap": round(float((tm & cm).sum() / tm.sum()), 4),
            "v7_capture": tp_t, "cand_capture": tp_c, "removed": int(((y3 == 1) & tm & (~cm)).sum()), "added": int(((y3 == 1) & cm & (~tm)).sum())})
    pd.DataFrame(prows).to_csv(os.path.join(OUT, "v9_top_risk_preservation.csv"), index=False)
    print(pd.DataFrame(prows).to_string(), flush=True)
    # ranking bands for selected
    rb = []
    for lo, hi, nm in [(0, 0.01, "0-1"), (0.01, 0.05, "1-5"), (0.01, 0.10, "1-10"), (0.05, 0.10, "5-10")]:
        a0, n0 = band_ap(y3, v7, v7, lo, hi); a1, _ = band_ap(y3, v7, c, lo, hi)
        rb.append({"band": nm, "n_fraud": n0, "v7_ap": round(a0, 5), "cand_ap": round(a1, 5), "delta": round(a1 - a0, 5)})
    pd.DataFrame(rb).to_csv(os.path.join(OUT, "v9_ranking_band_scores.csv"), index=False)
    print(pd.DataFrame(rb).to_string(), flush=True)
    C.jdump({"sel_cfg": str(sel["cfg"]), "sel_mode": str(sel["mode"]), "sel_w": float(sel["w"]), "sel_pass": bool(sel["pass"]), "sel_gain": float(sel["gain"])}, os.path.join(OUT, "v9_selected.json"))
    print("DONE_BLEND", flush=True)
if __name__ == "__main__":
    main()
