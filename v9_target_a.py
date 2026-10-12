"""V9 STEP 1a - residual target part A (bands + disagreement)."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
OUT = os.path.join(ROOT, "v9_out")
os.makedirs(OUT, exist_ok=True)
def band_of(r):
    if r < 0.01: return "A"
    if r < 0.02: return "B"
    if r < 0.05: return "C"
    if r < 0.10: return "D"
    return "E"
def main():
    _, _, y, _, y3 = C.load_core(verbose=False)
    v7f3 = np.load(os.path.join(ROOT, "v8_out", "v8_reproduced_fold3.npy")).astype(float).ravel()
    assert abs(C.ap(y3, v7f3) - 0.5277283858077173) < 1e-9
    print(f"[v9] V7 fold3 AP={C.ap(y3, v7f3):.6f}", flush=True)
    order = np.argsort(-v7f3, kind="mergesort")
    ranks = np.empty(len(v7f3)); ranks[order] = (np.arange(len(v7f3)) + 1) / len(v7f3)
    bands = np.array([band_of(r) for r in ranks])
    M = np.column_stack([C.load_pred(os.path.join(ROOT, "v7_out", f"{m}_fold3.npy")) for m in ["graphlgb", "hgb183", "lgb183", "xgb183"]])
    mr = np.argsort(np.argsort(-M, axis=0, kind="mergesort"), axis=0) / len(M)
    best = mr.min(axis=1)
    dis = (ranks >= 0.01) & (best < 0.05)
    for b in ["A", "B", "C", "D", "E"]:
        m = bands == b
        print(f" band {b}: n={m.sum()} fraud={int(y3[m].sum())} ap={C.ap(y3[m], v7f3[m]):.4f} mean={v7f3[m].mean():.5f}", flush=True)
    print(f" FN outside top1%: {int(((y3==1)&(ranks>=0.01)).sum())}/{int(y3.sum())} disagree={int(dis.sum())}", flush=True)
    np.save(os.path.join(OUT, "_rank_f3.npy"), ranks)
    np.save(os.path.join(OUT, "_v7_f3.npy"), v7f3)
    C.jdump({"v7_fold3_ap": float(C.ap(y3, v7f3))}, os.path.join(OUT, "v9_ranking_bands.json"))
    print("DONE_A", flush=True)
if __name__ == "__main__":
    main()
