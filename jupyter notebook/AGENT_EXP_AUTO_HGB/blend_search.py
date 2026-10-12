import numpy as np, os
from sklearn.metrics import average_precision_score as ap
BASE = r"jupyter notebook"
y = np.load(os.path.join(BASE, "Fold_3_v4_targets.npy"))
G = np.load(os.path.join(BASE, "EXP_W2B_COST_SENSITIVE", "best_extended_cost_global_fold3_pred.npy"))
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_HGB")
h1 = np.load(os.path.join(OUT, "HGB1_pred.npy"))
ht = np.load(os.path.join(OUT, "HGB_tuned_pred.npy"))
cands = {"G": G, "HGB600": h1, "HGBt": ht}
print("individuals:")
for k, p in cands.items():
    print(f"  {k} AP={ap(y, p):.6f}")
print("corr matrix:")
names = list(cands)
M = np.array([[np.corrcoef(cands[a], cands[b])[0, 1] for b in names] for a in names])
print(names)
print(np.round(M, 4))

def sweep(*items):
    # items = list of (name, pred). grid over weights
    n = len(items)
    if n == 2:
        best = (0, None)
        for w in np.arange(0.0, 1.001, 0.05):
            bp = (1 - w) * items[0][1] + w * items[1][1]
            a = ap(y, bp)
            if a > best[0]:
                best = (a, (round(w, 2), round(1 - w, 2)))
        print(f"  2-way {[i[0] for i in items]} best {best[0]:.6f} w={best[1]}")
        return best
    if n == 3:
        best = (0, None)
        for wa in np.arange(0.0, 1.001, 0.05):
            for wb in np.arange(0.0, 1.001 - wa + 1e-6, 0.05):
                wc = 1 - wa - wb
                bp = wa * items[0][1] + wb * items[1][1] + wc * items[2][1]
                a = ap(y, bp)
                if a > best[0]:
                    best = (a, (round(wa, 2), round(wb, 2), round(wc, 2)))
        print(f"  3-way {[i[0] for i in items]} best {best[0]:.6f} w={best[1]}")
        return best

print("--- blends ---")
sweep(("G", G), ("HGB600", h1))
sweep(("G", G), ("HGBt", ht))
sweep(("G", G), ("HGB600", h1), ("HGBt", ht))
best3 = sweep(("G", G), ("HGB600", h1), ("HGBt", ht), )