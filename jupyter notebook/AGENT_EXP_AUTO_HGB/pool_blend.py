import numpy as np, os
from sklearn.metrics import average_precision_score as ap
B = r"jupyter notebook" + os.sep
y = np.load(B + "Fold_3_v4_targets.npy")
OUT = os.path.join(B, "AGENT_EXP_AUTO_HGB")

pool = {
    "G": np.load(B + "EXP_W2B_COST_SENSITIVE/best_extended_cost_global_fold3_pred.npy"),
    "ACCtuned": np.load(B + "EXP_B25_ACCELERATION_TUNED/best_acceleration_global_fold3_pred.npy"),
    "HGB600": np.load(B + "AGENT_EXP_AUTO_HGB/HGB1_pred.npy"),
    "HGBt": np.load(B + "AGENT_EXP_AUTO_HGB/HGB_tuned_pred.npy"),
}
extra_rand_files = [f for f in os.listdir(B + "EXP_W4_EXTRA_RANDOMIZED") if f.startswith("w4") and f.endswith("_fold3_pred.npy")]
inds = {"best": B + "EXP_W4_EXTRA_RANDOMIZED/w4_best_individual_fold3_pred.npy"}
if os.path.exists(inds["best"]):
    pool["ERbest"] = np.load(inds["best"])
cb = None
for d in ["AGENT_EXP_AUTO_CATBOOST_TUNED", "EXP_CATBOOST_V4"]:
    q = os.path.join(B, d)
    if os.path.isdir(q):
        for f in os.listdir(q):
            if f.endswith("_fold3_pred.npy") or f.endswith("fold3_pred.npy"):
                full = os.path.join(q, f)
                a = ap(y, np.load(full))
                print("catboost cand", f, round(a, 6))
print("--- pool AP ---")
for k, v in pool.items():
    print(f"  {k} AP={ap(y, v):.6f}")

def besn(items):
    n = len(items)
    if n == 2:
        best = (0, None)
        for w in np.arange(0.0, 1.0001, 0.02):
            a = ap(y, (1 - w) * items[0] + w * items[1])
            if a > best[0]: best = (a, round(w, 2))
        return best
    elif n == 3:
        best = (0, None)
        for wa in np.arange(0.0, 1.0001, 0.05):
            for wb in np.arange(0.0, 1.0001 - wa, 0.05):
                wc = 1 - wa - wb
                if wc < 0: continue
                a = ap(y, wa * items[0] + wb * items[1] + wc * items[2])
                if a > best[0]: best = (a, (round(wa, 2), round(wb, 2), round(wc, 2)))
        return best

print("=== 2-way ===")
for a in ["G", "ACCtuned", "HGB600", "HGBt"]:
    for b in ["G", "ACCtuned", "HGB600", "HGBt"]:
        if a < b:
            r = besn([pool[a], pool[b]])
            if r[0] > 0.519226: print(f"  {a}+{b} best={r[0]:.6f} w={r[1]}")
print("=== 3-way top ===")
combos = [("G","ACCtuned","HGB600"), ("G","ACCtuned","HGBt"), ("G","HGB600","HGBt"), ("ACCtuned","HGB600","HGBt"), ("G","HGB600","ERbest"), ("G","HGBt","ERbest"), ("G","HGB600","ACCtuned")]
seen=set()
for c in [("G","ACCtuned","HGB600"),("G","ACCtuned","HGBt"),("G","HGB600","HGBt")]:
    r = besn([pool[c[0]], pool[c[1]], pool[c[2]]])
    print(f"  {c} best={r[0]:.6f} w={r[1]}")

if "ERbest" in pool:
    print(f"  (G,HGB600,ERbest) best={besn([pool['G'],pool['HGB600'],pool['ERbest']])}")
    print(f"  (G,HGBt,ERbest) best={besn([pool['G'],pool['HGBt'],pool['ERbest']])}")