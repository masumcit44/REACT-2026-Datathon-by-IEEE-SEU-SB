import numpy as np, os, pandas as pd
from sklearn.metrics import average_precision_score as ap
NB = r"jupyter notebook"
OUT = os.path.join(NB, "AGENT_EXP_AUTO_CATBOOST_TUNED")
y = np.load(os.path.join(NB, "Fold_3_v4_targets.npy"))
G = np.load(os.path.join(NB, "EXP_W2B_COST_SENSITIVE", "best_extended_cost_global_fold3_pred.npy"))
CB = np.load(os.path.join(OUT, "pruned103_noe_noe_fold3_pred.npy"))
X = np.load(os.path.join(NB, "EXP_W4_EXTRA_RANDOMIZED", "w4_best_individual_fold3_pred.npy"))
def logit(p): p = np.clip(p, 1e-7, 1-1e-7); return np.log(p/(1-p))
Lg, Lcb, Lx = logit(G), logit(CB), logit(X)
best = {}
results = []
for w1 in [0.1, 0.35, 0.5, 0.6, 0.7]:
    for w2 in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for w3 in [0.10, 0.15, 0.20, 0.25, 0.30]:
            if w1 + w2 + w3 != 1.0: continue
            bl = w1*Lg + w2*Lcb + w3*Lx
            s = ap(y, 1/(1+np.exp(-bl)))
            results.append((s, w1, w2, w3))
for s, w1, w2, w3 in sorted(results, reverse=True)[:10]:
    print(f"G={w1:.2f} CB={w2:.2f} X={w3:.2f} -> AP={s:.6f} gain={s-0.518226:+.6f}")
print("BEST:", sorted(results, reverse=True)[0])
pd.DataFrame(results, columns=["AP", "w_G", "w_CB", "w_X"]).sort_values("AP", ascending=False).to_csv(
    os.path.join(OUT, "three_way_stack.csv"), index=False)