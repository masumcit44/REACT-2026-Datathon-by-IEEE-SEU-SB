import numpy as np, os, pandas as pd
from sklearn.metrics import average_precision_score as ap
NB = r"jupyter notebook"
OUT = os.path.join(NB, "AGENT_EXP_AUTO_CATBOOST_TUNED")
y = np.load(os.path.join(NB, "Fold_3_v4_targets.npy"))
def load(p): return np.load(p)
G = load(os.path.join(NB, "EXP_W2B_COST_SENSITIVE", "best_extended_cost_global_fold3_pred.npy"))
members = {
    "G": G,
    "CB_full110": load(os.path.join(OUT, "full110_noe_noe_fold3_pred.npy")),
    "CB_pruned103": load(os.path.join(OUT, "pruned103_noe_noe_fold3_pred.npy")),
    "CB_known101": load(os.path.join(OUT, "known_good_101_fold3_pred.npy")),
    "CB_OLD": load("catboost_v4_fold3_pred.npy"),
}
def logit(p):
    p = np.clip(p, 1e-7, 1-1e-7); return np.log(p/(1-p))
print("members AP:", {k: round(ap(y, v), 6) for k, v in members.items()})
print("corr with G:", {k: round(np.corrcoef(v, G)[0, 1], 4) for k, v in members.items()})
best = (0, None)
W = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
for k in members:
    if k == "G": continue
    for w in W:
        b = logit((1-w)*G + w*members[k])  # NOT correct logit mix; fix below
# correct: weighted average of logits
results = []
for k in members:
    if k == "G": continue
    Lk = logit(members[k]); Lg = logit(G)
    for w in W:
        bl = (1-w)*Lg + w*Lk
        s = ap(y, 1/(1+np.exp(-bl)))
        results.append((s, k, w))
for s, k, w in sorted(results, reverse=True)[:10]:
    print(f"blend {k} w={w:.2f} -> AP={s:.6f} gain={s-0.518226:+.6f}")
pd.DataFrame(results, columns=["AP", "member", "weight"]).sort_values("AP", ascending=False).to_csv(
    os.path.join(OUT, "champion_catboost_blend.csv"), index=False)
print("BEST:", sorted(results, reverse=True)[0])