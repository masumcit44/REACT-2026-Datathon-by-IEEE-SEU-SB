import numpy as np, pandas as pd, os
from sklearn.metrics import average_precision_score as ap

NB = r"jupyter notebook"
OUT = os.path.join(NB, "AGENT_EXP_AUTO_BLEND_SEARCH")
y = np.load(os.path.join(NB, "Fold_3_v4_targets.npy"))
members = {
    "G_cost_final": os.path.join(NB, "EXP_W2B_COST_SENSITIVE", "best_extended_cost_global_fold3_pred.npy"),
    "W4_stageB":    os.path.join(NB, "EXP_W4_EXTRA_RANDOMIZED", "w4_best_global_blend_fold3_pred.npy"),
    "X_extra_rand": os.path.join(NB, "EXP_W4_EXTRA_RANDOMIZED", "w4_best_individual_fold3_pred.npy"),
    "CB_catboost":  "catboost_v4_fold3_pred.npy",
    "TA_accel":     os.path.join(NB, "EXP_B25_ACCELERATION_TUNED", "tuned_acceleration_fold3_pred.npy"),
    "V4_best3":     os.path.join(NB, "v4_best_three_model_fold3_prediction.npy"),
    "OLD_5076":     "Fold_3_v4_predictions.npy",
    "FGA_accelG":   os.path.join(NB, "EXP_B25_ACCELERATION_TUNED", "best_acceleration_global_fold3_pred.npy"),
}
preds = {k: np.load(f) for k, f in members.items()}
def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))
names_all = list(preds)

# Greedy forward: start {G}, repeatedly add the member/method/weight that improves AP most
cur = {"G_cost_final"}
# weights over up to 4 members; search simplex via coarse grid on logit space
import itertools
W = [0.05,0.10,0.15,0.20,0.25,0.30,0.40]
def best_add(selected, tried_globals):
    besthit = None
    cand = [k for k in names_all if k not in selected]
    for k in cand:
        for w in W:
            L = logit(preds[k])
            # current logit blend
            Lcur = sum(logit(preds[m]) * (1-w)/ (len(selected)) for m in selected)
            # scale: give k weight w, remaining 1-w split evenly
            Lb = (1 - w) * np.mean(np.stack([logit(preds[m]) for m in selected]), 0) + w * L
            s = ap(y, Lb)
            if besthit is None or s > besthit[0]:
                besthit = (s, k, w, "logit")
    return besthit

# coarse simlex search over combos of size 3 and 4 via coordinate search is heavy; use forward greedy w/ simplex normalize each step
selected = ["G_cost_final"]
steps = []
run = {"G_cost_final": 1.0}
for it in range(4):
    best = None
    for k in [x for x in names_all if x not in selected]:
        for m in ["logit", "prob"]:
            for w in W:
                cols = selected + [k]
                ws = {c: (1 - w) / len(selected) for c in selected}
                ws[k] = w
                M = np.stack([logit(preds[c]) if m == "logit" else preds[c] for c in cols])
                wv = np.array([ws[c] for c in cols])
                pb = np.clip((M * wv[:, None]).sum(0), 1e-12, 1)
                s = ap(y, pb)
                if best is None or s > best[0]:
                    best = (s, k, m, w, ws)
    if best is None: break
    s, k, m, w, ws = best
    gain = s - 0.518226
    steps.append({"iter": it + 1, "added": k, "method": m, "member_weight": w,
                  "AP": round(s, 6), "gain": round(gain, 6)})
    print(f"iter {it+1}: add {k} {m} w={w} -> AP={s:.6f} gain={gain:+.6f}")
    if gain < +0.0002:  # stop adding when no meaningful gain
        print("no meaningful further gain; stop adding members")
        break
    selected.append(k)

print("FINAL SELECTED:", selected)
pd.DataFrame(steps).to_csv(os.path.join(OUT, "blend_greedy_forward.csv"), index=False)