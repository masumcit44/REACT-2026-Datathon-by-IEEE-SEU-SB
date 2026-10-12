import numpy as np, pandas as pd, os, time, json
from sklearn.metrics import average_precision_score as ap
from scipy.stats import spearmanr

NB = r"jupyter notebook"
OUT = os.path.join(NB, "AGENT_EXP_AUTO_BLEND_SEARCH")
os.makedirs(OUT, exist_ok=True)

y = np.load(os.path.join(NB, "Fold_3_v4_targets.npy"))
members = {
    "G_cost_final":   os.path.join(NB, "EXP_W2B_COST_SENSITIVE", "best_extended_cost_global_fold3_pred.npy"),
    "W4_stageB":      os.path.join(NB, "EXP_W4_EXTRA_RANDOMIZED", "w4_best_global_blend_fold3_pred.npy"),
    "X_extra_rand":   os.path.join(NB, "EXP_W4_EXTRA_RANDOMIZED", "w4_best_individual_fold3_pred.npy"),
    "CB_catboost":    "catboost_v4_fold3_pred.npy",
    "TA_accel":       os.path.join(NB, "EXP_B25_ACCELERATION_TUNED", "tuned_acceleration_fold3_pred.npy"),
    "V4_best3":       os.path.join(NB, "v4_best_three_model_fold3_prediction.npy"),
    "OLD_5076":       "Fold_3_v4_predictions.npy",
}
preds = {}
for k, f in members.items():
    p = np.load(f)
    assert len(p) == len(y) and np.isfinite(p).all(), k
    preds[k] = p

def apc(y, p): return ap(y, p)
def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1.0 - p))
def blend(method, ws):
    names = list(ws)
    w = np.array([ws[n] for n in names]); w = w / w.sum()
    Ms = np.stack([preds[n] for n in names], 1)
    if method == "prob":
        return (Ms * w[None, :]).sum(1)
    if method == "logit":
        return (logit(Ms) * w[None, :]).sum(1)
    if method == "rank":
        R = np.stack([pd.Series(preds[n]).rank(pct=True).values for n in names], 1)
        return (R * w[None, :]).sum(1)

W0 = 0.518226
base = "G_cost_final"
print(f"baseline G AP = {apc(y, preds[base]):.6f}")

best = {"prob": (base, 1.0, apc(y, preds[base]))}
grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# stage 1: add one member to base
results = []
adds = {}
for other in [k for k in preds if k != base]:
    for m in ["prob", "logit", "rank"]:
        for w in grid:
            bp = blend(m, {base: 1 - w, other: w})
            s = apc(y, bp)
            results.append((m, other, w, s, s - W0))
            if other not in adds or s > adds[other][1]:
                adds[other] = (m, w, s, s - W0)

df = pd.DataFrame(results, columns=["method", "member", "weight", "AP", "gain"])
df.sort_values("gain", ascending=False).head(15).to_csv(os.path.join(OUT, "blend_stage1_addone.csv"), index=False)
print("BEST ADD-ONE:")
print(df.sort_values("gain", ascending=False).head(8).to_string())

names_all = list(preds)
M = np.stack([preds[n] for n in names_all])  # (k, n)
corr = pd.DataFrame(np.corrcoef(M, rowvar=True), index=names_all, columns=names_all).round(4)
corr.to_csv(os.path.join(OUT, "correlation_matrix.csv"))
sc = spearmanr(M, axis=1).statistic  # (k,k)
scdf = pd.DataFrame(1.0, index=list(preds), columns=list(preds))
for i, a in enumerate(names_all):
    for j, b in enumerate(names_all):
        scdf.loc[a, b] = round(sc[i, j], 4)
scdf.to_csv(os.path.join(OUT, "rank_correlation_matrix.csv"))

json.dump({
    "baseline": float(apc(y, preds[base])),
    "best_addone": {k: {"method": v[0], "weight": v[1], "AP": round(v[2], 6), "gain": round(v[3], 6)} for k, v in adds.items()},
}, open(os.path.join(OUT, "blend_addone_summary.json"), "w"), indent=2)
print(json.dumps(adds, indent=1, default=str, sort_keys=True))