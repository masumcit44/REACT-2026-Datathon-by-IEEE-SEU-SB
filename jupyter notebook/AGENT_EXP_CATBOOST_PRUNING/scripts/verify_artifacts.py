import numpy as np, os, glob, json
from sklearn.metrics import average_precision_score as ap

BASE = r"D:\Desktop\React datathon"
NB = os.path.join(BASE, "jupyter notebook")
OUT = os.path.join(NB, "AGENT_EXP_CATBOOST_PRUNING")

tgt = np.load(os.path.join(NB, "Fold_3_v4_targets.npy")).astype(float)

records = []
def check(name, path, expected=None, tol=2e-4):
    try:
        a = np.load(path)
        a = np.asarray(a, dtype=float).ravel()
        if len(a) != len(tgt):
            ok = "SHAPE_MISMATCH"
            score = np.nan
        else:
            score = ap(tgt, a)
            ok = "OK" if expected is None or abs(score - expected) <= tol else "MISMATCH"
        records.append({"artifact": name, "path": os.path.relpath(path, NB),
                        "n": len(a), "AP_recomputed": round(score, 6),
                        "recorded_AP": "" if expected is None else round(expected, 6),
                        "status": ok})
    except Exception as e:
        records.append({"artifact": name, "path": os.path.relpath(path, NB),
                        "n": -1, "AP_recomputed": np.nan, "recorded_AP": "",
                        "status": "ERROR: %s" % e})

# Root-level fold3 preds
check("catboost_v4_fold3_pred", os.path.join(BASE, "catboost_v4_fold3_pred.npy"))
check("Fold_3_v4_predictions", os.path.join(BASE, "Fold_3_v4_predictions.npy"))
check("v4_best_three_model_fold3_prediction", os.path.join(NB, "v4_best_three_model_fold3_prediction.npy"))

# Glob all fold3 preds
for f in sorted(glob.glob(os.path.join(NB, "**", "*fold3*.npy"), recursive=True)):
    check(os.path.basename(f), f)

# Recorded-value cross-checks from experiment CSVs
expected_map = {
    "best_extended_cost_global_fold3_pred.npy": 0.5182256022283107,
    "best_cost_global_blend_fold3_pred.npy": 0.5181935759348215,
    "w4_best_global_blend_fold3_pred.npy": 0.516533367110975,
    "w4_best_individual_fold3_pred.npy": 0.511358770385716,
    "w4_best_local_blend_fold3_pred.npy": 0.5115474695566031,
    "tuned_acceleration_fold3_pred.npy": 0.5128941468791068,
    "matched_tuned_v4_baseline_fold3_pred.npy": 0.5095137592687167,
    "top3_acceleration_fold3_pred.npy": 0.5124091242787774,
    "top7_acceleration_fold3_pred.npy": 0.512186,
    "acceleration_fold3_probe_pred.npy": 0.5109534157519429,
    "step5a_channel_fold3_cheap_pred.npy": 0.5062009011672867,
    "entity_age_density_fold3_cheap_pred.npy": 0.5068780790135216,
    "population_percentile_fold3_cheap_pred.npy": 0.5065267189359264,
}
for rec in records:
    base_name = os.path.basename(rec["path"])
    if base_name in expected_map:
        rec["recorded_AP"] = round(expected_map[base_name], 6)
        rec["status"] = "OK" if abs(rec["AP_recomputed"] - expected_map[base_name]) <= 2e-4 else "MISMATCH"

import csv
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "artifact_verification.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["artifact", "path", "n", "AP_recomputed", "recorded_AP", "status"])
    w.writeheader()
    for r in records:
        w.writerow(r)

nok = sum(1 for r in records if r["status"] == "OK")
print("Verified %d/%d artifacts OK" % (nok, len(records)))
for r in records:
    if r["status"] != "OK":
        print("  NOT-OK:", r)