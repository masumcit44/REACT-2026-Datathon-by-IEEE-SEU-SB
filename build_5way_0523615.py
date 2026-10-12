import numpy as np, pandas as pd, os
from sklearn.metrics import average_precision_score as ap

B = "jupyter notebook" + os.sep
OUT = "submission_file"
W = {"champ":0.05, "hgb600":0.25, "hgbt":0.30, "megalgb":0.25, "megacb":0.15}

y3 = np.load(B + "Fold_3_v4_targets.npy")

# fold-3 members
f3 = {
 "champ":   np.load(B + "EXP_W2B_COST_SENSITIVE/best_extended_cost_global_fold3_pred.npy"),
 "hgb600":  np.load(B + "AGENT_EXP_AUTO_HGB/HGB600_repro_fold3_pred.npy"),
 "hgbt":    np.load(B + "AGENT_EXP_AUTO_HGB/HGB_tuned_pred.npy"),
 "megalgb": np.load(B + "AGENT_EXP_MEGA_V2/lgb_mega_fold3_pred.npy"),
 "megacb":  np.load(B + "AGENT_EXP_MEGA_V2/cb_mega_fold3_pred.npy"),
}
# sanity: reproduce documented 0.523615 using ORIGINAL HGB600 fold-3 pred
h6_orig = np.load(B + "AGENT_EXP_AUTO_HGB/HGB1_pred.npy")
f3_orig = dict(f3); f3_orig["hgb600"] = h6_orig
blend_orig = sum(W[k]*f3_orig[k] for k in W)
print(f"5-way fold-3 with ORIGINAL hgb600 = {ap(y3, blend_orig):.6f}  (doc target 0.523615)")

# current repro version
blend3 = sum(W[k]*f3[k] for k in W)
print(f"5-way fold-3 with REPRO hgb600    = {ap(y3, blend3):.6f}  (+{ap(y3,blend3)-0.518226:.6f} vs champ)")

# fold-2 check (partial - only hgb600 has fold-2)
y2 = np.load(B + "Fold_2_v4_targets.npy")
try:
    h6_f2 = np.load(B + "AGENT_EXP_AUTO_HGB_VERIFY/HGB600_fold2_temporal_pred.npy")
    print(f"HGB600 fold2-temporal pred length {len(h6_f2)} vs y2 {len(y2)}")
    if len(h6_f2) == len(y2):
        print(f"HGB600 fold-2 AP = {ap(y2, h6_f2):.6f}")
except Exception as e:
    print(f"fold-2 check skipped: {e}")

# ---- build test submission ----
test = {
 "champ":   np.load(B + "FINAL_0518226_SUBMISSION/final_nested_0518226_test_pred.npy"),
 "hgb600":  np.load(B + "AGENT_EXP_AUTO_HGB/HGB600_fulldata_test_pred.npy"),
 "hgbt":    np.load(B + "AGENT_EXP_AUTO_HGB_VERIFY/HGB_tuned_fulldata_test_pred.npy"),
 "megalgb": np.load(B + "AGENT_EXP_MEGA_V2/lgb_mega_test_pred.npy"),
 "megacb":  np.load(B + "AGENT_EXP_MEGA_V2/cb_mega_test_pred.npy"),
}
n = len(test["champ"])
for k in test: assert len(test[k]) == n, k
blend_test = sum(W[k]*test[k] for k in W)

ss = pd.read_csv("sample_submission.csv")
sub = ss.copy(); sub["fraud"] = blend_test
path = os.path.join(OUT, "mega_5way_hgb600_0523382.csv")
sub.to_csv(path, index=False)
print(f"Saved {path}  rows={len(sub)} mean={blend_test.mean():.6f}")
np.save("jupyter notebook/AGENT_EXP_AUTO_HGB/HGB600_5way_blend_test_pred.npy", blend_test)

print("=== test pred stats ===")
for k in test:
    print(f"  {k:8s} mean={test[k].mean():.6f} min={test[k].min():.6f} max={test[k].max():.6f}")
print("DONE")