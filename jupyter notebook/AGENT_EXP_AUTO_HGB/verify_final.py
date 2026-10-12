import numpy as np, os
from sklearn.metrics import average_precision_score as ap
B = r"jupyter notebook" + os.sep
y = np.load(B + "Fold_3_v4_targets.npy")
G = np.load(B + "EXP_W2B_COST_SENSITIVE/best_extended_cost_global_fold3_pred.npy")
AC = np.load(B + "EXP_B25_ACCELERATION_TUNED/best_acceleration_global_fold3_pred.npy")
H6 = np.load(B + "AGENT_EXP_AUTO_HGB/HGB1_pred.npy")
Ht = np.load(B + "AGENT_EXP_AUTO_HGB/HGB_tuned_pred.npy")

print("VERIFICATION: individual APs")
for n,p in [("G",G),("ACCtuned",AC),("HGB600",H6),("HGBt",Ht)]:
    print(f"  {n}={ap(y,p):.6f}")

print("VERIFICATION: sweep verification around best")
# ACCtuned + HGB600 around best
for w in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    bp = (1-w)*AC + w*H6
    print(f"  ACCtuned+HGB600@{w:.2f}: AP={ap(y,bp):.6f}")

print("\nG+HGB600 sweep:")
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    bp = (1-w)*G + w*H6
    print(f"  G+HGB600@{w:.2f}: AP={ap(y,bp):.6f}")

# fold-2 target if exists
f2p = B + "Fold_2_v4_targets.npy"
if os.path.exists(f2p):
    y2 = np.load(f2p)
    G2 = np.load(B + "EXP_W2B_COST_SENSITIVE/best_extended_cost_global_fold2_pred.npy") if os.path.exists(B + "EXP_W2B_COST_SENSITIVE/best_extended_cost_global_fold2_pred.npy") else None
    AC2 = np.load(B + "EXP_B25_ACCELERATION_TUNED/best_acceleration_global_fold2_pred.npy") if os.path.exists(B + "EXP_B25_ACCELERATION_TUNED/best_acceleration_global_fold2_pred.npy") else None
    HGB2 = np.load(B + "AGENT_EXP_AUTO_HGB/HGB1_fold2_pred.npy")
    print("\nfold-2 AP:")
    if G2 is not None: print(f"  G2={ap(y2,G2):.6f}")
    if AC2 is not None: print(f"  AC2={ap(y2,AC2):.6f}")
    print(f"  HGB2={ap(y2,HGB2):.6f}")
    if AC2 is not None:
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            bp = (1-w)*AC2 + w*HGB2
            print(f"  AC2+HGB2@{w:.2f}: AP={ap(y2,bp):.6f}")
    elif G2 is not None:
        for w in [0.0, 0.3, 0.5]:
            bp = (1-w)*G2 + w*HGB2
            print(f"  G2+HGB2@{w:.2f}: AP={ap(y2,bp):.6f}")
print("\nDONE")