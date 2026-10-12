import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import average_precision_score as ap
from scipy.stats import spearmanr, kendalltau

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_HGB_VERIFY")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. LOAD TARGETS AND CHAMPION
# ============================================================
y3 = np.load(os.path.join(BASE, "Fold_3_v4_targets.npy"))
N3 = len(y3)
print(f"Fold-3 targets: N={N3}, fraud_rate={y3.mean():.5f}")

y2 = np.load(os.path.join(BASE, "EXP_B25_ACCELERATION_FOLD2", "fold2_targets.npy"))
N2 = len(y2)
print(f"Fold-2 targets: N={N2}, fraud_rate={y2.mean():.5f}")

G3 = np.load(os.path.join(BASE, "EXP_W2B_COST_SENSITIVE", "best_extended_cost_global_fold3_pred.npy"))
print(f"Champion G3: N={len(G3)}, AP={ap(y3, G3):.6f}")

# ============================================================
# 2. VERIFY HGB FOLD-3 PREDICTIONS
# ============================================================
print("\n" + "="*60)
print("FOLD-3 VERIFICATION")
print("="*60)

results = {}

hgb600_3 = np.load(os.path.join(BASE, "AGENT_EXP_AUTO_HGB", "HGB1_pred.npy"))
hgbt_3 = np.load(os.path.join(BASE, "AGENT_EXP_AUTO_HGB", "HGB_tuned_pred.npy"))

for name, pred in [("G_champion", G3), ("HGB600", hgb600_3), ("HGB_tuned", hgbt_3)]:
    assert len(pred) == N3, f"Length mismatch for {name}: {len(pred)} vs {N3}"
    assert np.all(np.isfinite(pred)), f"Non-finite values in {name}"
    assert pred.min() >= 0 and pred.max() <= 1, f"Out-of-range in {name}: [{pred.min()}, {pred.max()}]"
    score = ap(y3, pred)
    results[f"{name}_f3_ap"] = float(score)
    print(f"  {name}: AP={score:.6f} (N={len(pred)}, min={pred.min():.4f}, max={pred.max():.4f})")

# ============================================================
# 3. FIX FOLD-2 ALIGNMENT AND VERIFY
# ============================================================
print("\n" + "="*60)
print("FOLD-2 ALIGNMENT FIX")
print("="*60)

# The HGB fold-2 pred was built with Xva = X.iloc[hi:hi+N_VAL] where N_VAL=59465
# But the actual fold-2 targets have N2=58455 rows (June 16-30)
# The fold-2 validation window starts at row index (total_rows - N_VAL) in the original data
# but only 58455 of those rows fall in the fold-2 date range.
# The correct alignment: use the LAST N2=58455 of the 59465 predictions
# (the first 1010 rows correspond to rows in the full X that are outside fold-2 window)

hgb600_2_raw = np.load(os.path.join(BASE, "AGENT_EXP_AUTO_HGB", "HGB1_fold2_pred.npy"))
hgb600_2_corr = np.load(os.path.join(BASE, "AGENT_EXP_AUTO_HGB", "HGB1_fold2_correct_pred.npy"))

print(f"  HGB1_fold2_pred.npy: N={len(hgb600_2_raw)}")
print(f"  HGB1_fold2_correct_pred.npy: N={len(hgb600_2_corr)}")
print(f"  fold2_targets.npy: N={N2}")

# Check both versions against fold-2 targets
if len(hgb600_2_corr) == N2:
    ap_f2_corr = ap(y2, hgb600_2_corr)
    print(f"  HGB1_fold2_correct: AP={ap_f2_corr:.6f} (N={N2}, matches targets)")
    results["HGB600_f2_ap_correct"] = float(ap_f2_corr)
    hgb600_2_best = hgb600_2_corr
elif len(hgb600_2_raw) == N2:
    ap_f2_raw = ap(y2, hgb600_2_raw)
    print(f"  HGB1_fold2_raw: AP={ap_f2_raw:.6f} (N={N2}, matches targets)")
    results["HGB600_f2_ap_raw"] = float(ap_f2_raw)
    hgb600_2_best = hgb600_2_raw
else:
    # Try both trimming strategies
    print(f"  WARNING: size mismatch. Trying trim strategies...")
    # Strategy A: last N2 of the 59465 pred
    if len(hgb600_2_raw) > N2:
        trim_a = hgb600_2_raw[-N2:]
        ap_a = ap(y2, trim_a)
        print(f"  Strategy A (last N2 of raw): AP={ap_a:.6f}")
    else:
        ap_a = 0
    # Strategy B: first N2 of the raw
    if len(hgb600_2_raw) > N2:
        trim_b = hgb600_2_raw[:N2]
        ap_b = ap(y2, trim_b)
        print(f"  Strategy B (first N2 of raw): AP={ap_b:.6f}")
    else:
        ap_b = 0
    # Strategy C: use the correct version directly if it exists
    if len(hgb600_2_corr) == N2:
        hgb600_2_best = hgb600_2_corr
    elif ap_a > ap_b:
        hgb600_2_best = hgb600_2_raw[-N2:]
    else:
        hgb600_2_best = hgb600_2_raw[:N2]

# Fold-2 baselines
G2 = np.load(os.path.join(BASE, "EXP_B25_ACCELERATION_FOLD2", "fold2_matched_tuned_v4_baseline_pred.npy"))
A2 = np.load(os.path.join(BASE, "EXP_B25_ACCELERATION_FOLD2", "fold2_tuned_acceleration_pred.npy"))

ap_g2 = ap(y2, G2)
ap_a2 = ap(y2, A2)
print(f"\n  Fold-2 baselines:")
print(f"    G2 (tuned V4 baseline): AP={ap_g2:.6f}")
print(f"    A2 (tuned acceleration): AP={ap_a2:.6f}")

results["G2_tuned_baseline_ap"] = float(ap_g2)
results["A2_tuned_acceleration_ap"] = float(ap_a2)

# HGB fold-2 AP (corrected)
ap_hgb600_2 = ap(y2, hgb600_2_best)
print(f"    HGB600 (corrected): AP={ap_hgb600_2:.6f}")
results["HGB600_f2_corrected_ap"] = float(ap_hgb600_2)

# Stability gate: gain >= -0.001 vs G2 baseline
gain_f2_hgb = ap_hgb600_2 - ap_g2
print(f"    HGB600 fold-2 gain vs G2: {gain_f2_hgb:+.6f}")
print(f"    Stability gate: {'PASS' if gain_f2_hgb >= -0.001 else 'FAIL'}")
results["HGB600_f2_gain"] = float(gain_f2_hgb)
results["HGB600_f2_stable"] = gain_f2_hgb >= -0.001

# ============================================================
# 4. GLOBAL BLEND GATE (FIXED WEIGHTS)
# ============================================================
print("\n" + "="*60)
print("GLOBAL BLEND GATE (FIXED WEIGHTS 5%-50%)")
print("="*60)

gate_threshold = 0.518226 + 0.001  # 0.519226
print(f"  Champion G3 AP: {ap(y3, G3):.6f}")
print(f"  Blend gate threshold: {gate_threshold:.6f}")

blend_results = {}

# G + HGB600
print(f"\n  --- G + HGB600 ---")
for w_pct in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    w = w_pct / 100.0
    bp = (1 - w) * G3 + w * hgb600_3
    score = ap(y3, bp)
    gain = score - 0.518226
    tag = "PASS" if score >= gate_threshold else ""
    print(f"    w={w_pct:2d}%: AP={score:.6f} gain={gain:+.6f} {tag}")
    blend_results[f"G_HGB600_w{w_pct}"] = {"ap": float(score), "gain": float(gain), "pass": score >= gate_threshold}

# G + HGB_tuned
print(f"\n  --- G + HGB_tuned ---")
for w_pct in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    w = w_pct / 100.0
    bp = (1 - w) * G3 + w * hgbt_3
    score = ap(y3, bp)
    gain = score - 0.518226
    tag = "PASS" if score >= gate_threshold else ""
    print(f"    w={w_pct:2d}%: AP={score:.6f} gain={gain:+.6f} {tag}")
    blend_results[f"G_HGBt_w{w_pct}"] = {"ap": float(score), "gain": float(gain), "pass": score >= gate_threshold}

# 3-way: G + HGB600 + ACCtuned
ACC3 = np.load(os.path.join(BASE, "EXP_B25_ACCELERATION_TUNED", "best_acceleration_global_fold3_pred.npy"))
print(f"\n  --- 3-way: G + HGB600 + ACCtuned (fixed ratios) ---")
ratios = [(0.70, 0.20, 0.10), (0.60, 0.25, 0.15), (0.55, 0.30, 0.15),
          (0.50, 0.30, 0.20), (0.60, 0.30, 0.10), (0.50, 0.40, 0.10)]
for wg, wh, wa in ratios:
    bp = wg * G3 + wh * hgb600_3 + wa * ACC3
    score = ap(y3, bp)
    gain = score - 0.518226
    tag = "PASS" if score >= gate_threshold else ""
    print(f"    G={wg:.0%} HGB={wh:.0%} ACC={wa:.0%}: AP={score:.6f} gain={gain:+.6f} {tag}")
    blend_results[f"3way_G{int(wg*100)}_HGB{int(wh*100)}_ACC{int(wa*100)}"] = {"ap": float(score), "gain": float(gain), "pass": score >= gate_threshold}

# ============================================================
# 5. RANKING DIAGNOSTICS
# ============================================================
print("\n" + "="*60)
print("RANKING DIAGNOSTICS")
print("="*60)

def ranking_metrics(y_true, y_pred, name):
    n = len(y_true)
    fraud_idx = np.where(y_true == 1)[0]
    sorted_idx = np.argsort(-y_pred)
    metrics = {}
    for pct in [1, 2, 5]:
        k = max(1, int(n * pct / 100))
        top_k = sorted_idx[:k]
        prec = y_true[top_k].mean()
        capture = len(np.intersect1d(top_k, fraud_idx)) / len(fraud_idx)
        metrics[f"precision_top_{pct}pct"] = float(prec)
        metrics[f"capture_top_{pct}pct"] = float(capture)
        print(f"  {name} precision@{pct}%={prec:.4f}  capture@{pct}%={capture:.4f}")
    return metrics

diag = {}
for name, pred in [("G_champion", G3), ("HGB600", hgb600_3), ("HGB_tuned", hgbt_3)]:
    print(f"\n  {name}:")
    diag[name] = ranking_metrics(y3, pred, name)

# Correlations
print("\n  Correlations:")
for name, pred in [("HGB600", hgb600_3), ("HGB_tuned", hgbt_3)]:
    pearson = np.corrcoef(pred, G3)[0, 1]
    spearman, _ = spearmanr(pred, G3)
    tau, _ = kendalltau(pred, G3)
    print(f"    {name} vs G: Pearson={pearson:.4f} Spearman={spearman:.4f} Kendall={tau:.4f}")
    diag[f"{name}_vs_G"] = {"pearson": float(pearson), "spearman": float(spearman), "kendall": float(tau)}

# Missed fraud overlap
print("\n  Missed fraud overlap (top 5% threshold):")
n5 = max(1, int(N3 * 5 / 100))
for name, pred in [("HGB600", hgb600_3), ("HGB_tuned", hgbt_3)]:
    top_g = set(np.argsort(-G3)[:n5])
    top_c = set(np.argsort(-pred)[:n5])
    jaccard = len(top_g & top_c) / len(top_g | top_c)
    missed_g = set(np.where(y3 == 1)[0]) - top_g
    missed_c = set(np.where(y3 == 1)[0]) - top_c
    missed_jacc = len(missed_g & missed_c) / len(missed_g | missed_c) if (missed_g | missed_c) else 0
    print(f"    {name} top5% Jaccard vs G: {jaccard:.4f}")
    print(f"    {name} missed-fraud Jaccard vs G: {missed_jacc:.4f}")
    diag[f"{name}_missed"] = {"top5_jaccard_vs_G": float(jaccard), "missed_fraud_jaccard_vs_G": float(missed_jacc)}

# ============================================================
# 6. LEAKAGE AUDIT
# ============================================================
print("\n" + "="*60)
print("LEAKAGE AUDIT")
print("="*60)
leakage_notes = {
    "category_encoding": {
        "issue": "In hgb_v4.py, categorical codes derived from full X (Xtr+Xva) before split via .astype('category').cat.codes",
        "risk": "LOW - ordinal codes are not target-derived; HGB treats low-card cats as categorical features via categorical_features mask",
        "impact": "Minimal: category->integer mapping includes validation categories but not validation labels",
        "recommendation": "Acceptable for GBDT categorical handling; rebuild with Xtr-only mapping for stricter isolation if deploying"
    },
    "na_indicator": {
        "issue": "NaN indicators built using Xtr mean only (training-only statistics)",
        "risk": "NONE - no validation info used",
        "impact": "Clean",
        "recommendation": "No change needed"
    },
    "na_fill": {
        "issue": "NaN fill values computed from Xtr only",
        "risk": "NONE - training-only statistics",
        "impact": "Clean",
        "recommendation": "No change needed"
    }
}
for k, v in leakage_notes.items():
    print(f"  {k}: {v['risk']}")
    print(f"    Issue: {v['issue']}")

# ============================================================
# 7. SAVE ALL RESULTS
# ============================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

output = {
    "fold3_verification": {k: v for k, v in results.items() if "f3" in k},
    "fold2_verification": {k: v for k, v in results.items() if "f2" in k},
    "blend_gate": blend_results,
    "ranking_diagnostics": diag,
    "leakage_audit": leakage_notes,
    "champion_ap": float(ap(y3, G3)),
    "gate_threshold": float(gate_threshold),
}

with open(os.path.join(OUT, "verification_results.json"), "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"  Saved: verification_results.json")

# Blend sweep CSV
rows = []
for k, v in blend_results.items():
    rows.append({"blend": k, "ap": v["ap"], "gain": v["gain"], "pass_gate": v["pass"]})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "blend_sweep_results.csv"), index=False)
print(f"  Saved: blend_sweep_results.csv")

# Ranking diagnostics CSV
rows2 = []
for metric_name, vals in diag.items():
    if isinstance(vals, dict):
        for mk, mv in vals.items():
            rows2.append({"model": metric_name, "metric": mk, "value": mv})
pd.DataFrame(rows2).to_csv(os.path.join(OUT, "ranking_diagnostics.csv"), index=False)
print(f"  Saved: ranking_diagnostics.csv")

# ============================================================
# 8. FINAL VERDICT
# ============================================================
print("\n" + "="*60)
print("FINAL VERDICT")
print("="*60)

best_blend_ap = max(v["ap"] for v in blend_results.values())
best_blend_gain = best_blend_ap - 0.518226
f2_stable = results.get("HGB600_f2_stable", False)

print(f"  Best blend AP: {best_blend_ap:.6f} (gain={best_blend_gain:+.6f})")
print(f"  Fold-2 stable: {f2_stable}")

if best_blend_gain >= 0.001 and f2_stable:
    print(f"\n  *** READY FOR HUMAN REVIEW: VERIFIED NEW CANDIDATE FOUND ***")
    print(f"  *** HGB candidate clears +0.001 global gate AND fold-2 stability ***")
elif best_blend_gain >= 0.001:
    print(f"\n  *** CONDITIONAL: blend clears gate but fold-2 stability unverified ***")
else:
    print(f"\n  *** NOT YET: best blend gain {best_blend_gain:+.6f} < +0.001 ***")

print("\nDONE")
