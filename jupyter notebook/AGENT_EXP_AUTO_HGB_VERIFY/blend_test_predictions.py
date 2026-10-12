import numpy as np, os, json

BASE = r"jupyter notebook"
OUT = os.path.join(BASE, "AGENT_EXP_AUTO_HGB_VERIFY")

print("Loading predictions...")
champion_test = np.load(os.path.join(BASE, "FINAL_0518226_SUBMISSION", "final_nested_0518226_test_pred.npy"))
hgb_test = np.load(os.path.join(OUT, "HGB_tuned_fulldata_test_pred.npy"))

print(f"Champion: N={len(champion_test)}, mean={champion_test.mean():.6f}")
print(f"HGB:      N={len(hgb_test)}, mean={hgb_test.mean():.6f}")
print(f"Correlation: {np.corrcoef(champion_test, hgb_test)[0,1]:.4f}")

# Load acceleration test pred if available
acc_test = None
for p in [os.path.join(BASE, "EXP_B25_ACCELERATION_TUNED", "best_acceleration_global_test_pred.npy"),
          os.path.join(BASE, "EXP_B25_ACCELERATION", "acceleration_test_pred.npy")]:
    if os.path.exists(p):
        acc_test = np.load(p)
        print(f"ACC:      N={len(acc_test)}, mean={acc_test.mean():.6f}")
        break

print("\n--- Blend sweep ---")
results = []

# 2-way: G + HGBt
for w_pct in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    w = w_pct / 100.0
    bp = (1 - w) * champion_test + w * hgb_test
    tag = f"G_HGBt_w{w_pct}"
    results.append({"blend": tag, "mean": float(bp.mean()), "std": float(bp.std())})
    print(f"  {tag}: mean={bp.mean():.6f}")

# 3-way: G + HGBt + ACC
if acc_test is not None:
    for wg, wh, wa in [(0.50, 0.40, 0.10), (0.55, 0.30, 0.15), (0.60, 0.25, 0.15), (0.70, 0.20, 0.10)]:
        bp = wg * champion_test + wh * hgb_test + wa * acc_test
        tag = f"3way_G{int(wg*100)}_HGB{int(wh*100)}_ACC{int(wa*100)}"
        results.append({"blend": tag, "mean": float(bp.mean()), "std": float(bp.std())})
        print(f"  {tag}: mean={bp.mean():.6f}")

# Save recommended blend (G+HGBt@45%)
w_rec = 0.45
blend_rec = (1 - w_rec) * champion_test + w_rec * hgb_test
rec_path = os.path.join(OUT, "HGB_blend_45pct_test_pred.npy")
np.save(rec_path, blend_rec)
print(f"\nRecommended blend saved: {rec_path}")

# Save best 3-way blend (50/40/10)
if acc_test is not None:
    blend_3way = 0.50 * champion_test + 0.40 * hgb_test + 0.10 * acc_test
    path_3way = os.path.join(OUT, "HGB_3way_50_40_10_test_pred.npy")
    np.save(path_3way, blend_3way)
    print(f"3-way blend saved: {path_3way}")

# Verify saved artifacts
print("\n--- Verification ---")
for name, path in [("HGB test", os.path.join(OUT, "HGB_tuned_fulldata_test_pred.npy")),
                    ("Recommended blend", rec_path)]:
    p = np.load(path)
    assert len(p) == 262648, f"Wrong length: {len(p)}"
    assert np.all(np.isfinite(p)), "Non-finite values"
    assert p.min() >= 0 and p.max() <= 1, f"Out of range"
    print(f"  {name}: OK (N={len(p)}, [{p.min():.4f}, {p.max():.4f}])")

# Save summary
summary = {
    "champion_test_mean": float(champion_test.mean()),
    "hgb_test_mean": float(hgb_test.mean()),
    "recommended_blend": "G+HGBt@45%",
    "recommended_blend_mean": float(blend_rec.mean()),
    "all_blends": results,
    "n_test": 262648
}
with open(os.path.join(OUT, "fulldata_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved: {os.path.join(OUT, 'fulldata_results.json')}")
print("\nDONE")
