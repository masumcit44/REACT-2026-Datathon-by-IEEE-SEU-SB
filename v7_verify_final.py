import os, json
import pandas as pd
import numpy as np

ROOT = "d:/Desktop/React datathon"
SUB = os.path.join(ROOT, "submission_file", "v7_final_submission.csv")
CHECK = os.path.join(ROOT, "v7_out", "v7_submission_check_wide.json")
SAMPLE = os.path.join(ROOT, "sample_submission.csv")
STACK = os.path.join(ROOT, "v7_out", "v7_stack_results_wide.json")

print("=" * 60)
print("FINAL VERIFICATION — v7 final submission")
print("=" * 60)

# 1. Submission file
print("\n[1] FINAL SUBMISSION FILE")
if not os.path.exists(SUB):
    print("MISSING:", SUB)
    raise SystemExit(1)
df = pd.read_csv(SUB)
print("path      :", SUB)
print("shape     :", df.shape)
print("columns   :", list(df.columns))
print("id_min    :", df['transaction_id'].min())
print("id_max    :", df['transaction_id'].max())
print("pred_min  :", round(float(df['fraud'].min()), 6))
print("pred_max  :", round(float(df['fraud'].max()), 6))
print("pred_mean :", round(float(df['fraud'].mean()), 6))
print("nan_count :", int(df.isna().sum().sum()))
print("dup_ids   :", int(df['transaction_id'].duplicated().sum()))
print("range_0_1 :", bool((df['fraud'] >= 0).all() and (df['fraud'] <= 1).all()))

# 2. Submission checks JSON
print("\n[2] SUBMISSION CHECKS (v7_submission_check_wide.json)")
with open(CHECK) as f:
    check = json.load(f)
for k, v in check.items():
    tag = "OK" if v else "FAIL"
    print(f"  {tag:4s}  {k:22s} : {v}")
print("overall   :", "VALID" if check.get("valid") else "INVALID")

# 3. Compare with sample submission, if present
print("\n[3] SAMPLE SUBMISSION COMPARISON")
if os.path.exists(SAMPLE):
    sample = pd.read_csv(SAMPLE)
    print("sample_shape :", sample.shape)
    print("shape_match  :", df.shape == sample.shape)
    print("cols_match   :", list(df.columns) == list(sample.columns))
    print("ids_match    :", df['transaction_id'].equals(sample['transaction_id']))
    print("ids_equal_set:", set(df['transaction_id']) == set(sample['transaction_id']))
else:
    print("sample_submission.csv not found (may be fine)")

# 4. Stacking head summary
print("\n[4] STACKING HEAD RESULTS (v7_stack_results_wide.json)")
with open(STACK) as f:
    stack = json.load(f)
members = stack.get("members", [])
print("pool_name :", stack.get("pool_name"))
print("n_members :", len(members))
print("members   :", members)

cands = stack.get("candidates", [])
if cands:
    print("\nTop stacking candidates by fold3_ap:")
    for i, c in enumerate(sorted(cands, key=lambda x: float(x.get("fold3_ap", 0)), reverse=True)[:6], 1):
        print(f"  {i:2d}. {c.get('name')}")
        print(f"       fold2_ap={float(c.get('fold2_ap', 0)):<10.6f} fold3_ap={float(c.get('fold3_ap', 0)):<10.6f} mean={float(c.get('mean_ap', 0)):<10.6f} worst={float(c.get('worst_ap', 0)):<10.6f}")
        print(f"       head={c.get('head')} meta_mode={c.get('meta_mode')}")

# Correlation highlight
corr = stack.get("correlation", {}).get("fold2", {}).get("pearson", {})
print("\n[5] FOLD2 PEARSON CORRELATION (selected pairs)")
pairs = [
    ("hgb183", "lgb183"),
    ("hgb183", "graphlgb"),
    ("lgb183", "graphlgb"),
    ("hgb183", "xgb183"),
    ("graphlgb", "xgb183"),
]
for a, b in pairs:
    v = corr.get(a, {}).get(b, float('nan'))
    print(f"  corr({a:10s}, {b:10s}) = {v:.4f}")

print("\n" + "=" * 60)
print("STATUS: Final submission ready at", SUB)
print("=" * 60)
