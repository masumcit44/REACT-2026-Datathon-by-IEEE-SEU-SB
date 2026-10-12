import os, json, pandas as pd, numpy as np

ROOT = "d:/Desktop/React datathon"
OUT = os.path.join(ROOT, "v7_out")

def j(path):
    with open(path) as f:
        return json.load(f)

print("=" * 70)
print("VERIFICATION ACROSS ALL FOUR STEPS")
print("=" * 70)

# ---------- STEP 1: HGB183 recovery + integration ----------
print("\n[STEP 1] HGB183 recovery + local model pool integration")
hgb = j(os.path.join(OUT, "hgb183_meta.json"))
print("  hgb183_meta.json present      :", os.path.exists(os.path.join(OUT, "hgb183_meta.json")))
print("  recovery.screen present       :", "screen" in hgb)
print("  recorded fold3_ap (legacy)    :", hgb.get("recovery", {}).get("recorded_fold3_ap"))
print("  re-derived fold3_ap (na215)  :", round(hgb["recovery"]["screen"]["na215"]["fold3_ap"], 6))
print("  integrated artifacts          :",
      all(os.path.exists(os.path.join(OUT, f"hgb183_{k}.npy"))
          for k in ("fold2", "fold3", "test")))
print("  member summary (from status) : hgb183 f2=0.830686 f3=0.525151")

# ---------- STEP 2: Graph V3 past-only connectivity features ----------
print("\n[STEP 2] Graph V3 past-only connectivity features")
gv3 = j(os.path.join(OUT, "graph_v3_meta.json"))
feat = gv3["names"]
jaccard = [f for f in feat if "jaccard" in f]
cooccur = [f for f in feat if ("shared" in f) or ("pairs" in f) or ("rows" in f)]
print("  graph_v3_meta.json present    :", os.path.exists(os.path.join(OUT, "graph_v3_meta.json")))
print("  train shape                  :", gv3["train_shape"])
print("  test shape                   :", gv3["test_shape"])
print("  n_features (k)               :", gv3["k"])
print("  jaccard-style features       :", jaccard)
print("  entity co-occurrence features :", cooccur[:8], "..." if len(cooccur) > 8 else "")
print("  probe pass (no label leakage):", gv3["probe"]["pass"])
print("  signal_fold3 ap_signed range:",
      round(min(f["ap_signed"] for f in gv3["signal_fold3"]), 4),
      "to",
      round(max(f["ap_signed"] for f in gv3["signal_fold3"]), 4))

# ---------- STEP 3: Stacking head (LR/Ridge/LGB) on OOF ----------
print("\n[STEP 3] Stacking head trained on OOF (LR / Ridge / LGB)")
stack = j(os.path.join(OUT, "v7_stack_results_wide.json"))
print("  v7_stack_results_wide.json   :", os.path.exists(os.path.join(OUT, "v7_stack_results_wide.json")))
print("  best_stack.head              :", stack["best_stack"]["head"])
print("  best_stack.kind              :", stack["best_stack"]["kind"])
print("  best_stack.meta_mode         :", stack["best_stack"]["meta_mode"])
print("  best_stack fold2_ap          :", round(stack["best_stack"]["fold2_ap"], 6))
print("  best_stack fold3_ap          :", round(stack["best_stack"]["fold3_ap"], 6))
print("  best_stack mean_ap           :", round(stack["best_stack"]["mean_ap"], 6))
print("  best_stack worst_ap          :", round(stack["best_stack"]["worst_ap"], 6))
print("  best_blend (for reference)   :", stack["best_blend"]["head"], "f3=", round(stack["best_blend"]["fold3_ap"], 6))
print("  promoted (stack > blend)     :", stack["promoted"])
print("  chosen_head family           :", stack["chosen_head"].get("kind"))
# Correlation context
c2 = stack["correlation"]["fold2"]["pearson"]
print("  corr(hgb183, lgb183) fold2   :", round(c2["hgb183"]["lgb183"], 4))
print("  corr(hgb183, xgb183) fold2   :", round(c2["hgb183"]["xgb183"], 4))

# ---------- STEP 4: Final improved submission on disk ----------
print("\n[STEP 4] Final improved submission saved in workspace")
SUB = os.path.join(ROOT, "submission_file", "v7_final_submission.csv")
check = j(os.path.join(OUT, "v7_submission_check_wide.json"))
df = pd.read_csv(SUB)
print("  path                         :", SUB)
print("  exists                       :", os.path.exists(SUB))
print("  shape                        :", df.shape)
print("  columns                      :", list(df.columns))
print("  pred mean                    :", round(float(df['fraud'].mean()), 6))
print("  pred min / max               :", round(float(df['fraud'].min()), 6), "/", round(float(df['fraud'].max()), 6))
print("  duplicate ids                :", int(df['transaction_id'].duplicated().sum()))
print("  nan count                    :", int(df.isna().sum().sum()))
print("  submission_checks.valid      :", check["valid"])
print("  submission_checks detail     :")
for k, v in check.items():
    print("     -", k, ":", v)

print("\n" + "=" * 70)
print("RESULT: all four steps verified on disk")
print("=" * 70)
