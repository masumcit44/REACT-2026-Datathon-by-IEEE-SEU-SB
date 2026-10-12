"""V9 STEP 0 - freeze + verify incumbent V7 champion (READ-ONLY).

Verifies without rewriting:
  - champion test/fold3 prediction hashes vs v8 champion_backup manifest
  - exact test reproduction (v8_out/v7_exact_reproduction.json)
  - sample submission ID alignment
  - V7 stack member order + meta feature order
Writes only to v9_out/v9_freeze_check.json. Stops (exit 1) on any failure.
"""
import os, sys, json, hashlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
OUT = os.path.join(ROOT, "v9_out")
os.makedirs(OUT, exist_ok=True)
MAN = os.path.join(ROOT, "v8_out", "champion_backup_manifest.csv")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()

def main():
    man = pd.read_csv(MAN)
    mh = {os.path.basename(r["backup_path"]): r["sha256"] for _, r in man.iterrows()}
    checks = {}
    ok = True
    # 1. champion test prediction hash (submission csv bytes)
    sub = os.path.join(ROOT, "submission_file", "v7_final_submission.csv")
    checks["test_csv_hash_match"] = sha(sub) == mh.get("v7_final_submission.csv")
    # 2. champion fold3 prediction hash (hgb183_fold3 as sentinel + all 4 members)
    for m in ["graphlgb", "hgb183", "lgb183", "xgb183"]:
        for k in ["fold2", "fold3", "test"]:
            fn = f"{m}_{k}.npy"
            checks[f"hash_{fn}"] = sha(os.path.join(ROOT, "v7_out", fn)) == mh.get(fn)
    # 3. exact test reproduction record
    repro = json.load(open(os.path.join(ROOT, "v8_out", "v7_exact_reproduction.json")))
    checks["repro_lofo_f3"] = bool(repro["lofo_fold3_match"])
    checks["repro_lofo_f2"] = bool(repro["lofo_fold2_match"])
    checks["repro_test_exact"] = bool(repro["reproduced_test_exact_match"])
    # 4. sample submission ID alignment
    ss = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    df = pd.read_csv(sub)
    checks["id_alignment"] = bool((df["transaction_id"].to_numpy() == ss["transaction_id"].to_numpy()).all())
    checks["shape"] = [int(df.shape[0]), int(df.shape[1])]
    checks["mean"] = float(df["fraud"].mean())
    # 5. V7 stack member order (selection file canonical order)
    sel = json.load(open(os.path.join(ROOT, "v7_out", "v7_selection.json")))
    checks["member_order"] = list(sel["winner_members"])
    checks["member_order_ok"] = checks["member_order"] == ["graphlgb", "hgb183", "lgb183", "xgb183"]
    # 6. V7 stack feature order (full meta layout block order)
    names = checks["member_order"]
    checks["meta_feature_order"] = C.meta_feature_names(names)
    checks["meta_feature_order_ok"] = (checks["meta_feature_order"][:4] == [f"{n}_prob" for n in names]
        and len(checks["meta_feature_order"]) == 3 * 4 + 5 + 4
        and checks["meta_feature_order"][12:17] == ["logit_mean", "logit_std", "logit_min", "logit_max", "logit_spread"])
    bad = [k for k, v in checks.items() if v is False]
    checks["all_pass"] = len(bad) == 0
    C.jdump(checks, os.path.join(OUT, "v9_freeze_check.json"))
    print("[v9-freeze] " + ("PASS" if checks["all_pass"] else f"FAIL: {bad}"), flush=True)
    print(f"[v9-freeze] mean={checks['mean']:.6f} shape={checks['shape']}", flush=True)
    if not checks["all_pass"]:
        sys.exit(1)
    print("DONE_FREEZE", flush=True)

if __name__ == "__main__":
    main()
