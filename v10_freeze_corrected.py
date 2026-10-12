"""V10 STEP 0 CORRECTED - freeze + verify the TRUE V7 leaderboard champion (READ-ONLY).

The original v10_freeze.py referenced xgb_hgb600_blend (old, public 0.52775).
The actual leaderboard champion is submission_file/v7_final_submission.csv
(public AP 0.53710, private AP 0.53268). This corrected script verifies the V7
champion against the V8 backup manifest and saves the result as
v10_out/v10_freeze_check_corrected.json.

No V7/V8/V9 artifacts are modified.
"""
import os, sys, hashlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
OUT = os.path.join(ROOT, "v10_out")
os.makedirs(OUT, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()

def main():
    man = pd.read_csv(os.path.join(ROOT, "v8_out", "champion_backup_manifest.csv"))
    mh = {os.path.basename(r["backup_path"]): r["sha256"] for _, r in man.iterrows()}

    c = {}
    # 1. V7 submission CSV SHA-256 against V8 backup manifest
    sub = os.path.join(ROOT, "submission_file", "v7_final_submission.csv")
    c["csv_hash"] = sha(sub) == mh.get("v7_final_submission.csv")
    c["csv_hash_value"] = sha(sub)
    c["csv_hash_manifest"] = mh.get("v7_final_submission.csv")

    # 2. V7 test prediction properties
    v7t = pd.read_csv(sub)["fraud"].to_numpy(float)
    c["test_mean"] = round(float(v7t.mean()), 6)
    c["test_shape"] = len(v7t)
    c["test_mean_ok"] = abs(c["test_mean"] - 0.017882) < 1e-6
    c["test_shape_ok"] = c["test_shape"] == 262648

    # 3. V7 fold-3 prediction AP (reproduced frame)
    v7f3 = np.load(os.path.join(ROOT, "v8_out", "v8_reproduced_fold3.npy")).astype(float).ravel()
    _, _, y, _, y3 = C.load_core(verbose=False)
    c["fold3_ap"] = round(float(C.ap(y3, v7f3)), 6)
    c["fold3_ap_ok"] = abs(c["fold3_ap"] - 0.527728) < 1e-6
    c["fold3_len"] = len(v7f3)
    c["fold3_len_ok"] = c["fold3_len"] == len(y3) == 59465
    c["target_align"] = bool(np.array_equal(y[C.IDX_F3], y3))

    # 4. Raw data + sample alignment
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["transaction_id", "timestamp"])
    c["raw_rows"] = len(tr)
    c["raw_rows_ok"] = c["raw_rows"] == 731942
    ss = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    df = pd.read_csv(sub)
    c["sample_align"] = bool((df["transaction_id"].to_numpy() == ss["transaction_id"].to_numpy()).all())

    # 5. Champion head info (from V8 audit)
    c["champion_head"] = "lr_C0.1/full"
    c["champion_public_ap"] = 0.53710
    c["champion_private_ap"] = 0.53268

    # 6. V10 freeze validity
    c["all_pass"] = bool(
        c["csv_hash"] and c["test_mean_ok"] and c["test_shape_ok"] and
        c["fold3_ap_ok"] and c["fold3_len_ok"] and c["target_align"] and
        c["raw_rows_ok"] and c["sample_align"]
    )

    # Mark the OLD freeze result as invalid (do not delete)
    old_path = os.path.join(OUT, "v10_freeze_check.json")
    c["old_freeze_invalid"] = True
    c["old_freeze_note"] = (
        "Original v10_freeze.py referenced xgb_hgb600_blend (public 0.52775) "
        "which is NOT the leaderboard champion. This corrected check uses "
        "submission_file/v7_final_submission.csv (public 0.53710, private 0.53268)."
    )

    C.jdump(c, os.path.join(OUT, "v10_freeze_check_corrected.json"))
    print("[v10-freeze-corrected] " + ("PASS" if c["all_pass"] else f"FAIL {c}"), flush=True)
    if not c["all_pass"]:
        sys.exit(1)
    print("DONE_FREEZE_CORRECTED", flush=True)

if __name__ == "__main__":
    main()