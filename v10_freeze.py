"""V10 STEP 0 - freeze + verify champion (READ-ONLY, no V7/V8/V9 writes)."""
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
    ok = True
    c = {}
    sub = os.path.join(ROOT, "submission_file", "v7_final_submission.csv")
    c["csv_hash"] = sha(sub) == mh.get("v7_final_submission.csv")
    v7t = pd.read_csv(sub)["fraud"].to_numpy(float)
    c["test_mean"] = round(float(v7t.mean()), 6) == 0.017882
    c["test_shape"] = len(v7t) == 262648
    v7f3 = np.load(os.path.join(ROOT, "v8_out", "v8_reproduced_fold3.npy")).astype(float).ravel()
    _, _, y, _, y3 = C.load_core(verbose=False)
    c["fold3_ap"] = abs(C.ap(y3, v7f3) - 0.527728) < 1e-6
    c["fold3_len"] = len(v7f3) == len(y3) == 59465
    c["target_align"] = bool(np.array_equal(y[C.IDX_F3], y3))
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["transaction_id", "timestamp"])
    c["raw_rows"] = len(tr) == 731942
    ss = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    df = pd.read_csv(sub)
    c["sample_align"] = bool((df["transaction_id"].to_numpy() == ss["transaction_id"].to_numpy()).all())
    c["all_pass"] = all(c.values())
    C.jdump({k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in c.items()}, os.path.join(OUT, "v10_freeze_check.json"))
    print("[v10-freeze] " + ("PASS" if c["all_pass"] else f"FAIL {c}"), flush=True)
    if not c["all_pass"]:
        sys.exit(1)
    print("DONE_FREEZE", flush=True)
if __name__ == "__main__":
    main()
