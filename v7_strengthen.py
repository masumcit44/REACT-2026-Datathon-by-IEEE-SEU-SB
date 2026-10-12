"""V7 member strengthening pass.

Evidence from the first pass: the internal early-stopping slice (rows 553012:613012) peaks
very early for the gradient-boosted LGB/XGB members (25/39/145 trees), which leaves those
members noticeably below their documented counterparts (lgb183 0.5125 vs 0.5148-0.5159,
xgb183 below the recorded 0.523993). HGB183 is unaffected (fixed 600 by recipe).

Fix: retrain the affected members with A-PRIORI tree counts taken from documented prior
evidence (experiment_tracker.csv / best_models.md), never from a validation fold:
    lgb183 / graphlgb : 600   (champion LGB recipe: lr=0.05, leaves=63, n_est=600)
    xgb183            : 500   (recorded best_iter=408 at eta=0.05 + the 100-round
                               early-stopping slack that run used)
hgb183 is left untouched (already fixed at 600).

This is not hyperparameter tuning on a validation fold: the counts are fixed in advance
from prior documented runs, and the same count is used for fold-2, fold-3 and the full fit,
which also keeps the test predictions on the same scale as the fold OOF predictions.
"""
import os
import json
import shutil
import subprocess
import sys

import v7_common as C

FIXED = {"lgb183": 600, "graphlgb": 600, "xgb183": 500}
# documented fold-3 references for the same recipes (experiment_tracker.csv)
REF_F3 = {"lgb183": 0.514834, "graphlgb": 0.516014, "xgb183": 0.523993}
MIN_DEFICIT = 0.001          # only retrain members clearly below their documented score


def read_meta(member):
    p = os.path.join(C.OUT, f"{member}_meta.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def backup(member):
    """Keep the first-pass (internal-ES) artifacts for transparent comparison."""
    bdir = os.path.join(C.OUT, "es_backup")
    os.makedirs(bdir, exist_ok=True)
    for kind in ("fold2", "fold3", "test", "meta"):
        ext = "json" if kind == "meta" else "npy"
        src = os.path.join(C.OUT, f"{member}_{kind}.{ext}")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(bdir, f"{member}_{kind}.{ext}"))


def main():
    targets = []
    print("[strengthen] member            f3_ap     documented   deficit   action")
    for m, rounds in FIXED.items():
        d = read_meta(m)
        if d is None:
            print(f"[strengthen] {m:12s} missing meta -> skip")
            continue
        deficit = REF_F3[m] - d["fold3_ap"]
        act = f"RETRAIN fixed={rounds}" if deficit > MIN_DEFICIT else "keep"
        if deficit > MIN_DEFICIT:
            targets.append((m, rounds))
        print(f"[strengthen] {m:12s} {d['fold3_ap']:.6f}  {REF_F3[m]:.6f}   "
              f"{deficit:+.6f}  {act}  (iters {d['fold2_best_iter']}/{d['fold3_best_iter']})",
              flush=True)

    if not targets:
        print("[strengthen] nothing to do", flush=True)
        print("DONE_STRENGTHEN", flush=True)
        return

    for m, _ in targets:
        backup(m)
        print(f"[strengthen] backed up first-pass artifacts for {m} -> v7_out/es_backup/",
              flush=True)

    members = [m for m, _ in targets]
    for m, rounds in targets:
        cmd = [sys.executable, "v7_members.py", "--members", m, "--force",
               "--skip-variant-screen", "--fixed-rounds", str(rounds)]
        print(f"[strengthen] $ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=C.ROOT).returncode
        if rc != 0:
            print(f"[strengthen] !! {m} retrain failed rc={rc}", flush=True)
    print("[strengthen] retrain finished", flush=True)

    for m, _ in targets:
        d = read_meta(m)
        print(f"[strengthen] {m:12s} new f2={d['fold2_ap']:.6f} f3={d['fold3_ap']:.6f} "
              f"protocol={d.get('iteration_protocol')}", flush=True)
    print("DONE_STRENGTHEN", flush=True)


if __name__ == "__main__":
    main()
