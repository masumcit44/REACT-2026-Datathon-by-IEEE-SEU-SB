"""V7 pool selector - choose between the evaluated stack runs and promote the winner.

Two pools are evaluated:
    core : only V7-regenerated members -> fold OOF and test predictions come from ONE
           pipeline, so meta-weights transfer cleanly to test.
    wide : core + the documented legacy members -> maximum diversity, but the legacy
           members carry 1.3-1.5x fold->test scale drift (different feature frames).

Selection rule (all local evidence, no leaderboard):
    1. within a pool, the run already picked its own best candidate under the promotion rule
       (stack only if it beats the weighted blend on fold-3 while holding fold-2 within tol)
    2. across pools, take the higher fold-3 AP provided fold-2 stays within `--tol-cross`;
       if the two pools are within `--tie` on fold-3, prefer `core` (cleaner transfer).

Writes the canonical artifacts consumed by v7_report.py and the final submission paths:
    submission_file/v7_final_submission.csv  +  submission_v7_final.csv (workspace root)
"""
import os
import json
import shutil
import argparse

import numpy as np
import pandas as pd

import v7_common as C


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--tags", default="core,wide")
    ap_.add_argument("--tie", type=float, default=0.0005)
    ap_.add_argument("--tol-cross", type=float, default=0.001)
    ap_.add_argument("--coherence-margin", type=float, default=0.001,
                     help="extra fold-3 AP the legacy-member pool must beat the coherent "
                          "pool by to be chosen; compensates the measured fold->test "
                          "Spearman drift of legacy members (see v7_consistency.json)")
    args = ap_.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    runs = {}
    for t in tags:
        r = _load(os.path.join(C.OUT, f"v7_stack_results_{t}.json"))
        if r:
            ch = r.get("chosen_candidate", {})
            runs[t] = {"results": r, "chosen": ch,
                       "fold2_ap": ch.get("fold2_ap"), "fold3_ap": ch.get("fold3_ap"),
                       "members": r.get("members", []),
                       "promoted": r.get("promoted")}
            print(f"[select] {t:5s} pool={r.get('pool_name'):9s} "
                  f"chosen={ch.get('head')}({ch.get('meta_mode')}) "
                  f"f2={ch.get('fold2_ap', float('nan')):.6f} "
                  f"f3={ch.get('fold3_ap', float('nan')):.6f} "
                  f"promoted={r.get('promoted')}", flush=True)

    assert runs, "no stack runs found - run v7_stack.py first"

    ordered = sorted(runs.values(), key=lambda r: -r["fold3_ap"])
    winner = ordered[0]
    if len(ordered) > 1:
        a, b = ordered[0], ordered[1]           # a = higher fold-3
        coherent = {"core", "coh"}              # tags that use only V7-regenerated members
        if (a["fold3_ap"] - b["fold3_ap"]) <= args.tie and b["fold2_ap"] >= a["fold2_ap"]:
            winner = b                          # near-tie -> the fold-2-safer one
        elif a["fold2_ap"] < b["fold2_ap"] - args.tol_cross:
            winner = b                          # fold-2 guard across pools
        elif (b["results"]["tag"] in coherent and a["results"]["tag"] not in coherent
              and (a["fold3_ap"] - b["fold3_ap"]) <= args.coherence_margin):
            winner = b                          # legacy advantage too small vs transfer risk
    wtag = winner["results"]["tag"]

    path = winner["results"]["submission"]
    final_path = os.path.join(C.ROOT, "submission_file", "v7_final_submission.csv")
    root_alias = os.path.join(C.ROOT, "submission_v7_final.csv")
    shutil.copyfile(path, final_path)
    shutil.copyfile(path, root_alias)
    prob = C.load_pred(os.path.join(C.OUT, f"v7_final_test_scores_{wtag}.npy"))
    checks = C.verify_submission(final_path, prob)

    # canonical artifact names for v7_report.py
    shutil.copyfile(os.path.join(C.OUT, f"v7_stack_results_{wtag}.json"),
                    os.path.join(C.OUT, "v7_validation_results.json"))
    shutil.copyfile(os.path.join(C.OUT, f"v7_submission_check_{wtag}.json"),
                    os.path.join(C.OUT, "v7_submission_check.json"))
    shutil.copyfile(os.path.join(C.OUT, f"v7_candidate_scores_{wtag}.csv"),
                    os.path.join(C.OUT, "v7_candidate_scores.csv"))

    sel = {"tags_evaluated": tags,
           "per_pool": {t: {k: v for k, v in r.items() if k != "results"}
                        for t, r in runs.items()},
           "winner_tag": wtag, "winner_pool": winner["results"]["pool_name"],
           "winner_members": winner["members"],
           "winner_candidate": winner["chosen"],
           "winner_promoted_stack": winner["promoted"],
           "final_submission": final_path, "root_alias": root_alias,
           "final_submission_checks": checks,
           "tie_rule": args.tie, "cross_tolerance": args.tol_cross,
           "coherence_margin": args.coherence_margin}
    C.jdump(sel, os.path.join(C.OUT, "v7_selection.json"))
    print(f"[select] WINNER tag={wtag} pool={winner['results']['pool_name']} "
          f"f3={winner['fold3_ap']:.6f} f2={winner['fold2_ap']:.6f} valid={checks['valid']}",
          flush=True)
    print("DONE_SELECT", flush=True)


if __name__ == "__main__":
    main()