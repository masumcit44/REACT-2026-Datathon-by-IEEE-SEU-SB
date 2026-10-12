"""V7 status snapshot - one-line-per-member progress from v7_out meta files."""
import os
import json

import v7_common as C

MEMBERS = ["hgb183", "lgb183", "graphlgb", "xgb183"]
KINDS = ("fold2", "fold3", "test")


def main():
    print(f"{'member':10s} {'status':8s} {'f2_ap':>8s} {'f3_ap':>8s} {'f2_it':>6s} {'f3_it':>6s} "
          f"{'rounds':>6s} {'test_mean':>9s} {'sec':>5s}")
    for m in MEMBERS:
        meta_path = os.path.join(C.OUT, f"{m}_meta.json")
        files = [os.path.exists(os.path.join(C.OUT, f"{m}_{k}.npy")) for k in KINDS]
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            print(f"{m:10s} {'done':8s} {d['fold2_ap']:8.6f} {d['fold3_ap']:8.6f} "
                  f"{d['fold2_best_iter']:6d} {d['fold3_best_iter']:6d} "
                  f"{d['full_rounds']:6d} {d['test_mean']:9.6f} "
                  f"{d['fold2_seconds'] + d['fold3_seconds'] + d['full_seconds']:5.0f}")
        else:
            print(f"{m:10s} {sum(files)}/3      -        -        -      -      -         -     -")
    sel = os.path.join(C.OUT, "v7_selection.json")
    if os.path.exists(sel):
        with open(sel, "r", encoding="utf-8") as f:
            s = json.load(f)
        print(f"\n[status] SELECTED tag={s.get('winner_tag')} pool={s.get('winner_pool')} "
              f"candidate={s.get('winner_candidate', {}).get('head')} "
              f"f3={s.get('winner_candidate', {}).get('fold3_ap')}")
        print(f"[status] final submission: {s.get('final_submission')} "
              f"valid={s.get('final_submission_checks', {}).get('valid')}")
    else:
        print("\n[status] no pool selection yet")


if __name__ == "__main__":
    main()