"""V7 diagnostic: member prediction scale/consistency across fold-2, fold-3 and test.

A stacking head transfers weights learned on fold OOF rows to test-time meta-features.
If a member's test predictions are on a different scale than its fold predictions (which
happens when the fold variant and the test variant come from different feature frames),
that member transfers poorly. This prints the evidence without loading the parquet.
"""
import os

import numpy as np

import v7_common as C


def main():
    pool = C.load_legacy_pool(verbose=False)
    for m in ("hgb183", "lgb183", "graphlgb", "xgb183"):
        paths = [os.path.join(C.OUT, f"{m}_{k}.npy") for k in ("fold2", "fold3", "test")]
        if all(os.path.exists(p) for p in paths):
            pool[m] = {"fold2": C.load_pred(paths[0]), "fold3": C.load_pred(paths[1]),
                       "test": C.load_pred(paths[2])}

    print(f"{'member':16s} {'source':8s} {'f2_mean':>10s} {'f3_mean':>10s} {'test_mean':>10s} "
          f"{'test/f3':>8s} {'f3/f2':>7s}")
    for n in sorted(pool):
        d = pool[n]
        m2, m3, mt = d["fold2"].mean(), d["fold3"].mean(), d["test"].mean()
        print(f"{n:16s} {d.get('source', 'v7'):8s} {m2:10.6f} {m3:10.6f} {mt:10.6f} "
              f"{mt / max(m3, 1e-12):8.3f} {m3 / max(m2, 1e-12):7.3f}")

    print("\n[diag] fold-2/fold-3 label rates: "
          f"{np.load(os.path.join(C.JN, 'Fold_2_v4_targets.npy')).mean():.6f} / "
          f"{np.load(os.path.join(C.JN, 'Fold_3_v4_targets.npy')).mean():.6f}")


if __name__ == "__main__":
    main()