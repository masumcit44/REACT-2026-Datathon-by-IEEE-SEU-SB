"""V7 diagnostic: is member ranking consistency on TEST comparable to the folds?

If members rank transactions consistently on fold-2/fold-3 OOF but not on test, the
stacking head's fold-fitted weights cannot transfer, and the local gain is not real.
This compares pairwise Spearman per split and flags the biggest drops.
"""
import os

import numpy as np
import pandas as pd

import v7_common as C


def main():
    pool = C.load_legacy_pool(verbose=False)
    for m in ("hgb183", "lgb183", "graphlgb", "xgb183"):
        paths = [os.path.join(C.OUT, f"{m}_{k}.npy") for k in ("fold2", "fold3", "test")]
        if all(os.path.exists(p) for p in paths):
            pool[m] = {"fold2": C.load_pred(paths[0]), "fold3": C.load_pred(paths[1]),
                       "test": C.load_pred(paths[2])}
    names = sorted(pool)
    print("members:", names, flush=True)

    for tag in ("fold2", "fold3", "test"):
        M = pd.DataFrame({n: pool[n][tag] for n in names})
        S = M.corr(method="spearman").to_numpy()
        iu = np.triu_indices(len(names), 1)
        vals = S[iu]
        print(f"{tag:6s} median|spearman|={np.median(np.abs(vals)):.4f} "
              f"min={vals.min():.4f} max={vals.max():.4f}", flush=True)

    # biggest fold-3 -> test drops
    S3 = pd.DataFrame({n: pool[n]["fold3"] for n in names}).corr(method="spearman")
    ST = pd.DataFrame({n: pool[n]["test"] for n in names}).corr(method="spearman")
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append((a, b, float(S3.loc[a, b]), float(ST.loc[a, b]),
                         float(S3.loc[a, b] - ST.loc[a, b])))
    rows.sort(key=lambda r: -r[4])
    print("\nbiggest fold3->test Spearman drops:", flush=True)
    for a, b, s3, st, d in rows[:8]:
        print(f"  {a:14s} {b:14s} f3={s3:.3f} test={st:.3f} drop={d:+.3f}", flush=True)

    regenerated = [m for m in ("hgb183", "lgb183", "graphlgb", "xgb183") if m in pool]
    def _pair_med(names_, S):
        vals = [S.loc[a, b] for i, a in enumerate(names_) for b in names_[i + 1:]]
        return float(np.median(vals)) if vals else float("nan")
    C.jdump({
        "members": names,
        "median_abs_spearman": {t: _pair_med(names, pd.DataFrame({n: pool[n][t] for n in names})
                                              .corr(method="spearman"))
                                for t in ("fold2", "fold3", "test")},
        "regenerated_median_abs_spearman": {
            t: _pair_med(regenerated, pd.DataFrame({n: pool[n][t] for n in regenerated})
                         .corr(method="spearman")) for t in ("fold2", "fold3", "test")},
        "biggest_fold3_to_test_drops": [
            {"a": a, "b": b, "fold3": s3, "test": st, "drop": d} for a, b, s3, st, d in rows[:10]],
    }, os.path.join(C.OUT, "v7_consistency.json"))


if __name__ == "__main__":
    main()