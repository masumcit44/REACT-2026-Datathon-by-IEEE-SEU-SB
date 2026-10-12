"""V7 robustness check on the final submission: distribution, alignment, and how its
ranking relates to the member pool and the legacy best blend."""
import os

import numpy as np
import pandas as pd

import v7_common as C


def csv_pred(path):
    return pd.read_csv(path)["fraud"].to_numpy(dtype=np.float64)


def main():
    pool = C.load_legacy_pool(verbose=False)
    for m in ("hgb183", "lgb183", "graphlgb", "xgb183"):
        paths = [os.path.join(C.OUT, f"{m}_{k}.npy") for k in ("fold2", "fold3", "test")]
        if all(os.path.exists(p) for p in paths):
            pool[m] = {"fold2": C.load_pred(paths[0]), "fold3": C.load_pred(paths[1]),
                       "test": C.load_pred(paths[2])}

    final = csv_pred(os.path.join(C.ROOT, "submission_file", "v7_final_submission.csv"))
    core = csv_pred(os.path.join(C.ROOT, "submission_file", "v7_stack_core_submission.csv"))
    legacy = C.load_pred(os.path.join(C.ROOT, "submission_file",
                                      "xgb_hgb600_blend_test_pred.npy"))

    sub = pd.read_csv(os.path.join(C.ROOT, "submission_file", "v7_final_submission.csv"))
    ss = pd.read_csv(os.path.join(C.ROOT, "sample_submission.csv"))
    print(f"rows={len(sub)} mean={final.mean():.6f} min={final.min():.6f} max={final.max():.6f}")
    print("quantiles:", {q: round(float(np.quantile(final, float(q))), 5)
                         for q in ("0.5", "0.9", "0.99", "0.999")})
    print(f"unique values={len(np.unique(final))} "
          f"top-1% rows={int((final >= np.quantile(final, 0.99)).sum())}")
    print("id order matches sample:",
          bool((sub["transaction_id"].to_numpy() == ss["transaction_id"].to_numpy()).all()))

    items = {n: d["test"] for n, d in pool.items()}
    items["FINAL_stack"] = final
    items["CORE_nnls"] = core
    items["LEGACY_blend"] = legacy
    names = sorted(items)
    M = pd.DataFrame({n: items[n] for n in names})
    sp, pe = M.corr(method="spearman"), M.corr(method="pearson")
    print("\nSpearman vs FINAL_STACK:")
    print(sp["FINAL_stack"].round(3).to_string())
    print("\nPearson vs FINAL_STACK:")
    print(pe["FINAL_stack"].round(3).to_string())
    mm = [n for n in names if n not in ("FINAL_stack", "CORE_nnls", "LEGACY_blend")]
    vals = [float(sp.loc[a, b]) for i, a in enumerate(mm) for b in mm[i + 1:]]
    print(f"\nmember-member Spearman: median={np.median(vals):.3f} "
          f"min={np.min(vals):.3f} max={np.max(vals):.3f}")


if __name__ == "__main__":
    main()