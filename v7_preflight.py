"""V7 pre-flight: validate the submission writer/verifier and the meta-feature builders
before the final run, so the long pipeline cannot fail at the last step."""
import os

import numpy as np

import v7_common as C
import v7_stack as S


def main():
    # 1) submission writer + verifier on a known-good prediction vector
    pred = C.load_pred(os.path.join(C.ROOT, "submission_file",
                                    "xgb_hgb600_blend_test_pred.npy"))
    tmp = os.path.join(C.OUT, "_preflight_submission.csv")
    checks = C.build_submission(pred, tmp)
    print("[pre-flight] build_submission:", checks)
    assert checks["valid"], "submission writer/verifier FAILED"
    os.remove(tmp)

    # 2) meta-feature builders
    n = 1000
    rng = np.random.default_rng(0)
    pool = {"a": {"fold2": rng.random(n), "fold3": rng.random(n), "test": rng.random(n)},
            "b": {"fold2": rng.random(n), "fold3": rng.random(n), "test": rng.random(n)}}
    for mode in ("prob", "rank", "full"):
        M = C.member_matrix(pool, "fold2", ["a", "b"], mode)
        print(f"[pre-flight] member_matrix {mode}: {M.shape} finite={np.isfinite(M).all()}")
        assert np.isfinite(M).all()

    # 3) head fitting/prediction on synthetic meta data (all head kinds)
    y = (rng.random(n) < 0.05).astype(int)
    for h in S.candidate_heads():
        M = C.member_matrix(pool, "fold2", ["a", "b"], h.meta_mode)
        h.fit(M, y)
        h.predict(M)
        p = S.to_prob(h.predict(M), h)
        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    print("[pre-flight] all head kinds fit/predict OK")

    # 4) correlation report
    rep = C.pair_corr_report(pool, "fold3", ["a", "b"])
    print(f"[pre-flight] corr report median|spearman|={rep['median_abs_spearman']:.4f}")

    # 5) missing-member skip behaviour
    p2 = S.load_pool(["hgb183", "definitely_missing_member"], verbose=True)
    assert "definitely_missing_member" not in p2
    print("[pre-flight] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()