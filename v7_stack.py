"""V7 STEP 3/4 - STACKING HEAD and FINAL SUBMISSION.

Goal: break the ~0.97 member-correlation ceiling that makes weighted averaging saturate
(RESEARCH_DIRECTOR_REPORT.md TASK 6). A learned head can down-weight the shared gradient
component and exploit the diverse tail.

Protocol (honest, no fold leakage):
  * Meta-features per row: member probabilities, logits, within-split ranks, spread and
    disagreement statistics (v7_common.member_matrix).
  * Leave-one-fold-out: fit the head on fold-2 OOF (labels of fold-2) -> score fold-3;
    fit on fold-3 OOF -> score fold-2. Both directions are reported.
  * Final head: refit on fold-2 + fold-3 OOF rows, applied to the test-time meta matrix
    built from the members' full-data test predictions.
  * The internal ES slice never touches either validation fold, and the members used here
    were regenerated coherently (v7_members.py) or are documented legacy artifacts.

Promotion rule (project convention adapted to two folds):
  promote the stack only if BOTH hold
      stack.fold3  >  best_blend.fold3
      stack.fold2  >= best_blend.fold2 - 0.001
  otherwise keep the (gated) weighted blend.
"""
import os
import sys
import json
import time
import argparse
import itertools

import numpy as np
import pandas as pd

import v7_common as C

try:
    from scipy.optimize import nnls as _nnls
except Exception:                                    # pragma: no cover
    _nnls = None
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -50, 50)))


POOLS = {
    "core": ["xgb183", "hgb183", "lgb183", "graphlgb"],
    "core_plus": ["xgb183", "hgb183", "lgb183", "graphlgb", "xgb183_gate", "megacb_gate"],
    "wide": ["xgb183", "hgb183", "lgb183", "graphlgb", "xgb183_gate", "hgb600_gate",
             "megalgb_gate", "megacb_gate"],
}


def load_pool(names, verbose=True):
    """Assemble a member pool: regenerated members from v7_out + documented legacy ones."""
    pool = {}
    legacy = C.load_legacy_pool(verbose=False)
    for n in names:
        if n in legacy:
            pool[n] = {"fold2": legacy[n]["fold2"], "fold3": legacy[n]["fold3"],
                       "test": legacy[n]["test"], "source": "legacy"}
            continue
        paths = [os.path.join(C.OUT, f"{n}_{k}.npy") for k in ("fold2", "fold3", "test")]
        if not all(os.path.exists(p) for p in paths):
            if verbose:
                print(f"[stack] SKIP {n} (missing v7 artifacts: "
                      f"{[os.path.basename(p) for p in paths if not os.path.exists(p)]})",
                      flush=True)
            continue
        f2, f3, te = (C.load_pred(paths[0]), C.load_pred(paths[1]), C.load_pred(paths[2]))
        pool[n] = {"fold2": f2, "fold3": f3, "test": te, "source": "v7"}
    if verbose:
        print(f"[stack] pool ({len(pool)}): " + ", ".join(sorted(pool)), flush=True)
    return pool


class Head:
    """Meta-model wrapper: fit(X, y) -> predict(X) -> score (monotone scale)."""

    def __init__(self, kind, meta_mode="full", **kw):
        self.kind = kind
        self.meta_mode = meta_mode
        self.kw = kw
        self.scaler = None
        self.model = None
        self.weights = None

    def fit(self, X, y):
        if self.kind == "lr":
            self.scaler = StandardScaler().fit(X)
            self.model = LogisticRegression(C=self.kw.get("C", 1.0), max_iter=4000,
                                            solver="lbfgs", tol=1e-6)
            self.model.fit(self.scaler.transform(X), y)
        elif self.kind == "ridge":
            self.scaler = StandardScaler().fit(X)
            self.model = Ridge(alpha=self.kw.get("alpha", 1.0))
            self.model.fit(self.scaler.transform(X), y.astype(np.float64))
        elif self.kind == "lgb":
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=self.kw.get("n_estimators", 300),
                learning_rate=self.kw.get("learning_rate", 0.02),
                num_leaves=self.kw.get("num_leaves", 7),
                max_depth=self.kw.get("max_depth", 3),
                min_child_samples=self.kw.get("min_child_samples", 200),
                subsample=self.kw.get("subsample", 0.8), subsample_freq=1,
                colsample_bytree=self.kw.get("colsample_bytree", 0.8),
                reg_lambda=self.kw.get("reg_lambda", 5.0), random_state=4242,
                verbose=-1, n_jobs=8)
            self.model.fit(X, y)
        elif self.kind == "nnls":
            A = np.asarray(X, dtype=np.float64)
            w, _ = _nnls(A, np.asarray(y, dtype=np.float64))
            s = w.sum()
            self.weights = w / s if s > 0 else w
        elif self.kind == "equal":
            self.weights = np.ones(X.shape[1]) / X.shape[1]
        else:
            raise ValueError(self.kind)
        return self

    def predict(self, X):
        if self.kind in ("nnls", "equal"):
            return np.asarray(X, dtype=np.float64) @ self.weights
        if self.kind == "lgb":
            return self.model.predict_proba(X)[:, 1]
        if self.kind == "lr":
            return self.model.predict_proba(self.scaler.transform(X))[:, 1]
        return self.model.predict(self.scaler.transform(X))

    def describe(self):
        if self.kind in ("nnls", "equal"):
            return {"kind": self.kind, "weights": np.round(self.weights, 6).tolist()}
        if self.kind == "lr":
            return {"kind": self.kind, "C": self.kw.get("C", 1.0),
                    "coef": np.round(self.model.coef_.ravel(), 6).tolist(),
                    "intercept": float(self.model.intercept_[0])}
        if self.kind == "ridge":
            return {"kind": self.kind, "alpha": self.kw.get("alpha", 1.0),
                    "coef": np.round(self.model.coef_.ravel(), 6).tolist(),
                    "intercept": float(self.model.intercept_)}
        return {"kind": self.kind, **{k: v for k, v in self.kw.items()}}

    def name(self):
        return self.kind + "".join(f"_{k}{v}" for k, v in self.kw.items())


def is_probabilistic(head):
    # nnls/equal heads output convex combinations of member probabilities, i.e.
    # already-calibrated probabilities: they must pass through unchanged.
    # Only ridge emits unbounded raw regression scores needing expit.
    return head.kind in ("lr", "lgb", "nnls", "equal")


def to_prob(scores, head):
    s = np.asarray(scores, dtype=np.float64)
    return s if is_probabilistic(head) else expit(s)


def lofo(pool, names, y2, y3, head, verbose=True):
    """Leave-one-fold-out evaluation: fit on one fold's OOF rows, score the other fold."""
    Ma = C.member_matrix(pool, "fold2", names, head.meta_mode)
    Mb = C.member_matrix(pool, "fold3", names, head.meta_mode)
    t0 = time.time()
    h_a = Head(head.kind, head.meta_mode, **head.kw).fit(Ma, y2)   # fit fold-2
    ap_b = C.ap(y3, h_a.predict(Mb))                               # score fold-3
    h_b = Head(head.kind, head.meta_mode, **head.kw).fit(Mb, y3)   # fit fold-3
    ap_a = C.ap(y2, h_b.predict(Ma))                               # score fold-2
    res = {"candidate": head.name(), "head": head.name(), "kind": head.kind,
           "meta_mode": head.meta_mode,
           "n_members": len(names), "fold2_ap": ap_a, "fold3_ap": ap_b,
           "mean_ap": (ap_a + ap_b) / 2.0, "worst_ap": min(ap_a, ap_b),
           "f3_minus_f2": ap_b - ap_a, "seconds": time.time() - t0}
    if verbose:
        print(f"[stack] {head.name():34s} {head.meta_mode:5s} f2={ap_a:.6f} f3={ap_b:.6f} "
              f"mean={(ap_a + ap_b) / 2:.6f} {res['seconds']:.1f}s", flush=True)
    return res, (h_a, h_b)


def member_table(pool, names, y2, y3):
    rows = []
    for n in names:
        a, b = C.ap(y2, pool[n]["fold2"]), C.ap(y3, pool[n]["fold3"])
        rows.append({"candidate": f"member:{n}", "kind": "member", "meta_mode": "raw",
                     "fold2_ap": a, "fold3_ap": b, "mean_ap": (a + b) / 2,
                     "worst_ap": min(a, b), "source": pool[n].get("source", "?")})
    rows.sort(key=lambda r: -r["mean_ap"])
    return rows


def legacy_reference(y2, y3):
    """The documented 0.527088 blend evaluated on the same two folds (context only)."""
    try:
        p2 = C.load_pred(os.path.join(C.ROOT, "xgb_hgb600_blend_fold2_pred.npy"))
        p3 = C.load_pred(os.path.join(C.ROOT, "xgb_hgb600_blend_fold3_pred.npy"))
        a, b = C.ap(y2, p2), C.ap(y3, p3)
        return {"candidate": "legacy:xgb_hgb600_blend_0.527088", "kind": "legacy",
                "meta_mode": "weights", "fold2_ap": a, "fold3_ap": b,
                "mean_ap": (a + b) / 2, "worst_ap": min(a, b)}
    except Exception as e:                                   # pragma: no cover
        return {"candidate": "legacy:xgb_hgb600_blend", "kind": "legacy", "error": str(e)}


def candidate_heads():
    """Blend baselines + stack heads (logistic, ridge, LightGBM-on-meta)."""
    return [
        Head("equal", "prob"),
        Head("equal", "rank"),
        Head("nnls", "prob"),
        Head("lr", "prob", C=1.0),
        Head("lr", "full", C=1.0),
        Head("lr", "full", C=0.1),
        Head("lr", "full", C=10.0),
        Head("ridge", "prob", alpha=1.0),
        Head("ridge", "full", alpha=1.0),
        Head("ridge", "full", alpha=0.3),
        Head("lgb", "prob", n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3),
        Head("lgb", "full", n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3),
        Head("lgb", "full", n_estimators=600, learning_rate=0.01, num_leaves=4, max_depth=2),
        Head("lgb", "full", n_estimators=600, learning_rate=0.01, num_leaves=15, max_depth=4),
        Head("lgb", "full", n_estimators=150, learning_rate=0.05, num_leaves=7, max_depth=3),
        Head("ridge", "full", alpha=0.1),
    ]


def find_head(heads, res):
    for h in heads:
        if h.name() == res["head"] and h.meta_mode == res["meta_mode"]:
            return h
    raise KeyError(res)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--pool", default="wide", choices=sorted(POOLS))
    ap_.add_argument("--tol", type=float, default=0.001)
    ap_.add_argument("--tag", default="run",
                     help="artifact tag so several pools can be evaluated without clobbering")
    ap_.add_argument("--submission", default=None)
    ap_.add_argument("--alias", default=None)
    args = ap_.parse_args()
    if args.submission is None:
        args.submission = os.path.join(C.ROOT, "submission_file",
                                       f"v7_stack_{args.tag}_submission.csv")
    if args.alias is None:
        args.alias = os.path.join(C.ROOT, f"submission_v7_{args.tag}.csv")
    t00 = time.time()

    X, Xte, y, y2, y3 = C.load_core(verbose=False)
    pool = load_pool(POOLS[args.pool])
    names = sorted(pool)

    # ------- correlation evidence (the 0.97 bottleneck) -------
    corr = {tag: C.pair_corr_report(pool, tag, names) for tag in ("fold2", "fold3", "test")}
    print(f"[stack] median |Spearman| fold3={corr['fold3']['median_abs_spearman']:.4f} "
          f"fold2={corr['fold2']['median_abs_spearman']:.4f}", flush=True)

    # ------- candidate table -------
    rows = member_table(pool, names, y2, y3)
    rows.append(legacy_reference(y2, y3))
    heads = candidate_heads()
    for h in heads:
        r, _ = lofo(pool, names, y2, y3, h)
        rows.append(r)

    blends = [r for r in rows if r["kind"] in ("nnls", "equal")]
    stacks = [r for r in rows if r["kind"] in ("lr", "ridge", "lgb")]
    best_blend = max(blends, key=lambda r: r["fold3_ap"])
    eligible = [r for r in stacks if r["fold2_ap"] >= best_blend["fold2_ap"] - args.tol]
    best_stack = max(eligible, key=lambda r: r["fold3_ap"]) if eligible else None
    promote = bool(best_stack and best_stack["fold3_ap"] > best_blend["fold3_ap"])
    chosen = best_stack if promote else best_blend
    print(f"[stack] best_blend={best_blend['head']}({best_blend['meta_mode']}) "
          f"f3={best_blend['fold3_ap']:.6f} | best_stack="
          f"{best_stack['head'] if best_stack else None}"
          f"({best_stack['meta_mode'] if best_stack else '-'}) "
          f"f3={best_stack['fold3_ap'] if best_stack else float('nan'):.6f} | "
          f"PROMOTE={promote}", flush=True)

    # ------- final head refit on both folds' OOF, applied to test meta -------
    head = find_head(heads, chosen)
    M2 = C.member_matrix(pool, "fold2", names, head.meta_mode)
    M3 = C.member_matrix(pool, "fold3", names, head.meta_mode)
    Mtr = np.vstack([M2, M3])
    ytr = np.concatenate([y2, y3])
    final = Head(head.kind, head.meta_mode, **head.kw).fit(Mtr, ytr)
    Mte = C.member_matrix(pool, "test", names, head.meta_mode)
    raw_scores = final.predict(Mte)
    prob = np.clip(to_prob(raw_scores, final), 0.0, 1.0)
    assert np.isfinite(prob).all() and len(prob) == C.N_TEST

    head_detail = final.describe()
    if head.kind == "lgb":
        imp = final.model.booster_.feature_importance(importance_type="gain")
        fn = C.meta_feature_names(names) if head.meta_mode == "full" else names
        n = len(names)

        def _family(i):
            if head.meta_mode != "full":
                return "prob"
            if i < n:
                return "prob"
            if i < 2 * n:
                return "logit"
            if i < 3 * n:
                return "rank"
            if i < 3 * n + 5:
                return "spread_stat"
            return "disagreement_z"

        head_detail["feature_importance_gain"] = {
            f"{fn[i]} [{_family(i)}]": float(imp[i]) for i in np.argsort(-imp)[:12]}
        head_detail["non_probability_gain_share"] = float(
            sum(float(imp[i]) for i in range(len(fn)) if _family(i) != "prob")
            / max(imp.sum(), 1e-9))
        fam = {}
        for i in range(len(fn)):
            fam[_family(i)] = fam.get(_family(i), 0.0) + float(imp[i])
        head_detail["gain_share_by_meta_family"] = {
            k: round(v / max(imp.sum(), 1e-9), 4) for k, v in fam.items()}

    # ------- submission -------
    checks = C.build_submission(prob, args.submission)
    import shutil
    shutil.copyfile(args.submission, args.alias)
    alias_checks = C.verify_submission(args.alias, prob)
    print(f"[stack] submission {args.submission} valid={checks['valid']} "
          f"mean={prob.mean():.6f}", flush=True)

    # ------- artifacts -------
    df = pd.DataFrame(rows).sort_values("fold3_ap", ascending=False)
    df.to_csv(os.path.join(C.OUT, f"v7_candidate_scores_{args.tag}.csv"), index=False)
    detail = {
        "pool_name": args.pool, "tag": args.tag, "members": names,
        "member_sources": {n: pool[n].get("source", "?") for n in names},
        "correlation": corr, "best_blend": best_blend, "best_stack": best_stack,
        "promoted": promote, "tolerance": args.tol,
        "chosen_head": head_detail, "chosen_candidate": chosen,
        "submission": args.submission, "alias": args.alias,
        "submission_checks": checks, "alias_checks": alias_checks,
        "test_prob_mean": float(prob.mean()), "test_prob_min": float(prob.min()),
        "test_prob_max": float(prob.max()), "seconds": time.time() - t00,
    }
    C.jdump(detail, os.path.join(C.OUT, f"v7_stack_results_{args.tag}.json"))
    C.jdump(checks, os.path.join(C.OUT, f"v7_submission_check_{args.tag}.json"))
    np.save(os.path.join(C.OUT, f"v7_final_test_scores_{args.tag}.npy"), prob)
    print(f"[stack] DONE in {time.time() - t00:.0f}s  promote={promote} "
          f"submission_valid={checks['valid']}", flush=True)
    print("DONE_STACK", flush=True)


if __name__ == "__main__":
    main()