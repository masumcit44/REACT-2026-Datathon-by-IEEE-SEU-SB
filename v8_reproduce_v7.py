"""V8 exact reproduction of the V7 champion (core pool + lr_C0.1/full head).

Correct protocol (mirrors v7_stack.py):
  LOFO numbers in v7_stack_results_core.json are leave-one-fold-out:
    fold3_ap = fit head on fold-2 OOF -> score fold-3
    fold2_ap = fit head on fold-3 OOF -> score fold-2
  Final submission head = refit on fold-2+fold-3 OOF, applied to test meta.

This script verifies both the LOFO scores and the stored test vector exactly.
"""
import os, json, time
import numpy as np
import pandas as pd

ROOT = "d:/Desktop/React datathon"
OUT = os.path.join(ROOT, "v7_out")
AUDIT = os.path.join(ROOT, "v8_out")
os.makedirs(AUDIT, exist_ok=True)

import v7_common as C
from v7_stack import Head, to_prob, candidate_heads


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def ap_safe(y_true, y_score):
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def main():
    # V7 champion: CORE pool ['graphlgb', 'hgb183', 'lgb183', 'xgb183']
    # Head: lr_C0.1, meta_mode=full, fold3_ap=0.527728
    _, Xte, y, y2, y3 = C.load_core(verbose=False)
    log(f"labels: train={len(y)} fold2={len(y2)} fold3={len(y3)} test={len(Xte)}")

    v7_sub = pd.read_csv(os.path.join(ROOT, "submission_file", "v7_final_submission.csv"))
    v7_test = v7_sub["fraud"].to_numpy(dtype=np.float64)
    log(f"V7 submission: mean={v7_test.mean():.6f}")

    v7r = json.load(open(os.path.join(OUT, "v7_stack_results_core.json")))
    chosen = v7r["chosen_candidate"]
    names = v7r["members"]
    log(f"V7 champion: {chosen['head']} mode={chosen['meta_mode']} "
        f"f3={chosen['fold3_ap']:.6f} f2={chosen['fold2_ap']:.6f}")

    pool = {}
    for m in names:
        pool[m] = {"fold2": C.load_pred(os.path.join(OUT, f"{m}_fold2.npy")),
                   "fold3": C.load_pred(os.path.join(OUT, f"{m}_fold3.npy")),
                   "test": C.load_pred(os.path.join(OUT, f"{m}_test.npy")),
                   "source": "v7_core"}
        log(f"  {m}: f2={ap_safe(y2, pool[m]['fold2']):.6f} "
            f"f3={ap_safe(y3, pool[m]['fold3']):.6f}")

    heads = candidate_heads()
    th = None
    for h in heads:
        if h.name() == chosen["head"] and h.meta_mode == chosen["meta_mode"]:
            th = h
            break
    if th is None:
        log("ERROR: head not found")
        raise ValueError("head not found")

    M2 = C.member_matrix(pool, "fold2", names, chosen["meta_mode"])
    M3 = C.member_matrix(pool, "fold3", names, chosen["meta_mode"])

    # LOFO reproduction (same protocol as v7_stack.lofo):
    #   fit on fold-2 OOF -> score fold-3 ; fit on fold-3 OOF -> score fold-2
    h_on_2 = Head(th.kind, th.meta_mode, **th.kw).fit(M2, y2)
    h_on_3 = Head(th.kind, th.meta_mode, **th.kw).fit(M3, y3)
    lofo_f3 = ap_safe(y3, h_on_2.predict(M3))
    lofo_f2 = ap_safe(y2, h_on_3.predict(M2))
    log(f"LOFO reproduced fold3 AP: {lofo_f3:.6f} (target: {chosen['fold3_ap']:.6f})")
    log(f"LOFO reproduced fold2 AP: {lofo_f2:.6f} (target: {chosen['fold2_ap']:.6f})")

    # Final-head reproduction (same protocol as v7_stack final refit):
    #   refit on fold-2 + fold-3 OOF -> apply to test meta
    Mtr = np.vstack([M2, M3])
    ytr = np.concatenate([y2, y3])
    fh = Head(th.kind, th.meta_mode, **th.kw).fit(Mtr, ytr)
    Mte = C.member_matrix(pool, "test", names, chosen["meta_mode"])
    test_pred_raw = fh.predict(Mte)
    test_pred = np.clip(to_prob(test_pred_raw, fh), 0.0, 1.0)

    v7_stored = C.load_pred(os.path.join(OUT, "v7_final_test_scores_core.npy"))
    diff_stored = float(np.max(np.abs(test_pred - v7_stored)))
    diff_csv = float(np.max(np.abs(v7_stored - v7_test)))
    log(f"Stored core vs CSV diff: {diff_csv:.2e}")
    log(f"Repro vs stored diff: {diff_stored:.2e}")
    log(f"Test mean: repro={test_pred.mean():.6f} "
        f"stored={v7_stored.mean():.6f} csv={v7_test.mean():.6f}")

    np.save(os.path.join(AUDIT, "v8_reproduced_test.npy"), test_pred)
    np.save(os.path.join(AUDIT, "v8_reproduced_fold3.npy"), h_on_2.predict(M3))
    np.save(os.path.join(AUDIT, "v8_reproduced_fold2.npy"), h_on_3.predict(M2))

    repro = {
        "v7_chosen_head": chosen["head"],
        "v7_chosen_kind": chosen["kind"],
        "v7_chosen_meta_mode": chosen["meta_mode"],
        "v7_members": names,
        "v7_target_fold3_ap": chosen["fold3_ap"],
        "v7_target_fold2_ap": chosen["fold2_ap"],
        "lofo_reproduced_fold3_ap": float(lofo_f3),
        "lofo_reproduced_fold2_ap": float(lofo_f2),
        "lofo_fold3_match": bool(abs(lofo_f3 - chosen["fold3_ap"]) < 1e-9),
        "lofo_fold2_match": bool(abs(lofo_f2 - chosen["fold2_ap"]) < 1e-9),
        "reproduced_test_mean": float(test_pred.mean()),
        "reproduced_test_max_abs_diff_vs_stored": diff_stored,
        "reproduced_test_exact_match": bool(diff_stored < 1e-9),
        "stored_vs_submission_diff": diff_csv,
    }
    jdump(repro, os.path.join(AUDIT, "v7_exact_reproduction.json"))
    log("Saved v7_exact_reproduction.json")


if __name__ == "__main__":
    main()


