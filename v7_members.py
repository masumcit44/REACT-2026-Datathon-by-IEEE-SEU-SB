"""V7 MEMBER POOL - coherent local regeneration of the model pool.

Why regenerate: the legacy local artifacts mix incompatible fold/test variants (the gate
fold predictions were produced on a 215-column NA-indicator frame while
`HGB600_fulldata_test_pred.npy` came from a 124-feature frame, etc.). A stacking head
requires fold OOF and test predictions from the SAME recipe, so every member below is
trained with ONE pipeline (v7_common.build_matrix) on three splits:

    fold2 : train = all rows except [613012:672477)  -> predict fold-2
    fold3 : train = all rows except [672477:731942)  -> predict fold-3
    full  : train = all rows                        -> predict test

Members
-------
hgb183  : the documented "HGB183" recipe (best_models.md / experiment_tracker.csv):
          sklearn HistGradientBoostingClassifier, max_iter=600, learning_rate=0.05,
          max_leaf_nodes=63, min_samples_leaf=100, l2_regularization=1.0, seed 4242.
          This is the LOCAL RECOVERY of the stranded Colab-W3 artifact; its fold-3 AP is
          validated against the recorded 0.524681 (no early stopping -> directly
          comparable). A variant screen (base-183 native-NaN frame vs 215-col
          NA-indicator frame) identifies which frame reproduces the recorded score.
lgb183  : LightGBM recipe proven in this repo (gate): num_leaves=63, lr=0.05,
          feature_fraction=0.7, bagging=0.8/5, min_child_samples=100, l2=1.0.
xgb183  : canonical repo recipe: hist, max_depth=11, eta=0.05, min_child_weight=100,
          subsample=0.8, colsample_bytree=0.7, scale_pos_weight, lambda=1.0.
graphlgb: lgb183 recipe on base + Graph-V3 past-only features (v7_graph_features.py).

Honest iteration selection: early stopping uses an INTERNAL slice from the tail of fold-1
(rows 553012:613012). That slice belongs to the training rows of every fold fit and is
chronologically before both validation folds, so no evaluation fold ever influences the
iteration count. Full-data fits use round(median(fold2_best, fold3_best) * 1.05), min 100
(project convention, a.txt Phase 8).
"""
import os
import sys
import time
import json
import argparse

import numpy as np

import v7_common as C

ES_SLICE = (C.FOLD1_END - 60000, C.FOLD1_END)   # rows 553012:613012 (always in-sample)

CAT_COLS = ["customer_id", "merchant_id", "merchant_category", "device_id", "device_type",
            "location", "payment_method", "transaction_type", "customer_previous_device",
            "customer_previous_merchant", "customer_previous_location",
            "device_previous_customer", "merchant_previous_customer"]

HGB_PARAMS = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                  min_samples_leaf=100, l2_regularization=1.0, max_bins=255,
                  random_state=4242, early_stopping=False, validation_fraction=None)

LGB_PARAMS = dict(objective="binary", metric="average_precision", boosting_type="gbdt",
                  num_leaves=63, learning_rate=0.05, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=5, min_child_samples=100,
                  lambda_l2=1.0, seed=4242, verbose=-1, n_jobs=8)

XGB_PARAMS = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
                  eta=0.05, max_depth=11, min_child_weight=100, subsample=0.8,
                  colsample_bytree=0.7, gamma=0.0, lam=1.0, seed=4242, nthread=8)

ALL_MEMBERS = ["hgb183", "lgb183", "xgb183", "graphlgb"]


def cat_indices(feat_names):
    return [i for i, n in enumerate(feat_names) if n in CAT_COLS]


def path_of(member, kind):
    return os.path.join(C.OUT, f"{member}_{kind}.npy")


def meta_of(member):
    return os.path.join(C.OUT, f"{member}_meta.json")


def hgb_fit(Xtr, ytr, Xva, Xes, yes, cat_idx, params=None, rounds=600, patience=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = dict(HGB_PARAMS if params is None else params)
    clf = HistGradientBoostingClassifier(**p)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1], {"best_iter": int(p["max_iter"]), "params": p}


def lgb_fit(Xtr, ytr, Xva, Xes, yes, cat_idx, params=None, rounds=2000, patience=100):
    import lightgbm as lgb
    p = dict(LGB_PARAMS if params is None else params)
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_idx or "auto",
                      free_raw_data=True)
    des = lgb.Dataset(Xes, label=yes, reference=dtr, free_raw_data=True)
    cbs = [lgb.log_evaluation(0)]
    if patience and patience > 0:
        cbs.insert(0, lgb.early_stopping(int(patience), verbose=False))
    bst = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[des], callbacks=cbs)
    bi = int(bst.best_iteration) if (patience and patience > 0) else int(rounds)
    pred = bst.predict(Xva, num_iteration=bi)
    return pred, {"best_iter": bi, "params": p, "early_stopping": bool(patience)}


def xgb_fit(Xtr, ytr, Xva, Xes, yes, cat_idx, params=None, rounds=2000, patience=100):
    import xgboost as xgb
    p = dict(XGB_PARAMS if params is None else params)
    p["scale_pos_weight"] = float((1.0 - ytr.mean()) / max(ytr.mean(), 1e-9))
    dtr = xgb.DMatrix(Xtr, label=ytr)
    des = xgb.DMatrix(Xes, label=yes)
    esr = int(patience) if (patience and patience > 0) else None
    bst = xgb.train(p, dtr, num_boost_round=rounds, evals=[(des, "es")],
                    early_stopping_rounds=esr, verbose_eval=False)
    if esr is None:
        bi = int(rounds) - 1
    else:
        bi = int(bst.best_iteration if bst.best_iteration is not None else rounds - 1)
    pred = bst.predict(xgb.DMatrix(Xva), iteration_range=(0, bi + 1))
    return pred, {"best_iter": bi, "params": p, "early_stopping": bool(esr)}


FITTERS = {"hgb183": hgb_fit, "lgb183": lgb_fit, "xgb183": xgb_fit, "graphlgb": lgb_fit}
DEFAULT_ROUNDS = {"hgb183": 600, "lgb183": 2000, "xgb183": 2000, "graphlgb": 2000}


def _es_split(tr_idx):
    """Split a fold's training rows into (fit, early-stopping) rows.

    The early-stopping rows are the tail of fold-1, which is in-sample for every fold fit
    and strictly earlier than both validation folds.
    """
    in_es = (tr_idx >= ES_SLICE[0]) & (tr_idx < ES_SLICE[1])
    return tr_idx[~in_es], tr_idx[in_es]


def run_validation(member, X, y, tr_idx, va_idx, yva, cat_idx, frame="na215",
                   forced_rounds=None, verbose=True):
    """Fit on tr_idx (minus the internal ES slice) and predict the validation fold."""
    fit_idx, es_idx = _es_split(tr_idx)
    Xtr, ytr = X[fit_idx], y[fit_idx]
    Xes, yes = X[es_idx], y[es_idx]
    Xva = X[va_idx]
    fitter = FITTERS[member]
    rounds = int(forced_rounds if forced_rounds else DEFAULT_ROUNDS[member])
    patience = 0 if forced_rounds else (0 if member == "hgb183" else 100)
    t0 = time.time()
    pred, info = fitter(Xtr, ytr, Xva, Xes, yes, cat_idx, rounds=rounds, patience=patience)
    secs = time.time() - t0
    score = C.ap(yva, pred)
    info.update({"split_ap": score, "seconds": secs, "n_fit": int(len(fit_idx)),
                 "n_es": int(len(es_idx)), "frame": frame})
    if verbose:
        print(f"[v7] {member:9s} {info['frame']:6s} AP={score:.6f} best_iter={info['best_iter']} "
              f"fit={len(fit_idx)} es={len(es_idx)} {secs:.0f}s", flush=True)
    return pred, info


def run_full(member, X, y, Xte, cat_idx, rounds, frame="na215", verbose=True):
    """Fit on all labelled rows with a fixed tree count and predict test."""
    fitter = FITTERS[member]
    t0 = time.time()
    pred, info = fitter(X, y, Xte, X, y, cat_idx, rounds=int(rounds), patience=0)
    info.update({"seconds": time.time() - t0, "frame": frame, "n_fit": int(len(X)),
                 "rounds": int(rounds)})
    if verbose:
        print(f"[v7] {member:9s} {frame:6s} TEST mean={pred.mean():.6f} rounds={rounds} "
              f"{info['seconds']:.0f}s", flush=True)
    return pred, info


def hgb_variant_screen(Xraw, Xn, y, y3):
    """Which HGB183 frame reproduces the documented fold-3 AP 0.524681?

    variant 'base183' : 183 raw columns with native NaN (sklearn HGB NaN support)
    variant 'na215'   : 215 columns (NA indicators + fold-1 mean fill)
    Both variants are fitted on fold-1 + fold-2 and scored on fold-3, matching the
    protocol of the stranded W3 run.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr_idx = np.arange(0, C.FOLD2_END)          # fold-1 + fold-2
    ytr = y[tr_idx]
    out = {}
    for tag, mat in (("base183", Xraw), ("na215", Xn)):
        t0 = time.time()
        clf = HistGradientBoostingClassifier(**HGB_PARAMS)
        clf.fit(mat[tr_idx], ytr)
        p = clf.predict_proba(mat[C.IDX_F3])[:, 1]
        out[tag] = {"fold3_ap": C.ap(y3, p), "seconds": time.time() - t0,
                    "n_features": int(mat.shape[1])}
        print(f"[v7] hgb screen {tag:8s}: fold-3 AP={out[tag]['fold3_ap']:.6f} "
              f"nfeat={out[tag]['n_features']} ({out[tag]['seconds']:.0f}s)", flush=True)
    ref = C.REF["hgb183_fulldata_fold3"]
    for tag in out:
        out[tag]["delta_vs_recorded"] = out[tag]["fold3_ap"] - ref
        out[tag]["abs_delta"] = abs(out[tag]["delta_vs_recorded"])
    out["recorded_reference"] = ref
    return out


def main():
    argparse_ = argparse.ArgumentParser()
    argparse_.add_argument("--members", default=",".join(ALL_MEMBERS))
    argparse_.add_argument("--force", action="store_true")
    argparse_.add_argument("--skip-variant-screen", action="store_true")
    argparse_.add_argument("--fixed-rounds", type=int, default=0,
                           help="a-priori tree count for every split (no early stopping). "
                                "Use counts from documented prior evidence, never tuned on "
                                "a validation fold: lgb183/graphlgb=600, xgb183=500, "
                                "hgb183=600 (already fixed by its recipe).")
    args = argparse_.parse_args()
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    unknown = [m for m in members if m not in FITTERS]
    assert not unknown, f"unknown members: {unknown}"
    t00 = time.time()

    X, Xte, y, y2, y3 = C.load_core()
    Xn, Xte_n, feat_names, na_cols, fills = C.build_matrix(X, Xte)
    cat_idx = cat_indices(feat_names)
    views = C.fold_views(y)
    Ztr = Zte = None
    Xraw = Xte_raw = None
    if "graphlgb" in members:
        Gtr = np.load(os.path.join(C.OUT, "graph_v3_train.npy"))
        Gte = np.load(os.path.join(C.OUT, "graph_v3_test.npy"))
        assert Gtr.shape[0] == C.N_TRAIN and Gte.shape[0] == C.N_TEST
        Ztr = np.ascontiguousarray(np.hstack([Xn, Gtr.astype(np.float32)]))
        Zte = np.ascontiguousarray(np.hstack([Xte_n, Gte.astype(np.float32)]))
        print(f"[v7] graph-augmented frame {Ztr.shape} test {Zte.shape}", flush=True)
        del Gtr, Gte

    report = {"es_slice": list(ES_SLICE), "base_features": len(feat_names),
              "members": {}, "n_train": int(len(y)), "n_test": int(len(Xte_n))}
    for m in ALL_MEMBERS:                       # keep the full pool summary across runs
        if os.path.exists(meta_of(m)):
            with open(meta_of(m), "r", encoding="utf-8") as f:
                report["members"][m] = json.load(f)

    # ---------- HGB183 recovery: identify the frame that reproduces 0.524681 ----------
    frame_choice = "na215"
    if "hgb183" in members and not os.path.exists(meta_of("hgb183")):
        if args.skip_variant_screen:
            print("[v7] hgb variant screen skipped (--skip-variant-screen)", flush=True)
        else:
            Xraw = X.to_numpy(dtype=np.float32)
            Xte_raw = Xte.to_numpy(dtype=np.float32)
            screen = hgb_variant_screen(Xraw, Xn, y, y3)
            report["hgb183_frame_screen"] = screen
            frame_choice = min(("base183", "na215"),
                               key=lambda t: (screen[t]["abs_delta"], t != "na215"))
            print(f"[v7] hgb183 frame chosen = {frame_choice} "
                  f"(recorded ref {screen['recorded_reference']:.6f})", flush=True)

    # ---------- members ----------
    for member in members:
        mpath = meta_of(member)
        if os.path.exists(mpath) and not args.force:
            with open(mpath, "r", encoding="utf-8") as f:
                report["members"][member] = json.load(f)
            print(f"[v7] {member}: meta exists -> skip", flush=True)
            continue

        mat_tr, mat_te, frame = Xn, Xte_n, "na215"
        if member == "graphlgb":
            mat_tr, mat_te, frame = Ztr, Zte, "na215+gv3"
        elif member == "hgb183" and frame_choice == "base183":
            mat_tr, mat_te, frame = Xraw, Xte_raw, "base183"

        fr = int(args.fixed_rounds) if args.fixed_rounds > 0 else None
        p2, i2 = run_validation(member, mat_tr, y, views["fold2"][0], views["fold2"][1],
                                y2, cat_idx, frame=frame, forced_rounds=fr)
        p3, i3 = run_validation(member, mat_tr, y, views["fold3"][0], views["fold3"][1],
                                y3, cat_idx, frame=frame, forced_rounds=fr)
        if fr:
            rounds = fr
            protocol = f"fixed_a_priori_{fr}"
        else:
            rounds = max(100, int(round(float(np.median([i2["best_iter"], i3["best_iter"]])) * 1.05)))
            protocol = ("fixed_recipe_600" if member == "hgb183" else "es_internal_slice")
        pt, it = run_full(member, mat_tr, y, mat_te, cat_idx, rounds, frame=frame)

        np.save(path_of(member, "fold2"), p2.astype(np.float64))
        np.save(path_of(member, "fold3"), p3.astype(np.float64))
        np.save(path_of(member, "test"), pt.astype(np.float64))
        meta = {"member": member, "frame": frame, "n_features": int(mat_tr.shape[1]),
                "iteration_protocol": protocol,
                "fold2_ap": i2["split_ap"], "fold3_ap": i3["split_ap"],
                "fold2_best_iter": i2["best_iter"], "fold3_best_iter": i3["best_iter"],
                "full_rounds": rounds, "fold2_seconds": i2["seconds"],
                "fold3_seconds": i3["seconds"], "full_seconds": it["seconds"],
                "test_mean": float(pt.mean()), "mean_ap": (i2["split_ap"] + i3["split_ap"]) / 2.0,
                "worst_ap": min(i2["split_ap"], i3["split_ap"]),
                "es_slice": list(ES_SLICE), "early_stopping": bool(i2.get("early_stopping", False)),
                "params": i2["params"]}
        if member == "hgb183":
            meta["recovery"] = {
                "frame_chosen": frame_choice,
                "recorded_fold3_ap": C.REF["hgb183_fulldata_fold3"],
                "delta_vs_recorded": i3["split_ap"] - C.REF["hgb183_fulldata_fold3"],
                "screen": report.get("hgb183_frame_screen"),
                "note": ("local re-derivation of the Colab-W3 artifact; no network access to "
                         "the stranded worker"),
            }
        C.jdump(meta, mpath)
        report["members"][member] = meta

    report["seconds"] = time.time() - t00
    C.jdump(report, os.path.join(C.OUT, "member_scores.json"))

    print("\n[v7] ==== MEMBER SUMMARY (fold OOF AP) ====", flush=True)
    for m, d in report["members"].items():
        print(f"  {m:9s} f2={d['fold2_ap']:.6f} f3={d['fold3_ap']:.6f} "
              f"mean={d['mean_ap']:.6f} worst={d['worst_ap']:.6f} "
              f"iters={d['fold2_best_iter']}/{d['fold3_best_iter']} frame={d['frame']}",
              flush=True)
    print(f"[v7] total {report['seconds']:.0f}s", flush=True)
    print("DONE_MEMBERS", flush=True)


if __name__ == "__main__":
    main()
