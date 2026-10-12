"""V7 common utilities.

Shared fold protocol, coherent feature transform, metric helpers, submission IO.

Chronological fold protocol (verified against jupyter notebook/Fold_2|3_v4_targets.npy):
    fold-1 = rows [0       : 613012)
    fold-2 = rows [613012  : 672477)   (2026-06-15 .. 2026-06-30)
    fold-3 = rows [672477  : 731942)   (2026-07-01 .. 2026-07-15)
    test   = rows starting 2026-07-16  (262648 rows)

The metric is Average Precision (AP).
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
JN = os.path.join(ROOT, "jupyter notebook")
MEGA = os.path.join(JN, "AGENT_EXP_MEGA_V2")
OUT = os.path.join(ROOT, "v7_out")
os.makedirs(OUT, exist_ok=True)

N_TRAIN = 731942
N_TEST = 262648
FOLD1_END = 613012
FOLD2_START, FOLD2_END = 613012, 672477
FOLD3_START, FOLD3_END = 672477, 731942

IDX_F2 = np.arange(FOLD2_START, FOLD2_END)
IDX_F3 = np.arange(FOLD3_START, FOLD3_END)
IDX_F1 = np.arange(0, FOLD1_END)

EPS = 1e-9

# documented reference scores (from experiment_tracker.csv / best_models.md)
REF = {
    "hgb183_fulldata_fold3": 0.524681,   # stranded on Colab W3 (COL3)
    "xgb183_fulldata_fold3": 0.523993,
    "champion_fold3": 0.518226,
    "champion_public": 0.526420,
    "best_blend_fold3": 0.527088,
    "graph_screen_gain_fold3": 0.001180,
}

# existing local artifacts (legacy pool). (fold2_path, fold3_path, test_path)
LEGACY = {
    "xgb183_gate": ("gate_xgb183_FOLD2_pred.npy", "gate_xgb183_FOLD3_pred.npy",
                    os.path.join(JN, "AGENT_EXP_XGB", "xgb183_test_pred.npy")),
    "hgb600_gate": ("gate_hgb600_FOLD2_pred.npy", "gate_hgb600_FOLD3_pred.npy",
                    os.path.join(JN, "AGENT_EXP_AUTO_HGB", "HGB600_fulldata_test_pred.npy")),
    "megalgb_gate": ("gate_lgb_FOLD2_pred.npy", "gate_lgb_FOLD3_pred.npy",
                     os.path.join(JN, "AGENT_EXP_MEGA_V2", "lgb_mega_test_pred.npy")),
    "megacb_gate": ("gate_cb_FOLD2_pred.npy", "gate_cb_FOLD3_pred.npy",
                    os.path.join(JN, "AGENT_EXP_MEGA_V2", "cb_mega_test_pred.npy")),
}


def ap(y, p):
    """Average Precision (the competition metric)."""
    return float(average_precision_score(np.asarray(y), np.asarray(p)))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def rank01(v):
    """Strictly monotone rank transform scaled to [0,1] (ties averaged)."""
    s = pd.Series(np.asarray(v))
    r = s.rank(method="average").to_numpy(dtype=np.float64)
    return r / max(len(v) - 1, 1)


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def load_core(verbose=True):
    """Load base 183-feature matrices, labels and fold targets with alignment asserts."""
    X = pd.read_parquet(os.path.join(MEGA, "train_combined.parquet"))
    Xte = pd.read_parquet(os.path.join(MEGA, "test_combined.parquet"))
    y = np.load(os.path.join(MEGA, "y.npy")).astype(np.int8)
    y2 = np.load(os.path.join(JN, "Fold_2_v4_targets.npy")).astype(np.int8)
    y3 = np.load(os.path.join(JN, "Fold_3_v4_targets.npy")).astype(np.int8)

    assert len(X) == N_TRAIN, (len(X), N_TRAIN)
    assert len(Xte) == N_TEST, (len(Xte), N_TEST)
    assert X.columns.tolist() == Xte.columns.tolist(), "train/test column mismatch"
    assert len(y) == N_TRAIN
    assert np.array_equal(y[IDX_F2], y2), "fold-2 target alignment FAILED"
    assert np.array_equal(y[IDX_F3], y3), "fold-3 target alignment FAILED"
    assert set(np.unique(y)).issubset({0, 1})

    if verbose:
        print(f"[core] train={X.shape} test={Xte.shape} feats={X.shape[1]}", flush=True)
        print(f"[core] fold-2 rows {FOLD2_START}:{FOLD2_END} pos_rate={y2.mean():.4%}", flush=True)
        print(f"[core] fold-3 rows {FOLD3_START}:{FOLD3_END} pos_rate={y3.mean():.4%}", flush=True)
        print(f"[core] global pos_rate={y.mean():.4%}", flush=True)
    return X, Xte, y, y2, y3


def build_matrix(X, Xte, verbose=True):
    """Build ONE coherent feature frame used by every fold fit and the full fit.

    - adds `<col>_na` indicators for every column that has NaN inside fold-1,
    - fills NaN with the *fold-1* column mean.

    Fold-1 is part of the training rows of both fold-2 and fold-3 fits and of the full
    fit, so this transform is leakage-free with respect to every validation fold (no
    validation-fold statistics are used) and identical across folds, which keeps fold
    OOF predictions and test predictions on the same scale.
    """
    names = list(X.columns)
    ref = X.iloc[:FOLD1_END]
    na_cols = [c for c in names if bool(ref[c].isna().any())]
    fills = {c: float(ref[c].mean()) for c in na_cols}

    def _apply(df):
        out = df.copy()
        for c in na_cols:
            out[c + "_na"] = df[c].isna().astype(np.int8)
        if na_cols:
            out[na_cols] = out[na_cols].fillna(pd.Series(fills))
        return out

    Xf = _apply(X)
    Xte_f = _apply(Xte)
    feat_names = list(Xf.columns)
    Xn = np.ascontiguousarray(Xf.to_numpy(dtype=np.float32))
    Xte_n = np.ascontiguousarray(Xte_f.to_numpy(dtype=np.float32))
    del Xf, Xte_f

    assert Xn.shape[1] == Xte_n.shape[1] == len(feat_names)
    assert np.isfinite(Xn).all(), "train frame still contains non-finite values"
    assert np.isfinite(Xte_n).all(), "test frame still contains non-finite values"

    if verbose:
        print(f"[matrix] feats={len(feat_names)} (base={len(names)}, na_ind={len(na_cols)})", flush=True)
    return Xn, Xte_n, feat_names, na_cols, fills


def fold_views(y):
    """Return dict fold -> (tr_idx, va_idx) for the two backward folds."""
    views = {}
    for fold, va_idx in (("fold2", IDX_F2), ("fold3", IDX_F3)):
        mask = np.ones(len(y), dtype=bool)
        mask[va_idx] = False
        views[fold] = (np.flatnonzero(mask), va_idx)
    return views


def load_pred(path):
    return np.asarray(np.load(path), dtype=np.float64).ravel()


def load_legacy_pool(verbose=True):
    """Load the pre-existing local member pool (fold-2 OOF, fold-3 OOF, test prediction)."""
    pool = {}
    for name, (f2, f3, te) in LEGACY.items():
        p2, p3, pt = os.path.join(ROOT, f2), os.path.join(ROOT, f3), te
        if not (os.path.exists(p2) and os.path.exists(p3) and os.path.exists(pt)):
            if verbose:
                print(f"[legacy] SKIP {name} (missing artifact)", flush=True)
            continue
        a, b, c = load_pred(p2), load_pred(p3), load_pred(pt)
        if len(a) != len(IDX_F2) or len(b) != len(IDX_F3) or len(c) != N_TEST:
            if verbose:
                print(f"[legacy] SKIP {name} (length mismatch)", flush=True)
            continue
        pool[name] = {"fold2": a, "fold3": b, "test": c, "source": "legacy"}
    return pool


def build_submission(pred, path):
    ss = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    assert len(ss) == N_TEST, len(ss)
    sub = ss.copy()
    sub["fraud"] = np.asarray(pred, dtype=np.float64)
    sub.to_csv(path, index=False)
    return verify_submission(path, pred)


def verify_submission(path, pred):
    ss = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    sub = pd.read_csv(path)
    p = sub["fraud"].to_numpy(dtype=np.float64)
    src = np.asarray(pred, dtype=np.float64)
    checks = {
        "row_count": int(len(sub)) == N_TEST,
        "columns_exact": list(sub.columns) == ["transaction_id", "fraud"],
        "id_order_matches_sample": bool((sub["transaction_id"].to_numpy() ==
                                         ss["transaction_id"].to_numpy()).all()),
        "no_duplicates": bool(int(sub["transaction_id"].duplicated().sum()) == 0),
        "no_missing": bool(int(sub[["transaction_id", "fraud"]].isna().sum().sum()) == 0),
        "finite": bool(np.isfinite(p).all()),
        "in_range_0_1": bool((p >= 0).all() and (p <= 1).all()),
        "roundtrip_precision": bool(len(p) == len(src)
                                    and float(np.max(np.abs(p - src))) < 1e-9),
    }
    checks["valid"] = bool(all(checks.values()))
    return checks


def member_matrix(pool, tag, names, meta_mode="full"):
    """Stack member predictions for one split into a meta-feature matrix.

    meta_mode:
      'prob' -> raw probabilities only
      'full' -> probabilities + logits + within-split ranks + spread/disagreement stats
    """
    P = np.column_stack([pool[n][tag] for n in names])
    if meta_mode == "prob":
        return np.ascontiguousarray(P, dtype=np.float64)
    if meta_mode == "rank":
        R = np.column_stack([rank01(P[:, j]) for j in range(P.shape[1])])
        return np.ascontiguousarray(R, dtype=np.float64)
    L = np.column_stack([logit(P[:, j]) for j in range(P.shape[1])])
    R = np.column_stack([rank01(P[:, j]) for j in range(P.shape[1])])
    stats = np.column_stack([
        L.mean(axis=1), L.std(axis=1), L.min(axis=1), L.max(axis=1),
        L.max(axis=1) - L.min(axis=1),
    ])
    Z = L - L.mean(axis=1, keepdims=True)
    M = np.column_stack([P, L, R, stats, Z])
    return np.ascontiguousarray(M, dtype=np.float64)


def meta_feature_names(names):
    """Column names of the 'full' meta matrix produced by member_matrix().

    The matrix layout is [P, L, R, stats, Z] where P/L/R are grouped by block, so the
    names must be grouped the same way (not interleaved per member).
    """
    out = [f"{n}_prob" for n in names]
    out += [f"{n}_logit" for n in names]
    out += [f"{n}_rank" for n in names]
    out += ["logit_mean", "logit_std", "logit_min", "logit_max", "logit_spread"]
    out += [f"{n}_z" for n in names]
    return out


def pair_corr_report(pool, tag, names):
    """Pearson + Spearman correlation of the member pool on one split (0.97 bottleneck check)."""
    M = np.column_stack([pool[n][tag] for n in names])
    df = pd.DataFrame(M, columns=names)
    return {
        "pearson": df.corr(method="pearson").round(4).to_dict(),
        "spearman": df.corr(method="spearman").round(4).to_dict(),
        "median_abs_pearson": float(np.median(np.abs(
            df.corr(method="pearson").to_numpy()[np.triu_indices(len(names), 1)]))),
        "median_abs_spearman": float(np.median(np.abs(
            df.corr(method="spearman").to_numpy()[np.triu_indices(len(names), 1)]))),
    }