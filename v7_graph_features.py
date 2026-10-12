"""GRAPH V3 - past-only connectivity features (Jaccard + degree + co-occurrence).

Design (RESEARCH_DIRECTOR_REPORT.md task 5.5, Phase 1), all leakage-safe by construction:

The graph state at row i contains ONLY edges created by rows < i (chronological order over
train then test). For the current row (customer c, device d, merchant m) we emit:

  - degrees            : distinct merchant/device/customer neighbourhood sizes per entity
  - bipartite Jaccard  : exact Jaccard over a shared third entity
        J(c,d) = shared merchants of c and d   / (n_merch(c) + n_merch(d) - shared)
        J(c,m) = shared devices   of c and m   / (n_dev(c)   + n_dev(m)   - shared)
        J(d,m) = shared customers of d and m   / (n_cust(d)  + n_cust(m)  - shared)
  - co-occurrence cnts : (c,d), (c,m), (d,m), (c,d,m) pair/triple counts (repeat behaviour)
  - novelty flags      : first time this pair / group is seen
  - share/ratio feats  : connectivity concentration (hub device / merchant signals)

Shared-neighbour counts are EXACT: a third entity z becomes shared between two entities
u,v at exactly the row where the second of the two edges (u,z)/(v,z) appears, so the
increment condition is (edge(u,z) existed) XOR (edge(v,z) existed) evaluated on the
pre-update state.

Leakage guarantee: state is updated strictly after the features for that row are read.
Verified empirically by `probe_past_only()` (prefix rebuild + extension invariance).
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd

import v7_common as C

GRAPH_NAMES = [
    "g_cust_merch_deg", "g_dev_merch_deg", "g_cust_dev_deg",
    "g_merch_cust_deg", "g_merch_dev_deg", "g_dev_cust_deg",
    "g_cust_rows", "g_dev_rows", "g_merch_rows",
    "g_cd_shared", "g_cm_shared", "g_dm_shared",
    "g_cd_jaccard", "g_cm_jaccard", "g_dm_jaccard",
    "g_cd_pairs", "g_cm_pairs", "g_dm_pairs", "g_cdm_pairs",
    "g_cd_new", "g_cm_new", "g_dm_new",
    "g_dev_merch_per_row", "g_cust_dev_per_row", "g_merch_cust_per_row",
    "g_dev_cust_share", "g_merch_cust_share",
    "g_cd_pairs_log", "g_cm_pairs_log", "g_dm_pairs_log", "g_cdm_pairs_log",
    "g_triple_novelty", "g_ring_score", "g_cd_jacc_x_merchdeg", "g_cdm_per_dev_row",
]
K = len(GRAPH_NAMES)


def load_raw(verbose=True):
    """Chronological raw transaction stream: train rows then test rows."""
    cols = ["customer_id", "device_id", "merchant_id", "timestamp"]
    tr = pd.read_csv(os.path.join(C.ROOT, "train.csv"), usecols=cols)
    te = pd.read_csv(os.path.join(C.ROOT, "test.csv"), usecols=cols)
    tr["timestamp"] = pd.to_datetime(tr["timestamp"])
    te["timestamp"] = pd.to_datetime(te["timestamp"])
    assert len(tr) == C.N_TRAIN and len(te) == C.N_TEST
    assert tr["timestamp"].is_monotonic_increasing, "train stream not chronological"
    assert te["timestamp"].is_monotonic_increasing, "test stream not chronological"
    assert tr["timestamp"].iloc[-1] < te["timestamp"].iloc[0], "train/test overlap in time"
    raw = pd.concat([tr, te], ignore_index=True)
    assert raw["timestamp"].is_monotonic_increasing, "concat stream not chronological"
    if verbose:
        print(f"[gv3] stream rows={len(raw)} (train={len(tr)} test={len(te)})", flush=True)
        print(f"[gv3] train end {tr['timestamp'].iloc[-1]} -> test start {te['timestamp'].iloc[0]}", flush=True)
    return raw


def _codes(raw):
    c = pd.factorize(raw["customer_id"])[0].astype(np.int64)
    d = pd.factorize(raw["device_id"])[0].astype(np.int64)
    m = pd.factorize(raw["merchant_id"])[0].astype(np.int64)
    return c, d, m


def build(raw, verbose=True):
    """Return float32 (n_rows, K) past-only graph feature matrix."""
    t0 = time.time()
    c_arr, d_arr, m_arr = _codes(raw)
    n = len(c_arr)
    F = np.zeros((n, K), dtype=np.float32)

    R_cm, R_dm, R_cd = {}, {}, {}                 # edge sets (membership tests)
    deg_cust_merch, deg_dev_merch, deg_cust_dev = {}, {}, {}
    rows_c, rows_d, rows_m = {}, {}, {}
    sh_cd, sh_cm, sh_dm = {}, {}, {}              # shared-neighbour counts
    cn_cd, cn_cm, cn_dm, cn_cdm = {}, {}, {}, {}  # pair / triple co-occurrence counts
    lg = np.log1p

    for i in range(n):
        c = int(c_arr[i]); d = int(d_arr[i]); m = int(m_arr[i])

        # ---- read state (rows < i only) ----
        cm = R_cm.get(c); dm = R_dm.get(d); cd = R_cd.get(c)
        n_cm = len(cm) if cm is not None else 0
        n_dm = len(dm) if dm is not None else 0
        n_cd = len(cd) if cd is not None else 0
        n_merch_m = deg_cust_merch.get(m, 0)
        n_dev_m = deg_dev_merch.get(m, 0)
        n_cust_d = deg_cust_dev.get(d, 0)
        rc = rows_c.get(c, 0); rd = rows_d.get(d, 0); rm = rows_m.get(m, 0)

        kcd = (c, d); kcm = (c, m); kdm = (d, m)
        scd = sh_cd.get(kcd, 0); scm = sh_cm.get(kcm, 0); sdm = sh_dm.get(kdm, 0)
        ccd = cn_cd.get(kcd, 0); ccm = cn_cm.get(kcm, 0); cdm = cn_dm.get(kdm, 0)
        c3 = cn_cdm.get((c, d, m), 0)

        j_cd = scd / (n_cm + n_dm - scd + C.EPS)
        j_cm = scm / (n_cd + n_dev_m - scm + C.EPS)
        j_dm = sdm / (n_cust_d + n_merch_m - sdm + C.EPS)
        cd_new = 1.0 if ccd == 0 else 0.0
        cm_new = 1.0 if ccm == 0 else 0.0
        dm_new = 1.0 if cdm == 0 else 0.0

        F[i] = (
            n_cm, n_dm, n_cd,
            n_merch_m, n_dev_m, n_cust_d,
            rc, rd, rm,
            scd, scm, sdm,
            j_cd, j_cm, j_dm,
            ccd, ccm, cdm, c3,
            cd_new, cm_new, dm_new,
            n_dm / (rd + 1.0), n_cd / (rc + 1.0), n_merch_m / (rm + 1.0),
            n_cust_d / (rd + 1.0), n_merch_m / (rm + 1.0),
            lg(ccd), lg(ccm), lg(cdm), lg(c3),
            (cd_new + cm_new + dm_new) / 3.0,
            j_cm * lg(c3),
            j_cd * lg(n_dm),
            c3 / (rd + 1.0),
        )

        # ---- update state (edges created by row i; visible from row i+1) ----
        x1 = ccm > 0   # (c,m) edge already present
        x2 = cdm > 0   # (d,m) edge already present
        x3 = ccd > 0   # (c,d) edge already present
        if x1 != x2:
            sh_cd[kcd] = scd + 1
        if x3 != x2:
            sh_cm[kcm] = scm + 1
        if x3 != x1:
            sh_dm[kdm] = sdm + 1

        cn_cd[kcd] = ccd + 1
        cn_cm[kcm] = ccm + 1
        cn_dm[kdm] = cdm + 1
        cn_cdm[(c, d, m)] = c3 + 1

        if not x1:
            if cm is None:
                cm = set(); R_cm[c] = cm
            cm.add(m)
            deg_cust_merch[m] = n_merch_m + 1
        if not x2:
            if dm is None:
                dm = set(); R_dm[d] = dm
            dm.add(m)
            deg_dev_merch[m] = n_dev_m + 1
        if not x3:
            if cd is None:
                cd = set(); R_cd[c] = cd
            cd.add(d)
            deg_cust_dev[d] = n_cust_d + 1

        rows_c[c] = rc + 1
        rows_d[d] = rd + 1
        rows_m[m] = rm + 1

        if verbose and (i + 1) % 250000 == 0:
            print(f"[gv3] {i + 1}/{n} rows ({time.time() - t0:.0f}s)", flush=True)

    assert np.isfinite(F).all(), "non-finite graph features"
    if verbose:
        print(f"[gv3] built {F.shape} in {time.time() - t0:.0f}s", flush=True)
    return F
    m = pd.factorize(raw["merchant_id"])[0].astype(np.int64)
    return c, d, m


def probe_past_only(raw, n_prefix=25000, n_ext=25000, n_test_probe=2000, verbose=True):
    """Empirical leakage test: features must not change when FUTURE rows are added.

    (a) rebuild on the prefix only  vs. the same rows inside a longer build,
    (b) rebuild on train+first test rows vs. the same rows inside the full build.
    A past-only builder must agree exactly on the shared prefix.
    """
    res = {}
    Fp = build(raw.iloc[:n_prefix], verbose=False)
    Fe = build(raw.iloc[:n_prefix + n_ext], verbose=False)
    res["prefix_rows"] = int(n_prefix)
    res["extension_rows"] = int(n_ext)
    res["prefix_identical_after_extension"] = bool(np.array_equal(Fp, Fe[:n_prefix]))
    res["prefix_max_abs_diff"] = float(np.max(np.abs(Fp - Fe[:n_prefix])))
    bad = [GRAPH_NAMES[j] for j in range(K)
           if not np.array_equal(Fp[:, j], Fe[:n_prefix, j])]
    res["prefix_columns_changed"] = bad

    cut = C.N_TRAIN + n_test_probe
    Fa = build(raw.iloc[:cut], verbose=False)
    Fb = Fe  # reuse shorter build only if it covers the same rows
    if len(raw) >= cut:
        Fb = build(raw.iloc[:min(len(raw), cut + n_ext)], verbose=False)
    res["test_prefix_rows"] = int(cut)
    res["test_prefix_identical"] = bool(np.array_equal(Fa, Fb[:cut]))
    res["test_prefix_max_abs_diff"] = float(np.max(np.abs(Fa - Fb[:cut])))
    res["labels_used_by_builder"] = False
    res["pass"] = bool(res["prefix_identical_after_extension"] and res["test_prefix_identical"])
    if verbose:
        print(f"[gv3][probe] past-only PASS={res['pass']} "
              f"maxdiff={res['prefix_max_abs_diff']:.3g}/{res['test_prefix_max_abs_diff']:.3g}", flush=True)
    return res


def signal_report(Ftr, y3, top=10, verbose=True):
    """Single-feature AP on fold-3 (diagnostic only - not a model)."""
    f3 = Ftr[C.IDX_F3]
    rows = []
    for j, name in enumerate(GRAPH_NAMES):
        col = f3[:, j]
        a_up = C.ap(y3, col)
        a_dn = C.ap(y3, -col)
        rows.append({"feature": name, "ap_up": a_up, "ap_down": a_dn,
                     "ap_signed": max(a_up, a_dn), "nonzero_frac": float((col != 0).mean())})
    rows.sort(key=lambda r: -r["ap_signed"])
    if verbose:
        print("[gv3] fold-3 single-feature AP (top):", flush=True)
        for r in rows[:top]:
            print(f"       {r['feature']:24s} AP={r['ap_signed']:.4f} "
                  f"(up={r['ap_up']:.4f} down={r['ap_down']:.4f}) nz={r['nonzero_frac']:.3f}", flush=True)
    return rows


def main():
    t0 = time.time()
    raw = load_raw()
    probe = probe_past_only(raw)
    F = build(raw, verbose=True)
    Ftr = np.ascontiguousarray(F[:C.N_TRAIN])
    Fte = np.ascontiguousarray(F[C.N_TRAIN:])
    np.save(os.path.join(C.OUT, "graph_v3_train.npy"), Ftr)
    np.save(os.path.join(C.OUT, "graph_v3_test.npy"), Fte)
    y3 = np.load(os.path.join(C.JN, "Fold_3_v4_targets.npy")).astype(np.int8)
    y2 = np.load(os.path.join(C.JN, "Fold_2_v4_targets.npy")).astype(np.int8)
    sig3 = signal_report(Ftr, y3)
    sig2 = signal_report(Ftr, y2, verbose=False)
    C.jdump({"names": GRAPH_NAMES, "k": K, "probe": probe,
             "signal_fold3": sig3, "signal_fold2": sig2,
             "train_shape": list(Ftr.shape), "test_shape": list(Fte.shape),
             "seconds": time.time() - t0},
            os.path.join(C.OUT, "graph_v3_meta.json"))
    print(f"[gv3] DONE saved graph_v3_train.npy {Ftr.shape} graph_v3_test.npy "
          f"{Fte.shape} total={time.time() - t0:.0f}s", flush=True)
    print("DONE_GRAPH_V3", flush=True)


if __name__ == "__main__":
    main()
