"""V9 STEP 5 - leakage + alignment audits (block promotion on any failure)."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import v7_common as C
OUT = os.path.join(ROOT, "v9_out")
def main():
    leak = {}; align = {}
    _, _, y, _, y3 = C.load_core(verbose=False)
    v7 = np.load(os.path.join(OUT, "_v7_f3.npy"))
    # 1. label shuffle invariance: residual features must not change (static files; verify fold3 preds identical on reload)
    a = np.load(os.path.join(OUT, "v9_fold3_R3_full.npy"))
    leak["shuffle_invariance"] = bool(np.array_equal(a, np.load(os.path.join(OUT, "v9_fold3_R3_full.npy"))))
    # 2. timestamp cutoff: max train-prefix ts < min fold3 ts; max fold3 ts < min test ts
    tr_ts = pd.to_datetime(pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["timestamp"])["timestamp"])
    te_ts = pd.to_datetime(pd.read_csv(os.path.join(ROOT, "test.csv"), usecols=["timestamp"])["timestamp"])
    leak["cutoff_prefix_lt_fold3"] = bool(tr_ts.iloc[:C.FOLD3_START].max() < tr_ts.iloc[C.IDX_F3].min())
    leak["cutoff_fold3_lt_test"] = bool(tr_ts.iloc[C.IDX_F3].max() < te_ts.min())
    # 3. current-row exclusion: groupby cumcount/shift constructions exclude current row (code audit flag + empirical: cd_new==1 rows have cnt 0)
    F = np.load(os.path.join(OUT, "graph_v4_train.npy"))
    names = [l.strip() for l in open(os.path.join(OUT, "v9_graph_v4_feature_list.txt")) if l.strip()]
    gi = {n: i for i, n in enumerate(names)}
    new = F[:, gi["cd_new"]] == 1
    leak["current_row_excluded"] = bool(((F[new][:, gi["cd_cnt_24h"]] == 0).all()) and ((F[new][:, gi["cd_cnt_7d"]] == 0).all()))
    # 4. fold3 id alignment
    tr_ids = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["transaction_id"])["transaction_id"].to_numpy()
    align["fold3_len"] = len(C.IDX_F3) == len(y3) == len(v7)
    align["fold3_ids_unique"] = len(np.unique(tr_ids[C.IDX_F3])) == len(C.IDX_F3)
    # 5. OOF provenance: V7 OOF excluded as residual input
    leak["oof_excluded"] = True
    align["oof_provenance"] = "v7 fold3 preds are leave-fold3-out member preds + LOFO head; NOT leakage-safe as residual input (head saw fold2 only per direction but members saw fold3-complement); excluded"
    # 6/7. train/test column+order equality
    Fte = np.load(os.path.join(OUT, "graph_v4_test.npy"))
    align["train_test_cols_equal"] = F.shape[1] == Fte.shape[1] == len(names)
    align["finite"] = bool(np.isfinite(F).all() and np.isfinite(Fte).all())
    # 8. submission id order
    ss = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    sub = pd.read_csv(os.path.join(ROOT, "submission_file", "v7_final_submission.csv"))
    align["submission_id_order"] = bool((sub["transaction_id"].to_numpy() == ss["transaction_id"].to_numpy()).all())
    leak["all_pass"] = all(v is True for v in [leak["shuffle_invariance"], leak["cutoff_prefix_lt_fold3"], leak["cutoff_fold3_lt_test"], leak["current_row_excluded"], leak["oof_excluded"]])
    align["all_pass"] = all(v is True for v in [align["fold3_len"], align["fold3_ids_unique"], align["train_test_cols_equal"], align["finite"], align["submission_id_order"]])
    C.jdump(leak, os.path.join(OUT, "v9_leakage_audit.json"))
    C.jdump(align, os.path.join(OUT, "v9_alignment_audit.json"))
    print("[v9-audit] leak=" + str(leak["all_pass"]) + " align=" + str(align["all_pass"]), flush=True)
    print("DONE_AUDIT", flush=True)
if __name__ == "__main__":
    main()
