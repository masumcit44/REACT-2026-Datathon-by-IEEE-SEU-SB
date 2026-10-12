"""V7 docs - refresh the V7 section of best_models.md from the pipeline artifacts."""
import os
import json

import v7_common as C

MARKER = "## V7 ("


def _load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_section():
    members = _load(os.path.join(C.OUT, "member_scores.json"), {}) or {}
    val = _load(os.path.join(C.OUT, "v7_validation_results.json"), {}) or {}
    sel = _load(os.path.join(C.OUT, "v7_selection.json"), {}) or {}
    cons = _load(os.path.join(C.OUT, "v7_consistency.json"), {}) or {}
    mm = members.get("members", {})

    L = []
    A = L.append
    A(f"## V7 (2026-09-15) - HGB183 recovery + Graph V3 + stacking head")
    A("")
    A("Local chronological AP only (fold-2 = rows 613012:672477, fold-3 = 672477:731942).")
    A("No leaderboard feedback exists for any V7 number. Artifacts in `v7_out/`, report in")
    A("`V7_REPORT.md`, final submission `submission_file/v7_final_submission.csv`.")
    A("")
    hm = mm.get("hgb183", {})
    rec = (hm.get("recovery") or {})
    A("### 1. Recovered HGB183 (was stranded on Colab W3, not downloadable)")
    A("")
    if hm:
        A(f"- re-derived locally with the documented recipe (sklearn HGB 600it/lr.05/leaf63/"
          f"minleaf100/l2 1.0) on the {rec.get('frame_chosen', hm.get('frame'))} frame "
          f"({hm.get('n_features')} features).")
        A(f"- fold-3 AP **{hm.get('fold3_ap', float('nan')):.6f}** vs recorded 0.524681 "
          f"(delta {rec.get('delta_vs_recorded', float('nan')):+.6f}); fold-2 "
          f"{hm.get('fold2_ap', float('nan')):.6f}.")
        A(f"- integrated as a pool member with coherent fold-2/fold-3/test predictions.")
    A("")
    A("### 2. Graph V3 past-only features (35, leakage probe PASS, max diff 0.0)")
    A("")
    A("| member | feats | fold-2 AP | fold-3 AP | trees/protocol |")
    A("|---|---|---|---|---|")
    for m in ("lgb183", "graphlgb", "xgb183", "hgb183"):
        d = mm.get(m)
        if not d:
            continue
        A(f"| {m} | {d.get('n_features')} | {d['fold2_ap']:.6f} | {d['fold3_ap']:.6f} | "
          f"{d.get('full_rounds')} ({d.get('iteration_protocol', 'es_internal_slice')}) |")
    A("")
    A("### 3. Stacking head vs weighted blending (leave-one-fold-out)")
    A("")
    cpath = os.path.join(C.OUT, "v7_candidate_scores.csv")
    if os.path.exists(cpath):
        import pandas as pd
        cand = pd.read_csv(cpath)
        cand = cand.sort_values("fold3_ap", ascending=False).head(12)
        A("| candidate | kind | meta | fold-2 AP | fold-3 AP |")
        A("|---|---|---|---|---|")
        for _, r in cand.iterrows():
            A(f"| {r.get('candidate')} | {r.get('kind')} | {r.get('meta_mode')} | "
              f"{r['fold2_ap']:.6f} | {r['fold3_ap']:.6f} |")
    if val:
        bb, bs = val.get("best_blend", {}), val.get("best_stack") or {}
        A("")
        A(f"- best weighted blend: `{bb.get('head')}` f3={bb.get('fold3_ap', float('nan')):.6f} "
          f"f2={bb.get('fold2_ap', float('nan')):.6f}")
        if bs:
            A(f"- best stack head: `{bs.get('head')}` f3={bs.get('fold3_ap', float('nan')):.6f} "
              f"f2={bs.get('fold2_ap', float('nan')):.6f} -> "
              f"**{'PROMOTED' if val.get('promoted') else 'NOT promoted'}**")
        A(f"- documented legacy blend reference: 0.527088 (fold-3)")
        ch = val.get("chosen_head", {})
        if ch.get("gain_share_by_meta_family"):
            A(f"- head gain share by meta-feature family: {ch['gain_share_by_meta_family']}")
        A(f"- head top inputs: " + ", ".join(list(ch.get("feature_importance_gain", {}))[:5]))
    if sel:
        A("")
        A(f"- pool selection: `{sel.get('winner_tag')}` ({sel.get('winner_pool')}) "
          f"f3={(sel.get('winner_candidate') or {}).get('fold3_ap', float('nan')):.6f} "
          f"f2={(sel.get('winner_candidate') or {}).get('fold2_ap', float('nan')):.6f}; "
          f"final submission `submission_file/v7_final_submission.csv` (verified valid).")
    if cons:
        ms, rg = cons.get("median_abs_spearman", {}), cons.get("regenerated_median_abs_spearman", {})
        A(f"- transfer-risk evidence: member-vs-member Spearman falls "
          f"{ms.get('fold3', float('nan')):.3f} (fold-3) -> {ms.get('test', float('nan')):.3f} "
          f"(test) for all members, but only "
          f"{rg.get('fold3', float('nan')):.3f} -> {rg.get('test', float('nan')):.3f} for the "
          f"V7-regenerated members; the legacy members carry the variant mismatch.")
    A("")
    return "\n".join(L) + "\n"


def main():
    path = os.path.join(C.ROOT, "best_models.md")
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    idx = txt.find(MARKER)
    section = build_section()
    if idx >= 0:
        txt = txt[:idx] + section
    else:
        txt = txt.rstrip() + "\n\n" + section
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[docs] best_models.md V7 section refreshed ({len(section)} chars)")


if __name__ == "__main__":
    main()