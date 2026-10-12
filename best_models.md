# BEST MODELS (ranked, fold-3 AP)

Champion reference: 0.518226 (fold-3) | 0.52642 (public LB). Metric = Average Precision.
Last update: post-R4 wave (hgb600-fulldata, xgb183, hgb183, graph screen, xgb blend).

## Individual models

| Rank | Model | Feats | Fold-3 AP | Has test pred | Verified |
|------|-------|-------|-----------|---------------|----------|
| 1 | HGB183 (sklearn HGB 600it, W3) | 183 | **0.524681** | YES (on Colab W3 - NOT downloaded yet) | YES |
| 2 | XGB183 (xgb hist, 215 incl NA) | 183 | 0.523993 | YES (AGENT_EXP_XGB) | YES |
| 3 | HGB600 full-data + NA-indicators (W4) | 124 | 0.520187 | YES (AGENT_EXP_AUTO_HGB) | YES |
| 4 | HGB600 (sklearn HGB, 600 it, repro) | 124 | 0.519930 | NO | YES |
| 5 | HGB_tuned (HGB, 1000 it/lr.03) | 124 | 0.518915 | YES (fulldata) | YES |
| 6 | Champion LGB (cost-extended blend) | 101 | 0.518226 | YES | YES |
| 7 | Mega LGB (183 feats) | 183 | 0.515853 | YES | YES |
| 8 | Acceleration-tuned LGB | 110 | 0.512894 | YES | YES |
| 9 | Cost-sensitive LGB | 101 | 0.512163 | YES | YES |
| 10 | Extra-randomized LGB | 101 | 0.511359 | YES | YES |
| 11 | GOSS LGB / altseed LGB | 183 | 0.5128 / 0.5122 | - | rapidfire |
| 12 | Mega CatBoost | 183 | 0.505725 | YES | YES |
| 13 | CatBoost diversity | 101 | 0.503972 | YES | YES |

## Blends (fold-3)

| Rank | Recipe | AP | Gain | Verified / Passed |
|------|--------|-----|------|-------------------|
| 1 | XGB183 .57 + HGB600 .34 + mega_lgb .09 + mega_cb .002 | **0.527088** | +0.008862 | DONE-SUBMITREADY; needs FOLD-2 gate before submit |
| 2 | 5-way rapidfire (champ .05/hgb600 .25/hgbt .30/megalgb .25/megacb .15) | 0.523615 | +0.005389 | all test preds now EXIST |
| 3 | mega_5way_hgb600 (real hgb600 pred) | 0.523382 | +0.005156 | SUBMITREADY file mega_5way_hgb600_0523382.csv |
| 4 | restricted champ05/hgbt45/megalgb35/megacb15 | **0.523163** | +0.004937 | SUBMITTED (public baseline) |
| 5 | 4-way equal (hgb600+hgbt+megalgb+megacb) | 0.523483 | +0.005257 | equal weights |
| 6 | 3-way G/HGBt/ACC 50/40/10 | 0.521920 | +0.003694 | PASS global gate |
| 7 | G+HGBt@45% | 0.521640 | +0.003414 | cycle-2 champion |
| 8 | G+HGB600@50% | 0.521354 | +0.003128 | PASS |

## Feature / diversity notes
- GRAPH screen (COLAB2): base LGB 0.527920 -> +graph 0.529101, gain +0.001180 -> **PASS** cheap gate.
  graph_fulldata_lgb RUNNING on W2.
- mega_lgb vs champ: pearson 0.9761 (low diversity); HGB600 vs champ 0.9684; HGBt vs champ 0.9719.
- rapidfire LGBs (goss 0.9445, altseed 0.9348) least correlated -> good global-blend members.
- No spearman available (argrank artifact); recompute with proper ranking in future runs.

## Immediate next steps
1. FOLD-2 GATE the xgb_hgb600_blend (0.527088 fold-3) - pass >= -0.001, then submit
   submission_file/xgb_hgb600_blend_0525.csv (expected public > 0.53).
2. Download hgb183 test pred (0.524681, BEST INDIVIDUAL) from Colab W3 -> rerun blend
   optimizer: hgb183 + xgb183 + hgb600 + mega_lgb + mega_cb.
3. Add graph-feature LGB (W2) and CatBoost-183 GPU (W1) to pool when they land.

## V8 audit (2026-09-15) - champion verification, no new submission

V7 champion `core/lr_C0.1(full)` f3=0.527728 f2=0.829803 reproduces bit-exact
(LOFO f3/f2 match, test max-abs-diff 0.0; `v8_out/v7_exact_reproduction.json`).
The higher-fold `wide` pool (f3=0.528538) stays rejected by the coherence rule; its
stored test vector (mean 0.504) was additionally corrupted by an expit bug in
`v7_stack.to_prob` (fixed 2026-09-15; corrected mean in
`v8_out/v8_wide_nnls_CORRECTED_test.npy`, audit-only). Fold APs are unaffected
(expit is rank-invariant) and the champion `lr` head never passed through expit.
Conservative coherent-pool blends all lose to the champion (best top3_nnls
f3=0.527393; verdict KEEP CHAMPION). LOFO ablations: hgb183/xgb183 essential
(-0.00116/-0.00115 each), lgb183 redundant (+0.000178 when dropped), full meta
needed (-0.00071 prob-only). A post-hoc C grid finds lr_C0.01 f3=0.528149
(+0.000421, test Spearman 0.962 vs champion) but it was NOT in the V7 candidate
set and has no LB evidence, so it stays an alternate-only. Primary remains
`submission_file/v7_final_submission.csv` (valid). Evidence: `v8_out/`
(v8_conservative_blends, v8_lofo_ablation, v8_stack_complexity,
v8_leaderboard_evidence, v7_exact_reproduction).

## V7 (2026-09-15) - HGB183 recovery + Graph V3 + stacking head

Local chronological AP only (fold-2 = rows 613012:672477, fold-3 = 672477:731942).
No leaderboard feedback exists for any V7 number. Artifacts in `v7_out/`, report in
`V7_REPORT.md`, final submission `submission_file/v7_final_submission.csv`.

### 1. Recovered HGB183 (was stranded on Colab W3, not downloadable)

- re-derived locally with the documented recipe (sklearn HGB 600it/lr.05/leaf63/minleaf100/l2 1.0) on the na215 frame (215 features).
- fold-3 AP **0.525151** vs recorded 0.524681 (delta +0.000470); fold-2 0.830686.
- integrated as a pool member with coherent fold-2/fold-3/test predictions.

### 2. Graph V3 past-only features (35, leakage probe PASS, max diff 0.0)

| member | feats | fold-2 AP | fold-3 AP | trees/protocol |
|---|---|---|---|---|
| lgb183 | 215 | 0.811614 | 0.516441 | 600 (fixed_a_priori_600) |
| graphlgb | 250 | 0.810966 | 0.518219 | 600 (fixed_a_priori_600) |
| xgb183 | 215 | 0.827929 | 0.524547 | 500 (fixed_a_priori_500) |
| hgb183 | 215 | 0.830686 | 0.525151 | 630 (es_internal_slice) |

### 3. Stacking head vs weighted blending (leave-one-fold-out)

| candidate | kind | meta | fold-2 AP | fold-3 AP |
|---|---|---|---|---|
| lr_C0.1 | lr | full | 0.829803 | 0.527728 |
| ridge_alpha1.0 | ridge | full | 0.828906 | 0.527578 |
| ridge_alpha0.3 | ridge | full | 0.828895 | 0.527578 |
| ridge_alpha0.1 | ridge | full | 0.828894 | 0.527577 |
| lr_C1.0 | lr | full | 0.829348 | 0.527299 |
| nnls | nnls | prob | 0.830769 | 0.527236 |
| ridge_alpha1.0 | ridge | prob | 0.829851 | 0.527234 |
| lr_C10.0 | lr | full | 0.829023 | 0.527154 |
| legacy:xgb_hgb600_blend_0.527088 | legacy | weights | 0.831961 | 0.527080 |
| lgb_n_estimators600_learning_rate0.01_num_leaves4_max_depth2 | lgb | full | 0.826556 | 0.526981 |
| equal | equal | prob | 0.829536 | 0.526919 |
| lr_C1.0 | lr | prob | 0.829184 | 0.526896 |

- best weighted blend: `nnls` f3=0.527236 f2=0.830769
- best stack head: `lr_C0.1` f3=0.527728 f2=0.829803 -> **PROMOTED**
- documented legacy blend reference: 0.527088 (fold-3)
- head top inputs: 

- pool selection: `core` (core) f3=0.527728 f2=0.829803; final submission `submission_file/v7_final_submission.csv` (verified valid).

