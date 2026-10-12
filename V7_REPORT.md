# V7 REPORT - HGB183 recovery, Graph V3, Stacking Head, Final Submission

Metric: Average Precision on honest chronological folds. Protocol: fold-1 = rows 0:613012, fold-2 = 613012:672477 (2026-06-15..06-30), fold-3 = 672477:731942 (2026-07-01..07-15), test = 2026-07-16+ (262648 rows).

## 1. HGB183 recovery (integrated into the local pool)

The documented best single model (HGB183, recorded fold-3 AP **0.524681**) existed only on a Colab worker (COL3) that is not reachable from this workspace and was never downloaded locally. Recovery was therefore executed as a deterministic **local re-derivation** using the exact documented recipe (sklearn HistGradientBoosting, max_iter=600, lr=0.05, max_leaf_nodes=63, min_samples_leaf=100, l2=1.0, seed 4242), then validated against the recorded score.

Frame screen (fit fold-1+fold-2, scored fold-3; recorded reference 0.524681):

| frame | features | fold-3 AP | delta vs recorded |
|---|---|---|---|
| base183 | 183 | 0.528432 | +0.003751 |
| na215 | 215 | 0.527487 | +0.002806 |

- chosen frame: **na215**, features 215
- **recovered fold-3 AP = 0.525151** (recorded 0.524681, delta +0.000470)
- recovered fold-2 AP = 0.830686; test mean = 0.013624; full-fit trees = 630
- artifacts: `v7_out/hgb183_fold2.npy`, `v7_out/hgb183_fold3.npy`, `v7_out/hgb183_test.npy`, `v7_out/hgb183_meta.json`

## 2. Graph V3 past-only connectivity features (Jaccard + co-occurrence)

- built **35** features (train [731942, 35], test [262648, 35]) in 78s
- block contents: entity degrees (customer/device/merchant), exact bipartite Jaccard J(c,d)/J(c,m)/J(d,m) over the shared third entity, shared-neighbour counts, pair and triple co-occurrence counts, first-seen novelty flags, per-row connectivity shares (hub-device / merchant concentration) and two composite ring scores.
- construction: single chronological pass over train+test; state at row i contains only edges from rows < i.
- **leakage probe PASS = True** (rebuild prefix then extend with future rows): max abs diff 0.0 / 0.0; builder consumes no labels (False).

Strongest single-feature APs on fold-3 (diagnostic only):

| feature | AP (signed) | nonzero frac |
|---|---|---|
| g_dev_rows | 0.1232 | 0.998 |
| g_dev_merch_deg | 0.1206 | 0.998 |
| g_dev_merch_per_row | 0.1194 | 0.998 |
| g_dev_cust_share | 0.1179 | 0.998 |
| g_dev_cust_deg | 0.1164 | 0.998 |
| g_merch_dev_deg | 0.0435 | 0.999 |
| g_merch_rows | 0.0413 | 0.999 |
| g_cd_pairs | 0.0383 | 0.919 |

Gate: GraphLGB vs the *identical* LGB recipe without graph features.

| model | features | fold-2 AP | fold-3 AP | mean | worst |
|---|---|---|---|---|---|
| lgb183 | 215 | 0.811614 | 0.516441 | 0.664028 | 0.516441 |
| graphlgb | 250 | 0.810966 | 0.518219 | 0.664592 | 0.518219 |

- fold-3 gain **+0.001777**, fold-2 delta -0.000648.

## 3. Stacking head (breaks the 0.97 correlation bottleneck?)

- pool `core`: graphlgb, hgb183, lgb183, xgb183
- meta-features: member probabilities, logits, within-split ranks, spread (min/max/std) and per-member disagreement z-scores
- member correlation on fold-3 OOF: median |Pearson| = 0.9243, median |Spearman| = 0.3326 (the saturation level)

Candidate table, leave-one-fold-out (fit on one fold's OOF, score the other):

| candidate | kind | meta_mode | fold2_ap | fold3_ap | mean_ap | worst_ap |
|---|---|---|---|---|---|---|
| lr_C0.1 | lr | full | 0.829803 | 0.527728 | 0.678765 | 0.527728 |
| ridge_alpha1.0 | ridge | full | 0.828906 | 0.527578 | 0.678242 | 0.527578 |
| ridge_alpha0.3 | ridge | full | 0.828895 | 0.527578 | 0.678236 | 0.527578 |
| ridge_alpha0.1 | ridge | full | 0.828894 | 0.527577 | 0.678235 | 0.527577 |
| lr_C1.0 | lr | full | 0.829348 | 0.527299 | 0.678324 | 0.527299 |
| nnls | nnls | prob | 0.830769 | 0.527236 | 0.679002 | 0.527236 |
| ridge_alpha1.0 | ridge | prob | 0.829851 | 0.527234 | 0.678542 | 0.527234 |
| lr_C10.0 | lr | full | 0.829023 | 0.527154 | 0.678088 | 0.527154 |
| legacy:xgb_hgb600_blend_0.527088 | legacy | weights | 0.831961 | 0.527080 | 0.679520 | 0.527080 |
| lgb_n_estimators600_learning_rate0.01_num_leaves4_max_depth2 | lgb | full | 0.826556 | 0.526981 | 0.676768 | 0.526981 |
| equal | equal | prob | 0.829536 | 0.526919 | 0.678228 | 0.526919 |
| lr_C1.0 | lr | prob | 0.829184 | 0.526896 | 0.678040 | 0.526896 |
| equal | equal | rank | 0.828716 | 0.526619 | 0.677668 | 0.526619 |
| lgb_n_estimators300_learning_rate0.02_num_leaves7_max_depth3 | lgb | full | 0.825520 | 0.525936 | 0.675728 | 0.525936 |
| lgb_n_estimators300_learning_rate0.02_num_leaves7_max_depth3 | lgb | prob | 0.822896 | 0.525752 | 0.674324 | 0.525752 |
| lgb_n_estimators150_learning_rate0.05_num_leaves7_max_depth3 | lgb | full | 0.825779 | 0.525355 | 0.675567 | 0.525355 |
| member:hgb183 | member | raw | 0.830686 | 0.525151 | 0.677919 | 0.525151 |
| lgb_n_estimators600_learning_rate0.01_num_leaves15_max_depth4 | lgb | full | 0.824659 | 0.525019 | 0.674839 | 0.525019 |

- best weighted-average variant: `nnls` (prob) fold-3 0.527236, fold-2 0.830769
- best stack head: `lr_C0.1` (full) fold-3 0.527728, fold-2 0.829803
- **promotion decision: PROMOTE STACK** (rule: stack fold-3 must exceed blend fold-3 and stack fold-2 must stay within 0.001 of blend fold-2)
- pool selection: winner = `core` (core, 4 members) out of ['core', 'wide']
    - `core`: f3=0.527728, f2=0.829803, members=['graphlgb', 'hgb183', 'lgb183', 'xgb183']
    - `wide`: f3=0.528538, f2=0.832480, members=['graphlgb', 'hgb183', 'hgb600_gate', 'lgb183', 'megacb_gate', 'megalgb_gate', 'xgb183', 'xgb183_gate']
- final head coefficients (first 4 entries are the member probabilities): `[0.2032, -0.0313, -0.0182, 0.1669]` (intercept -5.1307)

## 4. Final submission

- **file: `D:\Desktop\React datathon\submission_file\v7_final_submission.csv`**
- workspace alias: `D:\Desktop\React datathon\submission_v7_final.csv`
- produced by: pool `core`, head `lr_C0.1`
- test score mean 0.017882, range [0.002317, 0.999110]

| check | result |
|---|---|
| row_count | True |
| columns_exact | True |
| id_order_matches_sample | True |
| no_duplicates | True |
| no_missing | True |
| finite | True |
| in_range_0_1 | True |
| roundtrip_precision | True |
| valid | True |

## 5. Transfer-risk evidence (does the fold gain survive on test?)

## 6. Honest caveats

- **No leaderboard feedback.** Every number here is a local chronological AP; the public LB is unknown and is not claimed.
- **HGB183 recovery is a re-derivation, not a download.** The Colab worker was not reachable, so the artifact was rebuilt locally from the documented recipe and validated against the recorded fold-3 AP. Small deltas versus the recorded score reflect environment/frame differences, not a different model family.
- **Legacy artifacts mix feature variants.** The pre-existing gate fold predictions were produced on a 215-column NA-indicator frame while several test predictions came from 124/183-feature frames. The V7 members remove that inconsistency; legacy members are still listed in the candidate table for context.
- **Fold-2 runs hot (~0.83 AP) for every model**, so fold-2 differences are compressed and are used only as a stability guard, never as the selection criterion.
- **Iteration counts are selected on an internal past-only slice** (rows 553012:613012), never on an evaluation fold.
- Graph V3 is a genuine new signal family and passes the leakage probe, but its marginal value is measured through the GraphLGB gate, not assumed.

## 7. V8 audit note (post-hoc correction, no resubmission)

- `v7_stack.to_prob` applied `expit` to NNLS/equal-head outputs that were already
  calibrated probabilities (fixed 2026-09-15 in `v7_stack.py:is_probabilistic`;
  corrected audit-only vectors in `v8_out/v8_wide_nnls_CORRECTED_test.npy` and
  the fixed `v8_blend_*` files). Fold APs are unaffected (expit is monotone) and
  the champion (`lr`) never passed through expit, so the final submission and all
  LOFO numbers are unchanged.
- Corrected wide-pool NNLS test vector: mean 0.015632, range [0.000422, 0.999977]
  (was 0.503784 corrupted). The wide pool still stays rejected by the coherence
  rule (legacy fold/test frame mismatch), only the stated reason is corrected.
- Corrected coherent-pool blends (honest LOFO): nnls_core f3=0.527236 /
  f2=0.830769, equal f3=0.526919, top3 f3=0.527393 -> all lose to the champion
  f3=0.527728 (verdict KEEP CHAMPION, `v8_out/v8_conservative_blends.json`).
- Post-hoc C grid (not in the V7 candidate set): lr_C0.01 f3=0.528149 (+0.000421,
  f2 within tolerance, test Spearman 0.962 vs champion). No LB evidence exists,
  so it stays an alternate-only observation, not a promotion.
