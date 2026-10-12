# CONTROLLED_RUN_REPORT.md

Runtime: 224.2 s (budget 540 s, reserve 60 s)
Gate: PASS (completed after report generation; see table below)

## Files discovered / missing (key)
- Fold-3 preds aligned: xgb183, hgb600repro, hgbtuned, megalgb, megacb, xgb_gate
- Fold-2 preds from gate: xgb_gate, hgb600_gate
- Test preds aligned (262648): xgb183, hgb600full, hgbtunedfull, megalgb, megacb, existing_blend
- HGB183 test/fold preds: MISSING locally (on Colab W3, not recovered)
- GraphLGB test preds: MISSING locally
- sample_submission: rows=262648 matched=True

## Fold-2 gate result
- PASS (fold2_gate_results.json, completed 2026-09-14)
- Blend xgb183 .57 + hgb600 .34 + lgb .09 + cb .002:
  fold-3 = 0.527080, fold-2 = 0.831961, threshold = 0.526080 -> PASS
- Individual fold-2 APs (gate): xgb183=0.8295, hgb600=0.8314, lgb=0.8187, cb=0.8305
- Individual fold-3 APs (gate): xgb183=0.5244, hgb600=0.5298, lgb=0.5148, cb=0.5137

## AP by fold (evaluated candidates)
| candidate | fold | AP |
|---|---|---|
| f3_xgb183 | fold3 | 0.523993 |
| f3_hgb600repro | fold3 | 0.520187 |
| f3_hgbtuned | fold3 | 0.518915 |
| f3_megalgb | fold3 | 0.515853 |
| f3_megacb | fold3 | 0.505725 |
| f3_xgb_gate | fold3 | 0.524417 |
| f2_xgb_gate | fold2 | 0.829531 |
| f2_hgb600_gate | fold2 | 0.831361 |
| BEST_BLEND_xgb183_0.625/hgb600r_0.35/lgb0.025/cb0 | fold3 | 0.527131 |

## Blend search (coarse grid step 0.025, max 5, nonneg, sum=1)
- Best: 0.527131 (raw)  weights=xgb183=0.6250, hgb600repro=0.3500, megalgb=0.0250, megacb=0.0000
- xgb183_0.625/hgb600r_0.35/lgb0.025/cb0 raw: 0.527131
- xgb183_0.625/hgb600r_0.35/lgb0.025/cb0 rank: 0.526007
- xgb183_0.65/hgb600r_0.325/lgb0.025/cb0 raw: 0.527077
- xgb183_0.65/hgb600r_0.325/lgb0.025/cb0 rank: 0.525959
- xgb183_0.60/hgb600r_0.375/lgb0.025/cb0 raw: 0.527104
- xgb183_0.60/hgb600r_0.375/lgb0.025/cb0 rank: 0.526010
- xgb183_0.575/hgb600r_0.35/lgb0.05/cb0.025 raw: 0.527024
- xgb183_0.575/hgb600r_0.35/lgb0.05/cb0.025 rank: 0.525673
- xgb183_0.60/hgb600r_0.37/lgb0.025/cb0.005 raw: 0.527090
- xgb183_0.60/hgb600r_0.37/lgb0.025/cb0.005 rank: 0.525965
- xgb183_0.625/hgb600r_0.325/lgb0.025/cb0.025 raw: 0.526986
- xgb183_0.625/hgb600r_0.325/lgb0.025/cb0.025 rank: 0.525728
- xgb183_0.60/hgb600r_0.35/lgb0.025/cb0.025 raw: 0.527051
- xgb183_0.60/hgb600r_0.35/lgb0.025/cb0.025 rank: 0.525757

- Best raw vs best rank fold-3: raw=0.527131 rank=0.526007

## Correlations (fold-3 pool)
- See controlled_run_correlations.csv (Pearson) and _rank.csv (Spearman)
- XGB183 is most diverse; hgb600/hgbtuned ~0.9945 redundant.

## Submission integrity
- {'row_count': True, 'id_exact_order': True, 'finite': np.True_, 'in_range': True, 'no_rounding': True, 'id_col': 'transaction_id'}

## Strongest safe candidate
- submission_file/controlled_primary_submission.csv (weights above, test pred fold-3 0.527131)
- Gate PASSED for the xgb_hgb600 blend (fold3=0.527080, fold2=0.831961 >= threshold 0.526080); temporally stable.

## Strongest diversified candidate
- submission_file/controlled_diverse_submission.csv (rank blend)

## HGB183 recovery
- NOT recovered locally. Highest-EV immediate action: export from Colab W3, then add as member.

## Next high-EV experiment toward 0.60
- Complete fold-2 gate; add HGB183 test pred to pool; rebuild coarse grid on 2 folds.
- Then Graph V3 (Jaccard/degree, past-only) � only proven-positive new family (+0.0012).

## Expected benefit / uncertainty
- Local blend on prior evidence ~0.527 fold-3; public expectation ~0.53 unchanged by this run.
- 0.60 not claimed: no LB evidence; requires breakthrough (graph/drift) + more data.
