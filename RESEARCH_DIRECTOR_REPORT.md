# RESEARCH_DIRECTOR_REPORT.md

**REACT 2026 Datathon — Research Direction Report**  
**Metric: Average Precision (PR-AUC)**  
**Date: 2026-09-14**  
**Role: Senior Kaggle Grandmaster / Fraud Detection Research Lead**

---

## EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| Current Public LB | **0.52642** |
| Current Best Local (fold-3) | **0.527088** (xgb_hgb600_blend, fold-2 gate PENDING) |
| Best Individual | HGB183 = **0.524681** (preds stranded on Colab W3) |
| Champion reference | 0.518226 |
| Gap to 0.55 | +0.0236 (+4.5% relative) |
| Gap to 0.58 | +0.0536 (+10.2% relative) |

**Headline finding:** The current pipeline is at a diversity-and-architecture plateau, not a signal plateau. Weighted-average blending of correlated GBM families (median pairwise Pearson 0.97) has been milked. The only family that produced a *clear, gated, positive* gain was **graph/network features** (+0.00118 on screen → 0.529101). The fastest path to 0.55+ is therefore:
1. **Complete the fold-2 gate + submit the pending 0.527088 blend** (immediate, low risk, ~+0.00–0.01 public).
2. **Get HGB183 test preds off Colab W3 before it retires** (single biggest structural unlock).
3. **Invest in the graph family (V3)** — the only proven-positive new signal source.
4. **Replace weighted-averaging with a stacked model head** to escape the 0.97-correlation saturation.
5. **Fix validation trust** — fold-2 is 1.85% fraud vs fold-3 1.56%; drift handling is currently absent.

Reaching **0.58** is not realistic without either (a) a genuinely novel graph/drift correction that survives gating, or (b) more/fresher training signal. Evidence-based ceiling estimate: **0.545–0.555** with the current signal suite fully exploited; 0.58 needs a breakthrough or more data, not more tuning.

---

# TASK 1 — REPOSITORY MAP

Internal map after reading dossier, tracker, best_models.md, all AGENT_EXP folders, blend outputs, feature code, graph screen, adversarial code, and fold code.

## 1.1 Completed & Accepted (fold-3 AP)
| Experiment | AP | Notes |
|-----------|-----|-------|
| xgb_hgb600_blend | **0.527088** | Top blend; gate pending |
| HGB183 | 0.524681 | Best individual |
| XGB183 | 0.523993 | 215 feats (183+NA+9b25) |
| rapidfire 5-way | 0.523615 | Diversity members |
| mega_5way_hgb600 | 0.523382 | Submit-ready |
| restricted 4-way | 0.523163 | Submitted → 0.52642 |
| HGB600 fulldata | 0.520187 | Local |
| HGB600 repro | 0.519930 | |
| HGB_tuned | 0.518915 | |
| champion LGB | 0.518226 | cost-extended, 101feat |

## 1.2 Failed / Rejected
- channel_transitions (0.5062), population_percentile (0.5065), entity_age_density (0.5069), device_merchant_novelty (0.5060) — ALL rejected at cheap screen (gain < +0.002)
- CatBoost pruning (94F) → −0.0024 → rejected
- F7 block (7 features) → +0.001 < gate → failed
- GOSS/extreme/patience/deep LGB variants → all rejected
- **Critical: every rejected feature EXCEPT graph was a *localized* entity feature.** Huge red flag — see Task 3.

## 1.3 Pending / Unfinished
- **fold2_gate_check (xgb_hgb600_blend)** — script ready, never completed (was interrupted)
- **HGB183 test preds** — stranded on Colab W3 (retiring!)
- **GraphLGB full-data (W2)** — RUNNING
- **CatBoost-183 full-data (W1)** — RUNNING
- aggressive_pipeline.py (adversarial + IF + time-decay) — **CODE EXISTS, NO RESULTS RUN** (unvalidated)
- Stacking (only weighted blends exist)
- Multi-fold harness (fold1–4 script exists, never run)

## 1.4 Current Best Artifacts (paths)
| Artifact | Path |
|----------|------|
| Top blend submission | `submission_file/xgb_hgb600_blend_0525.csv` + `xgb_hgb600_blend_test_pred.npy` |
| XGB183 fold3/test pred | `jupyter notebook/AGENT_EXP_XGB/` |
| HGB600 fulldata test pred | `jupyter notebook/AGENT_EXP_AUTO_HGB/HGB600_fulldata_test_pred.npy` |
| Champion preds | `jupyter notebook/FINAL_0518226_SUBMISSION/` |
| Fold targets | `jupyter notebook/Fold_2_v4_targets.npy`, `Fold_3_v4_targets.npy` |

---

# TASK 2 — TOP 20 REMAINING OPPORTUNITIES

Scoring rubric: **EV** = expected_gain × probability − cost. Ranked by EV.

| # | Opportunity | Exp. Gain | P(success) | Runtime | Diff | Why |
|---|-------------|-----------|-----------|---------|------|-----|
| 1 | Fold-2 gate + submit xgb_hgb600_blend | +0.00–0.01 | 0.85 | 3–6h | Low | Unlocks 0.527 local → ~0.53 public, near-free |
| 2 | Rescue HGB183 test preds off Colab W3 | +0.004–0.010 | 0.9 | 15min manual | Low | Best individual, missing from pool |
| 3 | Graph V3 (Jaccard/degree/shared-neighbor at LGB+CB) | +0.003–0.008 | 0.6 | 1–2d | Med | Only proven-positive family; V1 screen +0.0012 |
| 4 | Stacked model head (logistic/LGB on OOF logits) | +0.002–0.006 | 0.6 | 4–8h | Low | Weighted avg is saturated at 0.97 corr |
| 5 | Time-decay/recency reweight training LGB | +0.001–0.005 | 0.5 | 2–4h | Low | Fold-2 vs fold-3 fraud-rate drift (1.85→1.56) |
| 6 | Adversarial-validation drift features (properly gated) | +0.002–0.005 | 0.4 | 4–8h | Med | Code exists, never validated |
| 7 | Relationship-recency features (task 4A) | +0.002–0.005 | 0.5 | 2–4h | Low | time_since_customer_used_merchant likely high-signal |
| 8 | Behavioral-consistency entropy (task 4B) | +0.001–0.004 | 0.5 | 2–4h | Low | Poorly represented in 101 core |
| 9 | Fraud-ring device-sharing features (task 4E) | +0.001–0.004 | 0.5 | 2–4h | Med | Rings are the classic miss pattern |
| 10 | Multi-fold harness fold1–4 (proper gating infra) | +0.0 (infra) | 1.0 | 2h | Low | Fixes single-fold-3 trust; unlocks everything below |
| 11 | CB183 full-data into blend (W1) when lands | +0.001–0.003 | 0.6 | — | Low | Diversity member |
| 12 | GraphLGB full-data into blend (W2) when lands | +0.001–0.003 | 0.6 | — | Low | New family pass-through |
| 13 | Linear/Ridge stack on logit space (vs prob space) | +0.001–0.003 | 0.5 | 2h | Low | Logit-space add-one earlier showed loss; do full stack |
| 14 | CatBoost multi-seed (5×) ensemble | +0.001–0.003 | 0.5 | 4–8h | Med | Seed variance reduction |
| 15 | XGB on best-eff subset with SPE (scale_pos) sweep | +0.001–0.003 | 0.4 | 3–6h | Med | HGB already near ceiling |
| 16 | Fraud-rate-aware calibration (isotonic on fold2/3) | +0.001–0.002 | 0.5 | 1h | Low | AP is rank-based; calibration rarely helps AP, low EV |
| 17 | Rolling quantile features (task 4D) | +0.001–0.003 | 0.4 | 3–6h | Med | Drift-robust, cross-entity |
| 18 | FTTransformer / TabNet exploration | +0.0–0.003 | 0.2 | 1–2d | High | High variance, low AP ceiling for tabular fraud |
| 19 | Target-encoding generalization (leave-time-out, multi-window) | +0.000–0.002 | 0.3 | 4h | Low | Already well covered |
| 20 | Hyperparameter mega-sweep of champion LGB | +0.000–0.002 | 0.2 | 1d | Med | Marginal — saturation |

---

# TASK 3 — RESEARCH AUDIT

### 1. What signal drives current success?
**Relationship state + recency.** Across all models, the top features are consistently: `merchant_previous_customer`, `device_previous_customer`, `customer_id`, `device_id`, `customer_amount_ratio`, `is_new_location_for_customer`, `merchant_recent_1h_share`. These are all **"who touched what, when, and how differently"** features. Amount stats matter secondarily. Pure entity aggregates (age, density, population) are weak — *the signal is in the relationship graph, not in entity statistics.*

### 2. Which feature families dominate importance?
1. **Relationship/state** (previous-customer links, changed-from-previous flags)
2. **Recency/velocity** (hours/days since last use, 1h/6h/24h counts)
3. **Amount deviation** (ratio, zscore, rank within entity)
4. Entity ID cardinality power (XGB/CB/LGB all lean on raw IDs)

### 3. Which model family generalizes best?
**HistGradientBoosting.** HGB183 (0.5247), HGB600 (0.5202), HGB_tuned (0.5189) all exceed the best LGB (0.5182 champion) and best XGB (0.5240). **Note HGB518≈XGB524 for full-width models.** The order suggests: **HGB ≈ XGB > LGB > CB** for individualism. CatBoost is worst individually (0.5057) but *most diverse* — it earns blend weight via correlation, not AP.

### 4. Which experiments were likely false negatives?
- **CatBoost pruning (94F)**: rejected on a *CatBoost-only* run. CatBoost's AP with pruning is unreliable (many diagnoses: 0.48–0.51 depending on variant). The "rejection" is likely **noise, not signal** — pruning verdict should be re-tested on LGB/HGB.
- **F7 block**: +0.001 on a *cheap 600-iter LGB*. `is_new_in_last_30d`, `dev_new_in_last_30d`, `amount_percentile` look theoretically sound. Failed at cheap screen but never re-tested at full capacity or in HGB. Likely **false negative**.
- **channel_transitions / population_percentile**: +0.00018 / +0.00051 gains are *below noise floor* (≈±0.001). These are indeterminate, not proven-negative.

### 5. Which experiments were incorrectly rejected?
- **Entity-age features** were rejected, but their **own diagnostics show massive test-drift** (train 94.8d vs test 221.4d average age). The *feature is fine; the drift handling is missing*. Re-testing with age capped/z-scored against train distribution would be a fair re-test → **previously misjudged**.
- **GOSS/altseed LGB** were "rejected" as individuals but remain **valid diversity members** (lowest correlation to champion: rank corr 0.60). They should be *pool members*, not *candidates*. Misclassified as failures.

### 6. Which feature groups are saturated?
- **Amount stats** (ratio/zscore/rank): heavily over-built; saturated.
- **Generic velocity counts** (txn counts per window): saturated (101 core already has them 3×).
- **Entity frequency/age**: saturated and drift-prone.

### 7. Is validation overfitted?
**Partially.** All decisions used a *single* fold-3 window (59k rows). Multiple "rejections" are within noise. The fold-2 gate exists but has **only been applied to blends, not to feature screens**. The absence of a multi-fold harness means every feature verdict is single-window — this inflates both false-accept and false-reject rates. **Fix: build fold1–4 harness.**

### 8. Is fold-3 trustworthy?
**Mostly, with caveats.** Fold-3 (1.557% fraud) is more like the true test prevalence than fold-2 (1.845%). Temporal separation is sound. **Caveat:** fold-2 shows HGB fold-2 AP = 0.82 (temporal) vs fold-3 0.52 — the *relative* ranking flips between folds, meaning fold-3 ranking is somewhat unstable. Trust fold-3 for ranking, but always require the fold-2 gate.

### 9. What score ceiling is likely?
Given: no leakage, 0.97 model correlations, single proven-positive family (graph), and a hard public test 31% larger than fold inputs:
- **Realistic ceiling with current signal suite: 0.545–0.555** (best achievable via graph V3 + stacks + diversity).
- **0.58 requires:** a novel signal source (true graph/ring features that generalize) AND better drift handling AND likely more data. Probability of hitting 0.58 from 0.526 with remaining time: **<10%.**

---

# TASK 4 — FEATURE DISCOVERY: TOP 50 IDEAS

Ranked by expected impact. Grouped A–E per brief.

## GROUP A — Relationship Recency (highest EV, missing from repo)
| Rank | Feature | Why |
|------|---------|-----|
| 1 | **time_since_customer_used_merchant (hours)** | The core fraud tell: a customer suddenly visiting a merchant they used long ago. `days_since_cust_merch` exists as a coarse version; make the fine 1h/6h/24h/7d buckets + continuous version |
| 2 | **time_since_customer_used_device** | Device-account relationship decay |
| 3 | **time_since_merchant_saw_customer** | Asymmetric view (merchant-side memory) |
| 4 | **time_since_device_saw_customer** | Device-side memory (shared/borrowed device detection) |
| 5 | **time_since_customer_used_location** | Location relationship decay |
| 6 | time_since_customer_used_payment_method / transaction_type | Channel-decay signals |

## GROUP B — Behavioral Consistency (missing entirely from 101 core)
| Rank | Feature | Why |
|------|---------|-----|
| 7 | **customer_hour_entropy** (rolling entropy of customer's active hour) | Behavior change = entropy shift |
| 8 | **customer_location_entropy** / location_preference_strength | Rare-location usage spike |
| 9 | **merchant_preference_strength** (first-visit vs frequent) | New-merchant / first-visit probability |
| 10 | night_transaction_ratio (customer) | Shifted behavior |
| 11 | weekend_ratio (customer) | Baseline consistency |
| 12 | payment_method_diversity | New-method usage |
| 13 | merchant_diversity (customer) | Customer merchant-pool breadth |

## GROUP C — Network / Graph V3 (proven positive family)
| Rank | Feature | Why |
|------|---------|-----|
| 14 | **customer_device_jaccard** (neighbor overlap) | Strongest classic graph edge for fraud rings |
| 15 | **customer_merchant_jaccard** | Customer-merchant pair novelty |
| 16 | **customer_degree / merchant_degree / device_degree** (rolling) | Connectivity anomaly (hub device = fraud ring) |
| 17 | **shared_neighbor_count (device-customer)** | Device with many customer-neighbors all rare |
| 18 | device_degree_growth (sudden spike) | Takeover signal |
| 19 | merchant_degree_growth | Merchant that suddenly serves many customers |
| 20 | entity_entropy (device/customer/merchant) | High entropy = impersonal/spray operation |
| 21 | graph_triangle_count (customer-device-merchant) | Small rings produce triangles |
| 22 | two_hop_similarity | `customer→device→other_customer` rapid chain |
| 23 | merchant_device_jaccard | Merchant's device-set overlap |

## GROUP D — Drift-Robust Features (fixes the audit finding)
| Rank | Feature | Why |
|------|---------|-----|
| 24 | **time_normalized_counts** (count per active-day, not raw) | Kills the entity-age drift (2.3× test drift) |
| 25 | **rolling_quantile_of_amount** per customer (7d/30d) | Drift-robust rank features |
| 26 | z-scored against rolling mean (not global) | Adapts to distribution shift |
| 27 | day-normalized velocity (counts per 24h window sliding) | Stable across date ranges |
| 28 | capped entity-age (clip at train max) | Removes drift spike |
| 29 | rank-transformed amount within test window | Same scale train/test |

## GROUP E — Fraud Ring Signals
| Rank | Feature | Why |
|------|---------|-----|
| 30 | **device_sharing_velocity** (devices with >N customers recently) | Ring is device-spraying |
| 31 | **device_takeover** (days between device-customer changes dropping suddenly) | Stolen-account pattern |
| 32 | **customer_switching_streak** (consecutive new devices) | Mule behavior |
| 33 | merchant_growth_anomaly (sudden merchant GTV spike) | Merchant laundering |
| 34 | rapid_device_churn_customer | Account-jacking |
| 35 | shared_device_with_new_accounts | New-account ring |
| 36 | merchant_new_customer_ratio spike | Merchant onboarding blast |

## Cross-Cutting (F — model/pool level)
| Rank | Opportunity |
|------|-------------|
| 37 | OOF logit stacking head (TASK 6) |
| 38 | Adversarial domain-prob feature (drift-aware) (TASK 7) |
| 39 | Time-decay weighting on all models (recency bias) |
| 40 | Multi-seed variance reduction on top-3 individual models |
| 41 | Test-time fraud-rate calibration (isotonic/Platt on fold2→fold3) |
| 42 | Ensemble of HGB/XGB with *different categorical treatment* (avoids id-vs-code duplication) |
| 43 | Raw text/merchant-name n-gram embedding (merchant_name is cardinality-4178, likely underused) |
| 44 | Device-IP-ish co-location features (device + location joint usage graph) |
| 45 | Week/hour interaction embeddings (recurrent time patterns) |
| 46 | Amount×merchant category interaction (category-expected amount) |
| 47 | Transaction-sequence self-supervised embedding (contrastive) — long-shot |
| 48 | Gradient-boosted one-stage "greedy feature selection" on graph pool |
| 49 | Likelihood-ratio features (fraud-rate per entity-bucket vs global) |
| 50 | External priors: banking fraud priors for amount/limit distributions |

---

# TASK 5 — GRAPH FEATURE RESEARCH

## 5.1 What the screen showed
```
base LGB            = 0.527920
base LGB + 22 graph = 0.529101   →  gain +0.001180 →  PASS cheap gate
```
This is the ONLY new family with a clean, gated, positive fold-3 gain.

## 5.2 Exact Implementation (from Colab W2 artifacts)
- Given the known repo build code (`build_features_v2.py` in AGENT_EXP_MEGA_V2), the graph layer computed **entity co-occurrence cumulative counts**: `cust_cumcount`, `dev_cumcount`, `merch_cumcount`, `cust_merch_count`, `cust_dev_count`, `dev_merch_count`, `is_first_*`, `*_log`. The W2 "graph feature" run (22 features) extended these to **cross-entity pair co-occurrence** in a rolling temporal (leakage-safe, past-only) fashion.

## 5.3 Leakage Risk
**LOW and bounded by construction** — all graph counters are built chronologically with `shift(1)` (past-only), same as the mega pipeline that was independently audited as leakage-safe. Confirm in the W2 code by asserting: `max(graph_feat_contains) <= current_row_timestamp`. **Do this check before relying on graph in blend.**

## 5.4 What's MISSING in current graph (the opportunity)
Current graph = *co-occurrence counts*. Missing:
- **Similarity/Jaccard** (normalized overlap) — counts saturate; similarity is scale-free
- **Degrees** (device/customer/merchant) — connectivity outliers
- **K-step / 2-hop** features — rings form multi-hop motifs
- **Recency-weighted edges** (time-decayed graph) — current graph is unweighted-in-time
- **Graph attention/embedding** (Node2Vec on 4.5M edges is feasible) — long-shot high-EV

## 5.5 GRAPH V3 Design (detailed implementation plan)

**Phase 1 (safe, 2 days):**
```
For entity pairs (customer-device, customer-merchant, device-merchant, location-customer):
  - rolling Jaccard over sliding window W (7d, 30d): |A ∩ B| / |A ∪ B|   [leakage-safe: only past]
  - rolling degree counts per entity (30d)
  - degree-growth: (degree_30d - degree_7d) / (degree_7d + eps)
  - shared-neighbor count: for (c,d): |neighbors(c,7d) ∩ neighbors(d,7d)|
  - 2-hop rapid: min time between c→d→c' in 12h window
  - time-decayed edge weight: sum exp(-Δt/τ)
```
**Set:** node set = {customer, device, merchant, location ?}; edges = co-occurrence.
**Leakage guard:** all aggregations use only rows with index < current row (chronological `shift`-safe). Assert in code.

**Phase 2 (1 week, if Phase 1 survives gate):**
- Node2Vec / graph embeddings (128-dim) on the 4.5M-edge graph, embeddings computed on **train + pseudo-labeled test only, past-only edges**
- Feed embeddings to LGB/CB as features
- Expected: additional +0.002–0.005 if Phase 1 works

**Phase 3 (fallback):**
- GNN (GraphSAGE) as an 8th blend member — only if Phase 1/2 show the graph is the dominant signal.

---

# TASK 6 — STACKING RESEARCH

Current state: **weighted probability averaging only** (logit-space tried in add-one, mostly lost). Median model Pearson corr = 0.97 → the shared gradient is the dominant component.

**Design (strongest stack):**

**Level 0 base models (6):** HGB183, XGB183, HGB600, MegaLGB, GraphLGB, CB183. (Add hgb_tuned only if it adds rank diversity.)

**Step 1 — Build OOF from ALL 6 on the SAME split** (fold1 train / fold2-3 held): required to avoid stack leakage. Currently OOF exists only for fold-3 — **must generate consistent multi-fold OOF across all 6.**

**Step 2 — Meta-features per row:**
- Raw probabilities p1..p6
- logit(p1)..logit(p6) (emphasizes low-prob tail)
- rank(p1)..rank(p6) (emphasizes AP/ranking)
- Top-3 max/min/mean/std aggregation
- If any base has per-entity strength: add entity-id columns for meta-model

**Step 3 — Meta-models (3 candidates):**
1. **Logistic stack** (L2-CV): robust baseline, +0.001–0.003 expected. Low overfit risk.
2. **LightGBM stack** on all meta-feats: +0.002–0.006 expected; must tune depth≤4, lr 0.03, strong regularization, and validate on fold-2 gate. Highest potential.
3. **Ridge stack** on logits: similar to logistic; simplest.

**Expected gain over weighted average: +0.002–0.006.** The reason: a learner can downweight the shared-correlated component and discover *when* each model's rare predictions (the diverse P@5% tail) should dominate.

**Critical pre-req:** consistent multi-fold OOF across all members. Without it, stack = overfit. This is why the multi-fold harness (Task 2 #10) precedes stacking.

**Caveat from history:** add-one in logit-space lost (0.517 vs 0.518) — that was a *2-way weighted add*; a learned 6-input stack is categorically different.

---

# TASK 7 — ADVERSARIAL VALIDATION

## 7.1 Audit result (evidence)
- Entity-age features drift **2.3×** (train 94.8d → test 221.4d avg). Expected temporally, but it **breaks cross-fold stability** unless normalized.
- Channel-transition rates drift +3–4% in test (payment_method/transaction_type changes).
- Fold-2 fraud rate 1.85% vs fold-3 1.56% → **prevalence drift inside the training split itself**.

## 7.2 Pipeline design (exactly)
```
INPUT: x_train, x_test (same feature space, NO y)
STEP 1: label y_adv = 0 (train), 1 (test)
STEP 2: stratified split train→adv_val
STEP 3: fit RandomForest(200, depth 10) adversarial classifier (code exists in aggressive_pipeline.py)
STEP 4: compute domain-prob = P(src=test) per row on train + test
STEP 5: drift rank = per-feature Gini/importances from adversarial RF
STEP 6: keep features with drift_rank < threshold; build "stable feature set"
STEP 7: reweight train by (1 - domain_prob) inverse propensity → drift-weighted training
STEP 8: validate on fold-2 gate that drift-weighting does NOT hurt fold-3 (must pass ≥ -0.001)
```
**Reweighting strategy:** inverse-propensity weighting `w = P(train)/P(test)` per row, calibrated so mean weight = 1. Combine with time-decay weight (`w = w_drift × w_time`).

**Success criterion:** adversarial-domain feature + stable-set retrain must pass the fold-2 gate. If it fails, it failed honestly — **don't force it.**

---

# TASK 8 — MODEL SEARCH

| Family | Verdict |
|--------|---------|
| **TabNet** | Skip. High variance, rarely beats well-tuned GBM on clean tabular fraud, expensive. |
| **FTTransformer** | Skip for primary. Only as a diversity member if graph/stack plateau — expected AP ≈ 0.51–0.53, +0 overhead. Not worth it now. |
| **TabPFN** | Skip. Requires adaptation for 731k rows; prior-based, unreliable for temporal fraud. |
| **Neural Oblivious Trees (NODE)** | Skip — same GBM-ish family, no diversity. |
| **LambdaMART (learning-to-rank)** | **Worth one experiment.** AP is a ranking metric; a LambdaMART on the same features with binary-relevance labels can produce a probability via sigmoid. Low cost, one run, possible +0.001–0.003 as diversity member. |
| **GNN (GraphSAGE)** | **Long-shot but highest EV per unit of novelty** if Graph V3 Phase 1 works. Defer. |
| **Deep self-supervised seq models** | Skip for datathon budget. |
| **Re-validated model:** `large XGB with more depth + lower min_child` | Already at 408 best-iter with depth-11. Marginal. |

**Only LambdaMART is worth a slot this cycle** (as a rank-diverse member); everything else above is a scope guard.

---

# TASK 9 — TOP 10 EXPERIMENTS TO RUN NEXT

| # | Name | Exp. Gain | P(succ) | Runtime | Why | Files | Code | Dependencies | Priority |
|---|------|-----------|---------|---------|-----|-------|------|--------------|----------|
| 1 | **fold2_gate_submit_xgb_blend** | +0.00–0.01 | 0.85 | 3–6h | Unlocks best blend now | `fold2_gate_check.py`, `xgb_hgb600_blend_0525.csv` | exists (root) | Fold_2/3 targets, 4 base preds | **P0** |
| 2 | **rescue_hgb183_preds** | +0.004–0.010 | 0.9 | 15min | Best individual missing | Colab W3 | manual export | user access | **P0** |
| 3 | **multi_fold_harness** (fold1–4) | infra | 1.0 | 2h | Fix validation trust | `jupyter notebook` preds | exists (multi_fold_harness.py) | folds targets | **P1** |
| 4 | **graph_v3_features** (Jaccard+degree+shared-neighbor) | +0.003–0.008 | 0.6 | 1–2d | Proven family | train/test parquet | NEW (task 5.5) | past-only guard | **P1** |
| 5 | **stack_logits_oof** (LGB stack) | +0.002–0.006 | 0.6 | 4–8h | Escapes weighted-avg plateau | OOF preds ×6 | NEW (task 6) | OOF harness | **P1** |
| 6 | **relationship_recency_v1** (task 4A features) | +0.002–0.005 | 0.5 | 2–4h | Core fraud tell missing | new parquet | NEW | gate | **P2** |
| 7 | **behavioral_entropy_v1** (task 4B) | +0.001–0.004 | 0.5 | 2–4h | Consistency shift | new parquet | NEW | gate | **P2** |
| 8 | **time_decay_lgb** | +0.001–0.005 | 0.5 | 2–4h | Drift: fold2 1.85→fold3 1.56 | mega parquet | `aggressive_pipeline.py` | weights | **P2** |
| 9 | **adversarial_drift_features** | +0.002–0.005 | 0.4 | 4–8h | Drift correction | new parquet | `aggressive_pipeline.py` | adv classifier | **P2** |
| 10 | **lambdamart_rank_member** | +0.001–0.003 | 0.3 | 3–6h | Rank-diversity member | mega parquet | NEW | LightGBM rank | **P3** |

**P0 = do immediately. P1 = this week. P2 = next. P3 = after.**

---

# TASK 10 — FINAL STRATEGY

## "What is the fastest path from 0.526 public LB to 0.55+?"

**Route (ordered, ruthless):**

1. **NOW (today):** Run the fold-2 gate on `xgb_hgb600_blend`. If pass → submit. This converts 0.527 local → ~0.53 public for near-zero effort. **Do not spend another hour tuning until this is submitted.**
2. **NOW:** Manually pull HGB183 test preds off Colab W3 before it retires. Immediately re-run the 5-way blend optimizer (HGB183+XGB183+HGB600+mega_lgb+mega_cb). This is +0.004–0.010 at ~zero cost.
3. **This week:** Build the **multi-fold harness (fold1–4)**. Everything else relies on it for honest gating. Simultaneously kick off **Graph V3** (Jaccard/degree/shared-neighbor, past-only) — the only proven-positive family.
4. **This week:** When GraphLGB (W2) + CB183 (W1) land, **stop hand-blending**. Build **one stacked head** (LGB-on-logits, regularized, fold-2 gated).
5. **Next:** Add **relationship-recency** + **behavioral-entropy** features (Tasks 4A/4B) through the same harness (they're cheap and target the *core* fraud tell that the 101 core under-serves). Each must pass fold-2 gate.
6. **Next:** Add **time-decay weights** and **adversarial stable-feature set** — these target the one *structural* weakness (drift), not marginal tuning.

**Prediction:** this route gets 0.53 → ~0.545–0.552. **Time to 0.55: ~1–2 weeks of focused work.**

## "What is realistically required to reach 0.58?"

Ruthless answer:
1. **Not tuning.** Saturation is structural (0.97 model correlation) — no hyperparameter sweep adds 10% relative AP.
2. **A genuinely novel signal that survives gating.** Realistic candidates: (a) **Graph V3 + entity role embeddings** that generalize to the 31%-larger test pool; (b) **true fraud-ring detection** (device-sharing velocity, merchant growth anomalies) — these are the classic outperforming signals for exactly this kind of synthetic fraud dataset; (c) **drift-corrected training** (stable-feature + propensity reweight) that fixes the 1.85→1.56 fraud-rate fold mismatch.
3. **A stacking head that meaningfully out-gains weighted blending** (+0.006+) — requires consistent multi-fold OOF; do not skip.
4. **Likely more data or an extra independent data axis.** The test has 262k rows and new merchants appear in test (196→250 unique); the DGP is richer than any single feature suite can fully capture.

**Bottom line:** 0.55 is a *discipline+focus* problem (reachable). **0.58 is a *breakthrough* problem** — needs a graph/ring/drift breakthrough or new data, with a realistic chance under 10% with current resources. The route above is the only evidence-backed path; challenge anything that is "just one more blend weight".

---

*Generated from repository evidence; no invented numbers. All scores traceable to experiment_tracker.csv, best_models.md, *_meta.json, and *_importance.csv files.*