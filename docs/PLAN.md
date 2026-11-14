# PLAN - PhonePe Banking Decision Intelligence (Capstone, 100 marks)

Phase: PLAN only. No design or implementation here. Source: `docs/REQUIREMENTS.md`.
Window: 6 Nov to 16 Nov 2026 (11 days). Date at time of writing: 14 Nov 2026, so only about 3 days remain; the schedule below is compressed accordingly (see section 7).

## 1. Summary of the brief

One public GitHub repo, one branch `main`, text-only deliverables, graded by reading committed `outputs/` files against reference numbers (graders do not run code). Synthetic data (800 rows, 10 columns, 170 positives) from a provided generator, committed verbatim. Everything is pure Python 3.10+ and NumPy; every graded number must come from our own code (no ready-made regression, KNN, k-means, PCA or metric routines).

- Part 1 (25): estimate/test mean monthly spend, sample-size sizing, closed-form and gradient-descent regression, funnel check and log remedy, multiple regression by normal equation.
- Part 2 (30): stratified 70/30 split, logistic regression from scratch, learning-rate effect, odds ratios, threshold evaluation, KNN with 5-fold CV, cost-based threshold (INR 50 per false alarm, INR 400 per miss).
- Part 3 (25): K-means from scratch, elbow, segment profiling, hierarchical clustering on a supplied 6x6 table, PCA in seven steps.
- Part 4 (20): entropy/Gini splits, stump + hard/soft voting, two-round AdaBoost by hand, OOB fraction, parameter counting for MLP/CNN/RNN, 300-word memo.

Marks per Part are given in the brief; the per-item split within a Part is not stated (each results key, CSV and README answer is "scored on its own"). We therefore map items to tasks, not to sub-marks.

## 2. Deliverables and rubric mapping

Legend: K = results key group, C = CSV, R = README answer. Part marks: P1 25, P2 30, P3 25, P4 20.

### Repo-level deliverables
| Deliverable | Task |
|---|---|
| `README.md`, line 1 exactly `Domain: Banking`, sections in order: How to run, Part 1-4 findings, Model-choice memo, Limitations (>=3) | T-README |
| `requirements.txt` with `numpy==<version>` (+ pins for extras) | T-ENV |
| `generate_data.py` verbatim; prints `800 10 170 e85e7605b784f4142844f79ad2dd0b8e` | T-DATA |
| `data/records.csv` committed (MD5 must match) | T-DATA |
| `part1.py`..`part4.py` + shared helper (no Part re-run on import) | T-P1..T-P4, T-LIB |
| 13 files in `outputs/` (listed per Part below) | T-P1..T-P4 |
| Pushed to public GitHub, default branch `main`, correct link submitted | T-SUBMIT |

### Part 1 (25) - `outputs/part1_results.txt`, `part1.py`
| Item | Task | Output |
|---|---|---|
| 1.1 n, mean, sd, se, 95% CI (Checkpoint n=800, mean=8.9382) | T1.1 | K: n, mean_value, sd_value, se_value, ci95_low, ci95_high |
| 1.2 z-test vs 8.0 | T1.2 | K: z_stat, reject_h0 |
| 1.3 sample size for E=0.5 | T1.3 | K: n_required (+-1 gets half) |
| 1.4 closed-form regression value~X4, TSS/ESS/RSS | T1.4 | K: reg_slope, reg_intercept, tss, ess, rss, tss_equals_ess_plus_rss, r2, rmse, mae |
| 1.5 gradient descent, eta 0.1, 500 steps | T1.5 | K: gd_J0/J1/J10/J100/J500, gd_slope, gd_intercept, gd_slope_gap (<0.001) |
| 1.6 funnel check + log-log | T1.6 | K: resid_sd_lower/upper/ratio_raw, loglog_intercept/slope/r2/rmse, resid_sd_lower/upper/ratio_log |
| 1.7 normal-equation multiple regression | T1.7 | K: mr_beta_0..7, mr_r2, mr_adj_r2, loglog_adj_r2 |
| R1.1-R1.5 README `## Part 1 findings` | T-README | text quoting own outputs |

### Part 2 (30) - `part2_results.txt`, `part2_thresholds.csv`, `part2_knn_cv.csv`, `part2.py`
| Item | Task | Output |
|---|---|---|
| 2.1 stratified split + train-stat standardisation (Checkpoint 560/119/240/51) | T2.1 | K: train_rows, train_positives, test_rows, test_positives |
| 2.2 logistic regression, 1000 steps | T2.2 | K: loss_epoch_0/1/10/100/1000, weight_X1..X7, bias, test_bce (Checkpoint 0.6931) |
| 2.3 learning-rate effect | T2.3 | K: loss_epoch_200_eta_0.01/0.1/1.0 |
| 2.4 odds ratios | T2.4 | K: odds_ratio_X1..X7 |
| 2.5 evaluation at tau 0.5 and 0.3 | T2.5 | K: 9 keys x 2 taus, blind_accuracy |
| 2.6 threshold sweep 0.1..0.9 | T2.6 | C: tau,tp,fp,tn,fn,tpr,fpr (9 rows) |
| 2.7 KNN + 5-fold CV, K=1..21 odd | T2.7 | C: K,mean_cv_f1 (11 rows); K: knn_best_k, knn_tp/fp/tn/fn/accuracy/precision/recall/f1 |
| 2.8 cost-optimal threshold | T2.8 | K: cost_tau (exact), cost_train_at_tau, cost_test_at_tau, cost_test_at_0.5, cost_test_flag_nobody |
| R2.1-R2.5 README `## Part 2 findings` | T-README | text |

### Part 3 (25) - `part3_results.txt`, `part3_segments.csv`, `part3_hierarchical.csv`, `part3.py`
| Item | Task | Output |
|---|---|---|
| 3.1 K-means K=2..6, pinned init, run to convergence (K=4: 21 passes, K=5: 15, K=6: 27) | T3.1 | K: wcss_k2..k6 (Checkpoint wcss_k2=2262.6753) |
| 3.2 elbow rule, then use K=3 | T3.2 | K: elbow_k |
| 3.3 segments A/B/C ordered by mean X3 | T3.3 | K: size_segment_A/B/C; C: segment,size,mean_X1..X5,outcome_rate,mean_value (3 rows) |
| 3.4 single/complete/average agglomerative on supplied table | T3.4 | C: linkage,merge_no,cluster_a,cluster_b,height (15 rows); K: single/complete/average_clusters_at_cut |
| 3.5 PCA 7 steps, sign convention | T3.5 | K: eigenvalue_sum (Checkpoint 5.0000), eigenvalue_1..5, explained_ratio_1..5, cumulative_ratio_1..5, components_for_80pct, pc1/pc2_loading_X1..X5, row1..3_pc1/pc2, mean_pc1_segment_A/B/C |
| R3.1-R3.4 README `## Part 3 findings` | T-README | text; segment business names + action; PCA step 1 mean vector = zeros stated |

### Part 4 (20) - `part4_results.txt`, `part4_splits.csv`, `part4_voting.csv`, `part4_adaboost.csv`, `part4_architecture.csv`, `part4.py`
| Item | Task | Output |
|---|---|---|
| 4.1 entropy/Gini of 4 splits | T4.1 | K: root_entropy (Checkpoint 0.7462), root_gini, best_split_by_info_gain, best_split_by_gini_gain; C: splits (4 rows) |
| 4.2 stump + 5 models on test | T4.2 | K: stump_rate_yes/no; C: voting (5 rows: logistic, knn, stump, hard_vote, soft_vote) |
| 4.3 AdaBoost 2 rounds | T4.3 | K: round1/2_error, round1/2_alpha, ensemble_accuracy (Checkpoint round1_error=0.2500); C: adaboost (16 rows) |
| 4.4 OOB fraction | T4.4 | K: oob_fraction |
| 4.5 parameter counting | T4.5 | C: architecture (7 rows, fixed order); K: mlp_total, cnn_total, fc_layer_params, conv_layer_params, rnn_total |
| 4.6 memo (<=300 words, `## Model-choice memo`) | T-README | >=3 numbers from outputs, threshold + INR cost, logistic cost comparator, one limitation, one sentence on Model A vs B |
| R4.1-R4.4 README `## Part 4 findings` | T-README | includes per-layer output-shape table for CNN, RNN weight shapes, activations/loss, vanishing gradient + skip connection |

## 3. Constraints and rules from the brief

Repository and submission
- One public repo, branch `main` only; submitted link must be repo root (if `main` is default) or `/tree/main`. Verify default branch before submitting.
- Layout exactly as in the brief. README line 1 exactly `Domain: Banking`; section order fixed.
- Forbidden: any other link in the repo or README; images, screenshots, PDFs, slides, video, audio; notebooks as the means of delivery; paid/account-gated services, API keys, dataset downloads.
- Do not edit `generate_data.py` (verbatim) or hand-edit `data/records.csv`. Check output line `800 10 170 e85e7605b784f4142844f79ad2dd0b8e`.

Code rules
- Python 3.10+, standard library + NumPy only (pin the version used, e.g. 1.26.4 or 2.2.6). Extra packages pinned, and not used for any graded number.
- Forbidden for graded numbers: ready-made logistic regression, KNN, k-means, PCA, metric routines; also `polyfit`, `cov`, `corrcoef`, `lstsq` in Part 1. Allowed: array arithmetic, `numpy.linalg.solve` (1.7), `numpy.linalg.eigh` (3.5), stable `argsort`.
- Never type a reference or checkpoint value; zero for that line.
- Scripts run top to bottom on a clean clone in README order and regenerate outputs exactly. Shared helper file must not re-run a Part on import.
- Later Parts reuse earlier splits/models; no Part uses a number it did not compute or receive from an earlier Part (except constants printed in the brief).
- Original work; author must be able to explain every line.

Formats and conventions
- `key = value`; floats 4 dp, ints as ints, booleans `yes`/`no`; compute at full precision, round only on print. CSV: exact headers, comma separator, floats 4 dp.
- Money as `INR 400` / `400 Rupees`, never a dollar sign.
- Sample SD (n-1) everywhere; ln = natural log; positive predicted when p >= tau; division by zero in a metric gives 0.
- Tolerances: max(0.005, 0.0005 x |ref|); confusion cells +-1; rates +-0.02 (0.03 KNN/voting); CV F1 and outcome_rate +-0.01; PCA loadings/projections +-0.01, mean_pc1 +-0.02; money +-INR 450; `cost_tau` exact; `gd_slope_gap` < 0.001; `n_required` +-1 half marks.
- Pinned tie-breaks: stable sorts (Task 1.6, 2.7), ties to smaller K / smaller tau / lower centroid index; no test-set leakage into fitting, standardising, K or tau.
- README numbers must be quoted from committed outputs. Memo <= 300 words.

Dates
- Window 6-16 Nov 2026 (11 days); submit before the closing time on the course platform's page (time not in the brief; see open questions). Suggested pace: P1 days 1-3, P2 days 4-6, P3 days 7-8, P4 + README days 9-11.

## 4. Assumptions and open questions

Assumptions
- A1. "Today" is 14 Nov 2026; remaining time is roughly 2.5 days, so we run Parts back to back instead of the suggested pace.
- A2. Windows environment; Python 3.10+ must be installed (the `python` command currently resolves to the Microsoft Store shortcut, i.e. no working interpreter on PATH yet; `py` launcher not yet checked).
- A3. The existing local git repo (one commit, "Add project requirements") will be the source for the public repo; branch name must be `main`.
- A4. Task 4.2 uses the Task 2.2 logistic model and Task 2.7 KNN (best K) trained on the 560 rows; stump probabilities use training shares.
- A5. Task 3.3 "outcome_rate" and "mean_value" use the original columns; Part 3 ignores the pinned 560/240 split (all 800 rows).
- A6. For `elbow_k`, the first K in {3,4,5} satisfying the rule is used even if later drops are irregular (brief says the pinned solution is graded).

Open questions
- Q1. Which Python/NumPy versions are available on the machine, and which NumPy pin to use (1.26.4 or 2.2.6 are both verified)?
- Q2. GitHub account/username and repo name; is the repo to be created by the user (we do not push or add remotes unless asked)?
- Q3. Exact submission closing time on the course platform (brief says "the time shown there").
- Q4. Is the local branch already named `main`? If it is `master`, rename before pushing.
- Q5. Should `docs/REQUIREMENTS.md` and `docs/PLAN.md` stay in the submitted repo? Brief lists a layout without `docs/`; extra files are not forbidden, but graders read repo text, so this is a judgment call (default: keep, they contain no links or images).
- Q6. Task 4.2 KNN probability tie-handling for the soft vote: share of positives among K (given), no ambiguity; just confirm no extra convention is needed.
- Q7. Task 2.7 ambiguity: reference rows are "the other four folds"; assumed that file order is preserved within them for the tie-break.

## 5. Risks and mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Very little calendar time left (about 2.5 days to close) | Incomplete parts | Fix scope to the brief only; order work P1->P2->P3->P4->README; commit after each Part; README sections written as each Part completes |
| R2 | No working Python on PATH | Blocks all work | Resolve in first hour (install Python 3.10+, check `py` launcher), create venv, pin NumPy |
| R3 | MD5 mismatch from line endings on Windows (CSV written with `newline=""`; git autocrlf could alter committed file) | Data fails verification | Run generator on the machine, verify hash before and after commit; set `.gitattributes` `data/records.csv -text` and check `git ls-files --eol` |
| R4 | Convention slips (sample vs population SD, unstable sort, wrong tie-break, test leakage) | Numbers outside tolerance | Implement pinned conventions in the helper once; verify against all Checkpoints; test the brief's printed pass counts (K-means 21/15/27 passes) |
| R5 | Accidentally using forbidden library routines | Zero for those lines | Grep for `polyfit|cov|corrcoef|lstsq|sklearn|scipy|pandas` before each commit |
| R6 | Typed reference values | Zero | Never copy numbers into code; checkpoints only compared by eye or in a test that is not part of outputs |
| R7 | Non-reproducible outputs (outputs differ on a clean clone) | Mismatch | Rerun full pipeline from clean clone before submit; diff outputs |
| R8 | K-means with no iteration cap fails to terminate / empty clusters | Hang | Follow pinned rule; add a safeguard that only warns, does not alter the algorithm |
| R9 | README quotes numbers inconsistent with outputs | README answers lose marks | Generate a quote sheet from outputs files; fill README from it |
| R10 | Wrong branch/default branch or extra links/images in repo | Not graded | Final checklist (section 6, M5) |
| R11 | Memo over 300 words or missing required elements | Loses item | Word-count check; checklist of required elements |
| R12 | Cost-optimal model recommendation conflicts with what was computed (Task 4.6 only allows costs we computed) | Memo item | Compute cost for any recommended non-logistic model in part4.py using the 2.8 rule, or state it was not computed |

## 6. Work breakdown and milestones (lifecycle)

Cross-cutting tasks: T-ENV (Python, venv, pin), T-DATA (generator + hash), T-LIB (shared helper: loading, metrics, split/standardise, logistic, KNN, formatter for `key = value` and CSV writers), T-README, T-SUBMIT.

### M0 Plan (this document)
- Deliver: `docs/PLAN.md` committed.
- Exit: every rubric item mapped; open questions listed; user has read and accepted the plan.

### M1 Design
Tasks (documents/notes only, no code): choose file/module layout; define helper API signatures (data loader, split, standardise, sigmoid/loss/gradient, metrics from cell counts, KNN predict, KMeans, linkage, PCA, results writer); define output key order per Part; decide what each Part imports from Part 2 helpers; write a check list of checkpoints and pinned conventions per task; define the tolerance self-check approach (compare against brief-printed values only visually).
- Exit: written design covers every key and CSV with the formula and convention source; no unresolved convention; Part 4 dependency on Part 2 objects specified (model weights, K, folds, split).

### M2 Implement (staged, committing per sub-milestone)
- M2.0 Environment and data: Python 3.10+, venv, `requirements.txt`, `generate_data.py` verbatim, `data/records.csv`. Exit: printed line is exactly `800 10 170 e85e7605b784f4142844f79ad2dd0b8e`; hash still matches after commit.
- M2.1 Helper module + Part 1. Exit: `part1_results.txt` has all keys; checkpoints n=800, mean=8.9382; `gd_slope_gap` < 0.001; `tss_equals_ess_plus_rss = yes`; README Part 1 section drafted.
- M2.2 Part 2. Exit: checkpoints 560/119/240/51 and loss_epoch_0=0.6931; 9-row and 11-row CSVs; rates consistent with cells; no test-row leakage; README Part 2 section drafted.
- M2.3 Part 3. Exit: checkpoints wcss_k2=2262.6753 and eigenvalue_sum=5.0000; K-means pass counts 21/15/27 for K=4/5/6; segment sizes sum to 800; 3-row and 15-row CSVs; README Part 3 section drafted.
- M2.4 Part 4. Exit: checkpoints root_entropy=0.7462 and round1_error=0.2500; CSVs with 4/5/16/7 rows; importing the helper re-runs nothing; README Part 4 section and memo drafted.

### M3 Test
- Static: grep forbidden routines; check every key in the brief exists in each results file (key-list checker against the brief's task text); header/row-count checks; float format 4 dp; booleans yes/no.
- Cross-checks (allowed, not graded): independent sanity checks, e.g. compare closed form vs GD, confusion-cell identities, eigenvalues sum to 5, recompute selected values by hand.
- Clean-clone run: fresh clone into a temp dir (outside the project folder if needed, with approval), follow README commands, diff regenerated `outputs/` against committed.
- Exit: zero diffs on clean-clone rerun; all checkpoints pass; all pass-band-sensitive items reviewed (stable sorts, tie-breaks).

### M4 Deploy
- Create public GitHub repo (user action or on user's instruction), ensure default branch `main`, push `main`, open the repo URL logged out, confirm all files are visible and no images/links.
- Final checklist: README line 1; section order; `## How to run` commands match reality; memo <= 300 words; no `$` for money; no URLs inside repo (including `docs/`); no binary/image files; `requirements.txt` pin; MD5 verified.
- Submit link on course platform before closing time.
- Exit: submission confirmation received; link resolves to `main`.

### Timeline (compressed; today = 14 Nov, closes 16 Nov)
| When | Milestones |
|---|---|
| 14 Nov (today) | M0, M1, M2.0, M2.1, start M2.2 |
| 15 Nov | M2.2, M2.3, M2.4 (README sections written as each finishes) |
| 16 Nov (morning) | M3, M4; submit well before the posted closing time |

## 7. Proposed tech stack

- Python 3.10+ (3.11 or 3.12 acceptable) in a venv; NumPy pinned (1.26.4 or 2.2.6, whichever installs); standard library for csv, math, hashlib, os.
- Git for version control; GitHub for hosting (free account).
- No pandas/scikit-learn/scipy/matplotlib; no notebooks; no plotting (images not accepted).
- Project layout: exactly the brief's layout plus a shared helper (for example `common.py`) and `docs/`.
- Dev aids (optional, local only): a throwaway checker script for keys/headers/row counts kept outside the graded files, or clearly marked as a helper.

## 8. Dependencies and data needed

- Data: only `data/records.csv`, produced by `generate_data.py` (offline, deterministic, seed 2508). Hash gate before any analysis.
- Supplied constants inside the brief: 6x6 dissimilarity table (Task 3.4), AdaBoost table (Task 4.3), costs INR 50/400, thresholds, split rules, hyperparameters (eta, epochs, folds), architecture specs (Task 4.5).
- Inter-Part dependencies:
  - Part 2 -> Part 4: 560/240 split, standardisation statistics, final logistic weights/bias, KNN best K and reference set, cost rule.
  - Part 3.3 -> Part 3.5: segment membership for `mean_pc1_segment_*`; Part 3.1 standardised matrix reused in PCA.
  - Part 1 is independent of later Parts (uses all 800 rows); Part 1 6/7 share loaders.
- Tooling needed: a working Python install, git, a GitHub account, network access only for the final push (no network needed for running the code).
