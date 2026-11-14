# DESIGN - PhonePe Banking Decision Intelligence

Source of truth: `docs/REQUIREMENTS.md`. This file fixes layout, signatures, algorithms, output keys and checks. It contains no links.

## 1. Repository layout

```
README.md  requirements.txt  generate_data.py  common.py
data/records.csv
part1.py part2.py part3.py part4.py
outputs/ part1_results.txt part2_results.txt part3_results.txt part4_results.txt
         part2_thresholds.csv part2_knn_cv.csv part3_segments.csv part3_hierarchical.csv
         part4_splits.csv part4_voting.csv part4_adaboost.csv part4_architecture.csv
.gitattributes  .gitignore
docs/ (REQUIREMENTS, PLAN, DESIGN, DECISIONS)   tests/ (stdlib unittest)
```
`common.py` is the shared helper (importing it runs nothing). `docs/` and `tests/` are extras; see DECISIONS.md.

## 2. Data and generator
`generate_data.py` is committed verbatim from the brief. Run once from the repo root; it must print `800 10 170 e85e7605b784f4142844f79ad2dd0b8e`. `.gitattributes` marks `generate_data.py` and `data/records.csv` as `-text` so no line-ending conversion touches them (the CSV is written with `csv.writer` default CRLF terminators; the hash is of those bytes). Verified with `git cat-file blob HEAD:data/records.csv | md5sum`.

## 3. common.py API
- `load_data()` -> dict of float arrays: `X` (800x7, X1..X7), `y` (outcome int array), `value`; reads `data/records.csv` with `csv`.
- `fmt(v)`: int -> `str`, bool -> `yes`/`no`, float -> `f"{v:.4f}"`, str as is. Negative zero printed as `0.0000`.
- `write_results(path, pairs)` writes `key = value` lines, LF endings, `newline=""`. `write_csv(path, header, rows)` writes header + rows with `fmt`, LF endings.
- `stratified_split(y)` -> `(train_idx, test_idx)` (sorted, file order) using floor(7*count/10) per class.
- `standardise(Xtrain, Xother)` -> standardised arrays with training mean and ddof=1 SD (implemented by hand).
- `sigmoid`, `bce(y, p)`, `train_logistic(X, y, eta, updates, record_at=())` -> `(w, b, losses dict)`.
- `confusion(y, pred)` -> `(tp, fp, tn, fn)`; `metrics(tp,fp,tn,fn)` -> dict accuracy, precision, recall, f1, specificity, fpr (0 when denominator 0).
- `knn_prob(Xref, yref, Xq, K)` -> share of positives among K nearest (stable argsort on squared Euclidean distance, ties by reference position).
- `mean(a)`, `sd(a)` (ddof=1) hand-written reductions on top of `numpy.sum`.
- `prepare_split()` -> everything Part 2 and Part 4 share: indices, standardised train/test matrices, labels, trained logistic model (1000 updates, eta 0.1), probabilities. Deterministic; Part 4 recomputes it by calling this function (same code, same numbers). Part 4 additionally calls `choose_knn_k()` (shared, in common.py) to get the best K and the KNN CV table.

## 4. Part algorithms and exact outputs

### Part 1 (`part1.py`, all 800 rows) -> `outputs/part1_results.txt`
Key order: `n, mean_value, sd_value, se_value, ci95_low, ci95_high, z_stat, reject_h0, n_required, reg_slope, reg_intercept, tss, ess, rss, tss_equals_ess_plus_rss, r2, rmse, mae, gd_J0, gd_J1, gd_J10, gd_J100, gd_J500, gd_slope, gd_intercept, gd_slope_gap, resid_sd_lower_raw, resid_sd_upper_raw, resid_sd_ratio_raw, loglog_intercept, loglog_slope, loglog_r2, loglog_rmse, resid_sd_lower_log, resid_sd_upper_log, resid_sd_ratio_log, mr_beta_0..mr_beta_7, mr_r2, mr_adj_r2, loglog_adj_r2`.
- 1.3: `n_required = ceil((1.96*sd/0.5)^2)` computed with `math.ceil`.
- 1.4 closed form with divisor n-1 covariance; 1.5 GD on standardised X4, record J at 0,1,10,100,500 updates; convert back.
- 1.6: stable `argsort` of fitted values; halves of 400; sample SD of residuals; log-log by the same closed form helper `simple_ols(x, y)` returning slope, intercept, fitted.
- 1.7: design matrix [1, ln X4, X1, X2, X3, X5, X6, X7], `numpy.linalg.solve(XtX, XtY)`, R2 and adjusted R2 on ln(value).

### Part 2 (`part2.py`) -> `part2_results.txt`, `part2_thresholds.csv`, `part2_knn_cv.csv`
Result key order: `train_rows, train_positives, test_rows, test_positives, loss_epoch_0, loss_epoch_1, loss_epoch_10, loss_epoch_100, loss_epoch_1000, weight_X1..weight_X7, bias, test_bce, loss_epoch_200_eta_0.01, loss_epoch_200_eta_0.1, loss_epoch_200_eta_1.0, odds_ratio_X1..X7, tau_0.5_{tp,fp,tn,fn,accuracy,precision,recall,f1,specificity}, tau_0.3_{...}, blind_accuracy, knn_best_k, knn_tp, knn_fp, knn_tn, knn_fn, knn_accuracy, knn_precision, knn_recall, knn_f1, cost_tau, cost_train_at_tau, cost_test_at_tau, cost_test_at_0.5, cost_test_flag_nobody`.
- CSV `part2_thresholds.csv` header `tau,tp,fp,tn,fn,tpr,fpr`, tau 0.1..0.9 (computed as k/10 for k=1..9).
- CSV `part2_knn_cv.csv` header `K,mean_cv_f1`, K = 1,3,...,21; fold = train position % 5; reference = other folds in file order; predicted positive when positives*2 >= K+1; best K highest mean, ties smaller K (strict `>` while scanning ascending).
- 2.8: tau = k/20, k=1..19, training cost 50*fp+400*fn, strict `<` scan ascending; `cost_tau` printed with 4 dp (e.g. `0.2500`); costs are integers (INR).
- Positive when p >= tau; thresholds k/20 computed as `k/20` floats.

### Part 3 (`part3.py`) -> `part3_results.txt`, `part3_segments.csv`, `part3_hierarchical.csv`
Key order: `wcss_k2..wcss_k6, elbow_k, size_segment_A, size_segment_B, size_segment_C, single_clusters_at_cut, complete_clusters_at_cut, average_clusters_at_cut, eigenvalue_sum, eigenvalue_1..5, explained_ratio_1..5, cumulative_ratio_1..5, components_for_80pct, pc1_loading_X1..X5, pc2_loading_X1..X5, row1_pc1, row1_pc2, row2_pc1, row2_pc2, row3_pc1, row3_pc2, mean_pc1_segment_A/B/C`.
- `kmeans(Z, K)` returns labels, centroids, passes; init rows `i*(800//K)`; assignment via squared distance with `argmin` (first minimum = lowest index); empty cluster keeps centroid; stop on the pass with no changes; passes counted including the final one (sanity check against 21/15/27 printed in the brief, as a test, not typed into outputs).
- `elbow_k`: first K in 3,4,5 with `W(K)-W(K+1) < 0.5*(W(K-1)-W(K))`; if none, print the last K tested (5) and note in DECISIONS (value only used for the README).
- Segments: K=3 clusters sorted by ascending original-units mean of X3 -> A,B,C. CSV header `segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value`.
- Hierarchical: `agglomerate(D, names, linkage)`; clusters as tuples of indices; pick smallest linkage (strict `<`, scan in cluster-list order); `cluster_a` is the cluster with smaller minimum member; labels `E1+E3`. CSV header `linkage,merge_no,cluster_a,cluster_b,height`; rows single, complete, average. Cut: gaps between consecutive heights; largest gap index g (1-based, between merge g and g+1) means g merges below the cut; clusters = 6 - g.
- PCA seven steps: mean vector, centre, S = XtX/(n-1), `numpy.linalg.eigh`, sort descending, sign flip so largest-|entry| positive (first max on ties), project top 2.

### Part 4 (`part4.py`) -> `part4_results.txt` + 4 CSVs
Key order: `root_entropy, root_gini, best_split_by_info_gain, best_split_by_gini_gain, stump_rate_yes, stump_rate_no, round1_error, round1_alpha, round2_error, round2_alpha, ensemble_accuracy, oob_fraction, mlp_total, cnn_total, fc_layer_params, conv_layer_params, rnn_total`.
- Splits on original-unit training rows: S1 X6==1, S2 X7==1, S3 X3>0.5, S4 X1<=24. CSV header `split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain`. Best = highest gain, ties to earlier split.
- Stump: rates from training shares on each side; applied to test rows by the best-by-info-gain split condition. Voting CSV header `model,tp,fp,tn,fn,precision,recall,f1`, rows logistic, knn, stump, hard_vote, soft_vote; knn probability is the neighbour share with K = best K and reference = all 560 standardised train rows; knn positive when share*2 >= ... equivalently count >= (K+1)/2.
- AdaBoost: arrays y,h1,h2, weights; CSV header `round,point,weight_before,weight_unnormalised,weight_after`, 16 rows. `ensemble_accuracy` from `sign(a1*h1+a2*h2)` (zero sum counted wrong; checked not to occur).
- `oob_fraction = (1-1/560)**560`.
- Architecture CSV header `layer,parameters`: mlp_hidden1 784*16+16, mlp_hidden2 16*16+16, mlp_output 16*10+10; cnn_conv1 (5*5*1+1)*6; spatial 28->28->14; conv2 (5*5*6+1)*16, 14->10->5; flatten 5*5*16=400; dense1 400*32+32; dense2 32*10+10. Computed from shape formulas in code (`conv_out`), not typed. `fc_layer_params = (128*128*3)*1024+1024`; `conv_layer_params = (3*3*3+1)*64`; `rnn_total = 128*64 + 128*128 + 16*128 + 128 + 16`.
- Extra (README/memo support): cost at the recommended threshold for non-logistic models is NOT computed (memo says so) unless stated; the memo recommends logistic regression, whose cost is in part2 outputs.
- Part 4 imports `common` only (never `part2`) so no Part re-runs on import.

## 5. README outline (order fixed by the brief)
1. `Domain: Banking` (line 1 exactly), a one-paragraph synthetic-data disclaimer.
2. `## How to run`: `python -m venv .venv`, activate, `pip install -r requirements.txt`, `python generate_data.py`, `python part1.py` ... `python part4.py`, optional `python -m unittest discover tests`.
3. `## Part 1 findings` R1.1-R1.5; `## Part 2 findings` R2.1-R2.5; `## Part 3 findings` R3.1-R3.4; `## Part 4 findings` R4.1-R4.4 (with CNN shape table, RNN shapes, activations); `## Model-choice memo` (<= 300 words, addressed to the head of the lending-risk team); `## Limitations` (>= 4 bullets).
All numbers are copied by a one-off script that reads `outputs/` (not typed from the brief). Money written as `INR n`; no dollar sign; no URLs.

## 6. Test strategy (`tests/`, stdlib `unittest`)
- `test_outputs.py`: each results file has exactly the expected key list in order; CSV headers and row counts; formats (4 dp floats, `yes`/`no`); confusion-cell identities and rate consistency; invariants (segment sizes sum 800, eigenvalue sum 5, explained ratios sum 1, tss = ess + rss, gd gap < 0.001, adaboost weights sum to 1, test counts 240/51); checkpoints from the brief (n, mean 8.9382, 560/119/240/51, loss 0.6931, wcss_k2 2262.6753, eigenvalue_sum 5.0000, root_entropy 0.7462, round1_error 0.2500); K-means pass counts 21/15/27 via importable function.
- `test_hygiene.py`: forbidden-content grep over `*.py` (no sklearn, scipy, pandas, polyfit, `np.cov`, corrcoef, lstsq; `np.linalg.eigh` only in part3.py and `solve` only in part1.py), no `$` or URLs in README/outputs/py, no image/pdf files tracked, data MD5, generator verbatim hash line.
- `test_readme.py`: first line, section order, memo word count <= 300, each quoted key number appears in README as printed in outputs, at least 3 limitations.
- `test_determinism.py`: run `part1..part4` twice into the repo (outputs compared byte-wise before/after, also via a second run).

## 7. Deploy checklist (local only)
Clean clone inside the project folder (`_cleanclone_tmp`), fresh venv, `pip install -r requirements.txt`, run README commands, diff `outputs/` and `data/` vs committed, delete clone. Working tree clean, only branch `main`, no remotes, `git ls-files` matches layout plus documented extras, md5 of committed CSV via `git cat-file`. No push.
