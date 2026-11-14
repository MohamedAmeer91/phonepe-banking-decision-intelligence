Domain: Banking

# PhonePe Banking Decision Intelligence (capstone)

All data are synthetic, produced by `generate_data.py`. PhonePe's real records are never used and nothing here describes its actual data, products, policies or performance. Nothing here is financial or credit advice. Money is in Indian Rupees (INR); the value column is in thousands of INR. Only the Python standard library and NumPy are used; every number in this file is copied from the committed files in `outputs/`.

## How to run

From a clean clone, at the repository root:

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Linux, macOS)
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
python -m unittest discover tests      (optional self-checks)
```

`generate_data.py` must print `800 10 170 e85e7605b784f4142844f79ad2dd0b8e`. Each part script rewrites its files in `outputs/` byte for byte. `common.py` holds the shared helpers (importing it runs nothing).

## Part 1 findings

- R1.1 H0 (mean monthly spend is 8.0, that is INR 8,000) is rejected: `z_stat` = 2.8930, beyond the critical value 1.96 (`reject_h0` = yes). A Type I error here would be concluding that borrowers do not spend INR 8,000 a month on average when in truth they do, and so re-cutting the budget for nothing. The gap `mean_value` - 8.0 = 8.9382 - 8.0 = 0.9382 column units (about INR 938 per borrower per month, roughly 12 percent above the assumption). It is statistically clear; whether it matters in practice depends on the size of the budget, but a 12 percent underestimate across a whole book is not trivial. The 95% interval is 8.3026 to 9.5738.
- R1.2 `n_required` = 1293 for a margin of INR 500 (0.5 units), against 800 rows available. The current 800 rows give a margin of about 0.64 units (INR 636), so the team would need about 493 more borrowers to reach INR 500; with the present data the average is known only to about plus or minus INR 636.
- R1.3 `gd_slope_gap` = 0.0000. The cost J is a smooth bowl with a single minimum, and with standardised X4 and a step of 0.1 the updates shrink the gradient to zero, so gradient descent converges to the same line the closed form solves for directly (J falls from `gd_J0` = 163.9178 to `gd_J500` = 40.3698).
- R1.4 `resid_sd_ratio_raw` = 3.7525 and `resid_sd_ratio_log` = 1.0396. The raw model shows a strong funnel: residual spread in the upper half of fitted values is almost four times that of the lower half. The log-log model removes it (ratio close to 1).
- R1.5 Predictors with an absolute coefficient above 0.05 (constant excluded): ln(X4) (`mr_beta_1` = 1.1576), X2 failed transactions (`mr_beta_3` = 0.0940), X3 credit utilisation (`mr_beta_4` = 0.1067) and X6 irregular salary (`mr_beta_6` = 0.1518). This is a mechanical rule, not a ranking of importance, because the units differ. `mr_adj_r2` = 0.7403 against `loglog_adj_r2` = 0.6561, so the extra predictors do earn their place.

## Part 2 findings

- R2.1 Final training loss after 200 updates, lowest to highest: eta 1.0 (0.3703), eta 0.1 (0.3776), eta 0.01 (0.5091). A learning rate that is too small has barely started to learn after 200 updates, while the largest rate here still converged smoothly; it was not so large that it overshot.
- R2.2 The largest absolute weights are X3 credit utilisation (0.9013), X1 account age (-0.5554) and X2 failed transactions (0.5505). The largest odds ratio is `odds_ratio_X3` = 2.4629: each extra training SD of credit utilisation multiplies the odds of default or 90 days past due by about 2.5, other things fixed.
- R2.3 `blind_accuracy` = 0.7875 (predict 0 for everyone) against 0.8042 accuracy at tau = 0.5, so accuracy barely beats a model that finds no defaulter at all; the metric hides the imbalance (51 positives in 240). Lowering tau from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 and lowered precision from 0.5455 to 0.4634.
- R2.4 At tau = 0.3 the logistic model has `f1` = 0.5714 and recall 0.7451; KNN (K = 5) has `knn_f1` = 0.3696 and recall 0.3333. The logistic model is better on both.
- R2.5 I would deploy the logistic model at the cost-chosen threshold `cost_tau` = 0.1500. On the test rows it costs INR 6050 (`cost_test_at_tau`), against INR 11800 at tau = 0.5 and INR 20400 if nobody is flagged. The threshold was chosen on training rows so that the test rows stay an untouched, honest check of the choice.

## Part 3 findings

- R3.1 WCSS drops: WCSS(2) - WCSS(3) = 659.8282, WCSS(3) - WCSS(4) = 108.0739, WCSS(4) - WCSS(5) = 171.0575, WCSS(5) - WCSS(6) = 45.5466 (from `wcss_k2` = 2262.6753, `wcss_k3` = 1602.8471, `wcss_k4` = 1494.7732, `wcss_k5` = 1323.7157, `wcss_k6` = 1278.1691). `elbow_k` = 3: the drop after K = 3 is far below half of the drop before it. The drops do not shrink steadily (the K = 4 to 5 drop is larger than the 3 to 4 drop), which shows that the pinned starting rows give one reproducible solution, not necessarily the global optimum.
- R3.2 Segments from `part3_segments.csv`:
  - A, "Steady veterans" (322 records): long tenure (mean 47.5 months), low utilisation (0.2250), outcome rate 0.0186. Action: keep light-touch, offer limit or product upgrades.
  - B, "Busy high-inflow accounts" (255 records): highest inflow (28.7 thousand) and most failed transactions (6.07), outcome rate 0.2353, highest spend (17.1 thousand). Action: fix failed-payment friction and send payment-success tips and early reminders.
  - C, "New heavy borrowers" (223 records): short tenure (14.4 months), utilisation 0.7118, lowest repayment score (38.5). Highest `outcome_rate` is segment C at 0.4664. Action: start collections nudges before the due date and cap new credit lines.
- R3.3 Single linkage at the cut (`single_clusters_at_cut` = 4): {E2, E3, E6}, {E1}, {E4}, {E5}. Complete linkage (`complete_clusters_at_cut` = 2): {E1, E4, E5} and {E2, E3, E6}. Single linkage joins clusters on their closest pair, so a chain of small steps (E3-E6, then E2, then E5-E4, then E1) keeps merging at low heights and the largest gap is found early. Complete linkage uses the farthest pair, so the very large E1-E2 distance (7.9) forces the final merge to be much higher and the cut falls there.
- R3.4 PC1 is driven, with positive sign, by X1 account age (0.5317) and X5 repayment score (0.4418), and with negative sign by X3 utilisation (-0.5187), X2 failed transactions (-0.4100) and X4 inflow (-0.2916): a tenure-and-repayment axis against a stress axis. The mean vector of the standardised data (step 1) is all zeros. Components needed for 80 percent of the variance: `components_for_80pct` = 3 (cumulative 0.7593 after two, 0.8616 after three). `mean_pc1_segment_A` = 1.7773, `mean_pc1_segment_B` = -0.7831 and `mean_pc1_segment_C` = -1.6709, so PC1 orders the segments from safest (A) to riskiest (C).

## Part 4 findings

- R4.1 S4 (account age at most 24 months) is the best split by both measures (`info_gain` 0.1252, `gini_gain` 0.0606) because it separates the training rows into one side with 86 of 188 positives (0.4574) and one with 33 of 372 (0.0887), the largest drop in impurity. A tree grown until every leaf is pure memorises noise and individual records, so it fits the training rows perfectly and generalises badly. A maximum depth, or a minimum number of records per leaf, stops it.
- R4.2 I would ship the logistic model (at the cost-chosen threshold 0.15; `part4_voting.csv` compares all five at 0.3): f1 0.5714 and recall 0.7451 for logistic. Stump: f1 0.5532, recall 0.7647; hard vote: f1 0.5344, recall 0.6863; soft vote: f1 0.5507, recall 0.7451; KNN: f1 0.3696, recall 0.3333. Combining models did not help; the best single model beat both votes on f1.
- R4.3 After round 1 normalisation the two misclassified points (3 and 7) carry a total weight of 0.5000 (0.2500 each). Round 2 trusts h2 more because its weighted error under the new weights is 0.1667 against 0.2500 for h1 in round 1, which gives it `round2_alpha` = 0.8047 against `round1_alpha` = 0.5493. `ensemble_accuracy` = 0.7500; `oob_fraction` = 0.3676.
- R4.4 Model A option 2 (28 x 28 x 1 input):

  | Layer | Output shape (h x w x d) | Parameters |
  |---|---|---|
  | conv 5x5, 6 filters, pad 2 | 28 x 28 x 6 | 156 |
  | ReLU, max-pool 2x2 | 14 x 14 x 6 | 0 |
  | conv 5x5, 16 filters | 10 x 10 x 16 | 2416 |
  | ReLU, max-pool 2x2 | 5 x 5 x 16 | 0 |
  | flatten | 1 x 1 x 400 | 0 |
  | dense 32, ReLU | 1 x 1 x 32 | 12832 |
  | dense 10 | 1 x 1 x 10 | 330 |

  `cnn_total` = 15734 against `mlp_total` = 13002 for the fully connected option. A 128 x 128 x 3 input into 1024 neurons needs `fc_layer_params` = 50332672 parameters, while 64 convolution filters of 3 x 3 need `conv_layer_params` = 1792, because filters share weights across positions.

  Model B: W_xh is 128 x 64, W_hh is 128 x 128 and W_hy is 16 x 128 (`rnn_total` = 26768). The total does not grow with longer sequences because the same weights are reused at every time step. For 10 classes I would use ReLU in the hidden layers and a softmax output trained with categorical cross-entropy loss. In a very deep plain network the gradient is a product of many small factors on its way back, so the early layers receive almost no signal and stop learning; a skip connection adds the input to the output of a block, giving the gradient a direct path back and keeping it from vanishing.

## Model-choice memo

To: Head of the lending-risk team. Recommendation: put the logistic-regression model into production, with a flag threshold of 0.15.

On the 240 held-out rows it is the strongest structured model. At 0.3 its f1 is 0.5714 and recall 0.7451, against f1 0.3696 for KNN (K = 5) and 0.5532 for the one-question stump; neither hard vote (0.5344) nor soft vote (0.5507) beat it. The threshold of 0.15 was chosen on training rows only, by the rule of INR 50 per false alarm and INR 400 per missed default. Its test cost is INR 6050, against INR 11800 at the default 0.5 and INR 20400 if nobody is flagged. These are the logistic model's costs; I did not compute costs for the other models.

Limitation: the evidence is one split with only 51 positive test records, on synthetic data, so a swing of a few records moves the costs by hundreds of Rupees and the threshold may not carry over to the real book.

Neither neural model should come first: build Model A (the cheque-digit reader, 15734 parameters in its convolutional form) only after the structured model is in production, and Model B later still.

## Limitations

- The data are synthetic, generated from a known formula; real PhonePe lending data would be noisier, drift over time and carry reporting biases.
- One fixed 560/240 split and one set of pinned folds: all test results, the best K = 5 and the threshold 0.1500 are valid for this split only and have no confidence intervals.
- Only 51 positive test records: a single record changes recall by about 2 percentage points and the cost by INR 400.
- The cost rule (INR 50 per false alarm, INR 400 per miss) is a rough assumption, and cost was computed only for the logistic model.
- K-means uses pinned starting rows, so its segments are reproducible but not necessarily the global optimum; the six-cohort dissimilarity table is supplied, not derived from the data.
- Models A and B are sized by counting parameters only; nothing was trained, so their accuracy is unknown.
