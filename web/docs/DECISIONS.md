# DECISIONS

1. Work on local branch `dev`; `main` untouched; nothing pushed.
2. Root `.gitignore` ignores `tests/` at any depth, so `web/.gitignore` re-includes `web/tests/`.
3. The web layer imports `common.prepare_split()` and `part3.kmeans` (both import-safe) rather than
   re-implementing models, so probabilities are identical to the graded Part 2/4 numbers.
4. Charts are hand-written inline SVG (no vendored library): fewer files, no licensing, no build step.
5. Findings/memo/limitations text is parsed from the root README at request time (read only) so the site
   cannot drift from the graded write-up.
6. Cost rule from the brief: INR 50 per false alarm, INR 400 per missed default. Cost is reported on test rows;
   the cost-optimal tau shown comes from `outputs/part2_results.txt` (chosen on training rows).
7. Ensemble: hard vote = strict majority of selected models; KNN hard vote uses its majority rule (as in Part 4),
   logistic and stump use tau; soft vote averages probabilities and compares with tau. With all three at
   tau 0.3 this reproduces `part4_voting.csv`.
8. Score form validates ranges (account age 0-600, failed txns 0-100 integer, utilisation 0-1, inflow 0-1000,
   score 0-100, flags 0/1). Segment assignment uses nearest K=3 centroid in standardised X1..X5.
9. Dark/light via `prefers-color-scheme`; a single teal-blue accent shared with sibling portfolio pages.
10. No external fonts or CDNs: system font stack; works offline.
11. Unpinned transitive deps are not listed; `web/requirements.txt` pins the direct ones plus numpy (matching root).
