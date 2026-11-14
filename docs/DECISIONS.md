# DECISIONS

1. Line endings: `generate_data.py` and `data/records.csv` are marked `-text` in `.gitattributes`. The generator writes CRLF row terminators (csv default), and the brief's MD5 is of those bytes; the committed blob hashes to e85e7605b784f4142844f79ad2dd0b8e (checked with git cat-file). All other text files are LF.
2. NumPy pinned to 1.26.4 (the version installed in the project venv; brief says identical results on 1.26.4 and 2.2.6).
3. Extra files: `common.py` (shared helper, allowed by the brief), `docs/` and `tests/`, `.gitattributes`, `.gitignore`. The brief does not forbid extras, so they stay tracked. docs/REQUIREMENTS.md and docs/PLAN.md contain the brief's placeholder repository address pattern; they are documentation, not links for a grader, and the hygiene test exempts only those two files.
4. `elbow_k`: if no K in {3,4,5} satisfied the rule the code would print 5; on this data it prints 3.
5. Tau values in `part2_thresholds.csv` are printed as 4-dp floats (0.1000 ...) per "floats printed to 4 decimal places". `cost_tau` prints as 0.1500.
6. Task 4.2 KNN uses all 560 standardised training rows as reference and the Part 2 best K. Part 4 recomputes the Part 2 split, model and CV through `common.py` rather than importing `part2.py`, so no Part runs on import; results are identical because the code is shared.
7. Memo recommends the logistic model, so the Task 2.8 cost of other models is deliberately not computed; the memo says so.
8. AdaBoost ensemble: a zero score would count as wrong (does not occur).
9. Checkpoints all matched (n, mean 8.9382, 560/119/240/51, loss 0.6931, wcss_k2 2262.6753, K-means passes 21/15/27, eigenvalue_sum 5.0000, root_entropy 0.7462, round1_error 0.2500); no discrepancies to resolve.
10. Deploy check (local only): clean clone with a fresh venv reproduced data/ and outputs/ byte for byte and passed all tests; the clone was deleted. docs/ and tests/ were kept tracked (the brief does not forbid extra files). No remote was added and nothing was pushed.
