# PLAN - PhonePe Banking portfolio web layer

## Purpose
Turn the graded capstone (lending-risk analytics on 800 synthetic PhonePe borrower records) into an
interactive portfolio site. The site only READS `outputs/` and IMPORTS `common.py` / `part3.kmeans`
(import-safe, no side effects). It never changes the graded code, `data/`, `outputs/` or root files.

## Pages (single page app, hash routed, plain ES modules)
1. Overview - problem statement, three questions, headline numbers, key findings.
2. Part 1 - Estimation and regression: hypothesis test, sample size, regression baselines, gradient
   descent loss curve.
3. Part 2 - Classification: loss curve, weights and odds ratios, INTERACTIVE threshold lab (slider ->
   confusion matrix, precision/recall/F1, ROC curve, INR cost), KNN K selector with CV table, score-a-borrower form.
4. Part 3 - Segments: segment explorer (feature selector, comparison with book average), K-means elbow,
   PCA scree, hierarchical merge tables, assign-a-borrower-to-a-segment form.
5. Part 4 - Model comparison: split purity comparison, ensemble voting lab (pick models, hard/soft vote, tau),
   AdaBoost round-by-round weights, architecture parameter counts.
6. Memo - model-choice memo. 7. Limitations.

## API endpoints (all JSON, under /api)
| Endpoint | Purpose |
|---|---|
| GET /api/health | liveness, counts |
| GET /api/overview | headline numbers and key findings |
| GET /api/part/{1-4} | results key/values, tables, chart series, README findings |
| GET /api/memo, GET /api/limitations | README text sections |
| GET /api/model | weights, odds ratios, feature metadata for the form |
| GET /api/threshold?tau= | confusion matrix, metrics, INR cost at tau on test rows |
| GET /api/curve | ROC points and cost-versus-tau curve (test + train) |
| GET /api/knn?k= | KNN confusion/metrics for odd K 1..21, CV table |
| POST /api/score | features -> probability, flag, per-feature contributions |
| POST /api/segment/assign | X1..X5 -> nearest K=3 segment |
| GET /api/segments | segment table, centroids, book average |
| POST /api/ensemble | model subset + vote mode + tau -> confusion/metrics |

## Data flow
Startup: `common.prepare_split()` (same 560/240 split, same logistic training) -> cache test and train
probabilities, KNN probabilities, stump rates, K-means K=3 centroids. Requests only do cheap array
arithmetic over cached vectors. Static files (HTML/CSS/JS, inline SVG charts) are served by FastAPI.

## Scope
In: everything above, pydantic validation, JSON errors, tests, Dockerfile, README.
Out: retraining models on demand, auth, database, any change to graded files, network calls.
