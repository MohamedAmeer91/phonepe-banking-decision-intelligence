# DESIGN

## Module layout
```
web/
  __init__.py
  app.py            FastAPI app, routes, error handlers, static mount
  state.py          cached model state (split, probabilities, KNN, stump, kmeans)
  outputs_io.py     read-only parsers for outputs/*.txt, *.csv and README sections
  schemas.py        pydantic request models
  static/index.html, css/style.css, js/{main,api,charts,md,pages}.js
  tests/            unittest + TestClient
```

## API schema (JSON)
- Confusion block `{tp,fp,tn,fn}`; metrics `{accuracy,precision,recall,f1,specificity,fpr}`.
- `GET /api/threshold?tau=0.3` -> `{tau, confusion, metrics, flagged, cost_inr, cost_formula,
  baselines:{flag_nobody, at_0_5, at_cost_tau, cost_tau}}`. tau in [0,1].
- `GET /api/curve` -> `{roc:[{tau,tpr,fpr}], cost:[{tau,test,train}], auc}` (101 points).
- `GET /api/knn?k=5` -> `{k, confusion, metrics, cv:[{k,f1}], best_k}`; k odd in 1..21.
- `POST /api/score` body `{X1..X7}` -> `{probability, tau, flagged, logit, bias, contributions:[{feature,label,z,weight,contribution}]}`.
- `POST /api/segment/assign` body `{X1..X5}` -> `{segment, distances:{A,B,C}}`.
- `GET /api/segments` -> `{segments:[...csv rows], book_mean:{X1..X5, outcome_rate, mean_value}}`.
- `POST /api/ensemble` body `{models:[logistic|knn|stump], mode:hard|soft, tau}` -> `{confusion, metrics, members}`.
- `GET /api/part/{n}` -> `{part, title, results:[{key,value}], tables:{name:{header,rows}}, series:{name:[[x,y]]}, findings:str}`.
- Errors: `{"detail": ...}` with 404/422/500; unknown part -> 404.

## Frontend structure
Hash router in `main.js` loads a render function per page from `pages.js`; `api.js` wraps fetch with error
handling; `charts.js` builds SVG line/bar/ROC charts; `md.js` renders the small markdown subset of the README
(bullets, bold, code, tables). Each interactive control is a native input with a label, debounced requests,
`aria-live` result regions.

## Visual design
Neutral page background, white/near-black cards with 1px borders and 12px radius, one accent (teal-blue
`#0f6e8c` light / `#5cc0de` dark). System sans-serif stack, 16px base, 1.55 line height, tabular numerals for
numbers. CSS custom properties on `:root`, redefined in `@media (prefers-color-scheme: dark)`. Responsive grid
(cards wrap, nav scrolls horizontally on phones), focus rings, text contrast >= 4.5:1.
