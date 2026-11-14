# Web portfolio layer

An interactive site for the PhonePe banking capstone: FastAPI backend plus plain HTML, CSS and ES modules
(no build step). It only reads `outputs/` and imports `common.py` and `part3.kmeans` to recompute model
probabilities on the same 560/240 split. It never modifies the graded code or data.

## Run locally (from the repository root)

Windows (PowerShell):

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r web/requirements.txt
    .venv\Scripts\python.exe -m uvicorn web.app:app --port 8000

macOS or Linux:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r web/requirements.txt
    .venv/bin/python -m uvicorn web.app:app --port 8000

Then open 127.0.0.1:8000 in a browser. Interactive API docs are at `/docs`.

## Run the tests

    .venv\Scripts\python.exe -m unittest discover -s web/tests -t .

## Docker

Build from the repository root so the graded files are in the context:

    docker build -f web/Dockerfile -t phonepe-web .
    docker run --rm -p 8000:8000 phonepe-web

## Pages

Overview, Part 1 to Part 4, Memo, Limitations. Interactive pieces: threshold lab (confusion matrix, metrics,
ROC point, INR cost), KNN K selector, score-a-borrower form, segment explorer and assignment form,
split comparison, ensemble voting lab, AdaBoost round viewer.

## Layout

    web/app.py        routes and error handling
    web/state.py      cached split, probabilities, KNN, stump, K-means
    web/outputs_io.py read-only parsers for outputs/ and README sections
    web/schemas.py    pydantic request models
    web/static/       index.html, css/, js/
    web/tests/        unittest plus FastAPI TestClient
    web/docs/         PLAN, DESIGN, DECISIONS
