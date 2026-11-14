"""FastAPI portfolio layer for the PhonePe banking capstone.

Reads outputs/ and imports common.py only; never writes to the project.
Run from the repository root:  uvicorn web.app:app
"""
import os
import re
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from web import outputs_io as oio
from web.schemas import EnsembleRequest, ScoreRequest, SegmentRequest
from web.state import ALL_KS, COST_FALSE_ALARM, COST_MISS, FEATURES, cost_of, get_state

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@asynccontextmanager
async def lifespan(app):
    get_state()  # build and cache the split, probabilities, KNN and K-means once
    yield


app = FastAPI(title="PhonePe Banking Decision Intelligence", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal error: " + type(exc).__name__})


def _round(d, nd=6):
    return {k: round(float(v), nd) for k, v in d.items()}


def _rd(name):
    try:
        return oio.results_dict(name)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Missing output file: " + name)


# ------------------------------------------------------------------ meta

@app.get("/api/health")
def health():
    st = get_state()
    return {"status": "ok", "train_rows": len(st.ytr), "test_rows": len(st.yte), "best_k": st.best_k}


@app.get("/api/overview")
def overview():
    r1, r2, r3, r4 = (_rd("part%d_results.txt" % i) for i in (1, 2, 3, 4))
    X, y, value = get_state().s["X"], get_state().s["y"], get_state().s["value"]
    headline = [
        {"label": "Borrower records", "value": len(y), "note": "synthetic, 7 predictors"},
        {"label": "Default / 90 days past due rate", "value": round(float(np.mean(y)), 4),
         "note": "share of outcome = 1"},
        {"label": "Mean monthly spend", "value": r1["mean_value"], "note": "INR thousands; plan assumes 8.0"},
        {"label": "Sample needed for INR 500 margin", "value": r1["n_required"], "note": "records"},
        {"label": "Logistic F1 at tau 0.3", "value": r2["tau_0.3_f1"], "note": "held-out 240 rows"},
        {"label": "Test cost at tau %.2f (INR)" % r2["cost_tau"], "value": r2["cost_test_at_tau"],
         "note": "vs INR %d at tau 0.5" % r2["cost_test_at_0.5"]},
        {"label": "Segments found", "value": r3["elbow_k"], "note": "K-means elbow"},
        {"label": "Best split by gain", "value": r4["best_split_by_info_gain"], "note": "account age <= 24 months"},
    ]
    findings = [
        "Mean monthly spend is INR %.0f, above the INR 8,000 plan (z = %.2f, hypothesis rejected); "
        "%d records are needed for a INR 500 margin." % (r1["mean_value"] * 1000, r1["z_stat"], r1["n_required"]),
        "Credit utilisation is the strongest default driver (odds ratio %.2f per standard deviation)."
        % r2["odds_ratio_X3"],
        "Lowering the flag threshold from 0.5 to 0.3 lifts recall from %.2f to %.2f."
        % (r2["tau_0.5_recall"], r2["tau_0.3_recall"]),
        "Choosing the threshold by money (tau %.2f) cuts test cost from INR %d to INR %d."
        % (r2["cost_tau"], r2["cost_test_at_0.5"], r2["cost_test_at_tau"]),
        "Three borrower segments exist; the new heavy borrowers (C) default at %.0f%%."
        % (100 * oio.read_table("part3_segments.csv")["rows"][2][7]),
        "Combining models did not beat the single logistic model.",
    ]
    return {
        "title": "PhonePe Banking Decision Intelligence",
        "problem": [
            "Is the planning assumption about monthly spend still right, and how far can a simple model of spend be trusted?",
            "Which borrowers will default or fall 90 days behind, and where should the alert threshold sit?",
            "What kinds of borrowers does the book contain, and how large would later neural models be?",
        ],
        "headline": headline,
        "findings": findings,
        "cost_rule": {"false_alarm_inr": COST_FALSE_ALARM, "miss_inr": COST_MISS},
    }


PART_META = {
    1: ("Part 1 - Estimation and regression", [], [
        ("gd_loss", r"gd_J(\d+)$")]),
    2: ("Part 2 - Classification and evaluation",
        ["part2_thresholds.csv", "part2_knn_cv.csv"],
        [("train_loss", r"loss_epoch_(\d+)$"), ("lr_loss", r"loss_epoch_200_eta_([\d.]+)$")]),
    3: ("Part 3 - Segments and compression",
        ["part3_segments.csv", "part3_hierarchical.csv"],
        [("wcss", r"wcss_k(\d+)$"), ("explained", r"explained_ratio_(\d+)$"),
         ("cumulative", r"cumulative_ratio_(\d+)$")]),
    4: ("Part 4 - Model choice and architecture",
        ["part4_splits.csv", "part4_voting.csv", "part4_adaboost.csv", "part4_architecture.csv"],
        []),
}


@app.get("/api/part/{n}")
def part(n: int):
    if n not in PART_META:
        raise HTTPException(status_code=404, detail="Unknown part: %d" % n)
    title, tables, series_spec = PART_META[n]
    pairs = oio.read_results("part%d_results.txt" % n)
    series = {}
    for name, pattern in series_spec:
        if not pattern:
            continue
        pts = []
        for k, v in pairs:
            m = re.fullmatch(pattern, k)
            if m:
                pts.append([float(m.group(1)), v])
        series[name] = pts
    return {
        "part": n,
        "title": title,
        "results": [{"key": k, "value": v} for k, v in pairs],
        "tables": {t[:-4]: oio.read_table(t) for t in tables},
        "series": series,
        "findings": oio.readme_section("Part %d findings" % n),
    }


@app.get("/api/memo")
def memo():
    return {"title": "Model-choice memo", "text": oio.readme_section("Model-choice memo")}


@app.get("/api/limitations")
def limitations():
    return {"title": "Limitations", "text": oio.readme_section("Limitations")}


# ------------------------------------------------------------------ Part 2 interactive

@app.get("/api/model")
def model():
    st = get_state()
    X = st.s["X"]
    odds = np.exp(st.w)
    feats = []
    for j, (code, col, label) in enumerate(FEATURES):
        flag = code in ("X6", "X7")
        feats.append({
            "code": code, "column": col, "label": label,
            "kind": "flag" if flag else ("count" if code == "X2" else "number"),
            "min": float(X[:, j].min()), "max": float(X[:, j].max()),
            "mean": round(float(X[:, j].mean()), 4),
            "train_mean": round(float(st.mu[j]), 6), "train_sd": round(float(st.sd[j]), 6),
            "weight": round(float(st.w[j]), 6), "odds_ratio": round(float(odds[j]), 6),
        })
    return {"bias": round(float(st.b), 6), "features": feats,
            "train_rows": len(st.ytr), "test_rows": len(st.yte),
            "test_positives": int(st.yte.sum()), "default_tau": 0.15}


@app.get("/api/threshold")
def threshold(tau: float = Query(0.3, ge=0.0, le=1.0)):
    st = get_state()
    conf, m = st.block(st.yte, st.p_test >= tau)
    r2 = _rd("part2_results.txt")
    return {
        "tau": tau,
        "confusion": conf,
        "metrics": _round(m),
        "flagged": conf["tp"] + conf["fp"],
        "cost_inr": COST_FALSE_ALARM * conf["fp"] + COST_MISS * conf["fn"],
        "cost_formula": "%d x false alarms + %d x missed defaults" % (COST_FALSE_ALARM, COST_MISS),
        "baselines": {
            "flag_nobody": cost_of(st.yte, np.zeros(len(st.yte), dtype=bool)),
            "at_0_5": cost_of(st.yte, st.p_test >= 0.5),
            "cost_tau": r2["cost_tau"],
            "at_cost_tau": cost_of(st.yte, st.p_test >= r2["cost_tau"]),
        },
    }


@app.get("/api/curve")
def curve():
    return get_state().curve


@app.get("/api/knn")
def knn(k: int = Query(5, ge=1, le=21)):
    if k not in ALL_KS:
        raise HTTPException(status_code=422, detail="k must be odd between 1 and 21")
    st = get_state()
    conf, m = st.block(st.yte, st.knn_pred(k))
    return {"k": k, "confusion": conf, "metrics": _round(m),
            "cv": [{"k": kk, "f1": round(f, 4)} for kk, f in st.cv], "best_k": st.best_k}


@app.post("/api/score")
def score(req: ScoreRequest):
    st = get_state()
    raw = [getattr(req, c) for c, _, _ in FEATURES]
    p, logit, z, contrib = st.score(raw)
    return {
        "probability": round(p, 6), "tau": req.tau, "flagged": bool(p >= req.tau),
        "logit": round(logit, 6), "bias": round(float(st.b), 6),
        "contributions": [
            {"feature": c, "label": lab, "value": raw[j], "z": round(float(z[j]), 6),
             "weight": round(float(st.w[j]), 6), "contribution": round(float(contrib[j]), 6)}
            for j, (c, _, lab) in enumerate(FEATURES)
        ],
    }


# ------------------------------------------------------------------ Part 3 / 4 interactive

@app.get("/api/segments")
def segments():
    tbl = oio.read_table("part3_segments.csv")
    segs = [dict(zip(tbl["header"], r)) for r in tbl["rows"]]
    st = get_state()
    X, y, value = st.s["X"], st.s["y"], st.s["value"]
    book = {"X%d" % (j + 1): round(float(X[:, j].mean()), 4) for j in range(5)}
    book["outcome_rate"] = round(float(y.mean()), 4)
    book["mean_value"] = round(float(value.mean()), 4)
    book["size"] = len(y)
    return {"segments": segs, "book_mean": book,
            "features": [{"code": c, "label": lab} for c, _, lab in FEATURES[:5]]}


@app.post("/api/segment/assign")
def segment_assign(req: SegmentRequest):
    seg, dist = get_state().assign([req.X1, req.X2, req.X3, req.X4, req.X5])
    return {"segment": seg, "distances": _round(dist)}


@app.post("/api/ensemble")
def ensemble(req: EnsembleRequest):
    conf, m = get_state().ensemble(req.models, req.mode, req.tau)
    return {"models": req.models, "mode": req.mode, "tau": req.tau,
            "confusion": conf, "metrics": _round(m),
            "cost_inr": COST_FALSE_ALARM * conf["fp"] + COST_MISS * conf["fn"]}


# ------------------------------------------------------------------ static

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
