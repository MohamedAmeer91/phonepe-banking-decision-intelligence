"""API tests: status, schema, cross-checks against outputs/ files, validation, static files."""
import os
import sys
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")
from fastapi.testclient import TestClient  # noqa: E402

from web import outputs_io as oio  # noqa: E402
from web.app import app  # noqa: E402

GOOD = {"X1": 24, "X2": 3, "X3": 0.5, "X4": 20, "X5": 60, "X6": 0, "X7": 1}


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = TestClient(app)
        cls.c = cls.ctx.__enter__()
        cls.r2 = oio.results_dict("part2_results.txt")

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def get(self, url, status=200):
        r = self.c.get(url)
        self.assertEqual(r.status_code, status, r.text)
        return r.json()

    def post(self, url, body, status=200):
        r = self.c.post(url, json=body)
        self.assertEqual(r.status_code, status, r.text)
        return r.json()

    # ---- basic
    def test_health(self):
        d = self.get("/api/health")
        self.assertEqual((d["status"], d["train_rows"], d["test_rows"]), ("ok", 560, 240))

    def test_overview_schema_and_numbers(self):
        d = self.get("/api/overview")
        for k in ("title", "problem", "headline", "findings", "cost_rule"):
            self.assertIn(k, d)
        r1 = oio.results_dict("part1_results.txt")
        vals = {h["label"]: h["value"] for h in d["headline"]}
        self.assertEqual(vals["Records"] if "Records" in vals else vals["Borrower records"], 800)
        self.assertAlmostEqual(vals["Mean monthly spend"], r1["mean_value"])
        self.assertEqual(vals["Sample needed for INR 500 margin"], r1["n_required"])

    def test_parts(self):
        for n in (1, 2, 3, 4):
            d = self.get("/api/part/%d" % n)
            self.assertEqual(d["part"], n)
            self.assertTrue(d["results"] and d["findings"])
            self.assertEqual(len(d["results"]), len(oio.read_results("part%d_results.txt" % n)))
        self.get("/api/part/9", 404)
        self.get("/api/part/abc", 422)

    def test_part_tables_match_csv(self):
        d = self.get("/api/part/3")
        self.assertEqual(d["tables"]["part3_segments"]["header"][0], "segment")
        self.assertEqual([r[1] for r in d["tables"]["part3_segments"]["rows"]], [322, 255, 223])
        self.assertEqual(len(d["series"]["wcss"]), 5)
        self.assertEqual(len(self.get("/api/part/4")["tables"]["part4_adaboost"]["rows"]), 16)

    def test_memo_and_limitations(self):
        m = self.get("/api/memo")
        self.assertIn("logistic", m["text"])
        self.assertGreaterEqual(self.get("/api/limitations")["text"].count("\n- "), 2)

    # ---- model and threshold
    def test_model_matches_part2(self):
        d = self.get("/api/model")
        self.assertEqual(len(d["features"]), 7)
        self.assertAlmostEqual(d["bias"], self.r2["bias"], places=3)
        for f in d["features"]:
            self.assertAlmostEqual(f["weight"], self.r2["weight_" + f["code"]], places=3)
            self.assertAlmostEqual(f["odds_ratio"], self.r2["odds_ratio_" + f["code"]], places=3)

    def test_threshold_matches_results(self):
        for tau in ("0.5", "0.3"):
            d = self.get("/api/threshold?tau=" + tau)
            for c in ("tp", "fp", "tn", "fn"):
                self.assertEqual(d["confusion"][c], self.r2["tau_%s_%s" % (tau, c)])
            for m in ("accuracy", "precision", "recall", "f1", "specificity"):
                self.assertAlmostEqual(d["metrics"][m], self.r2["tau_%s_%s" % (tau, m)], places=3)

    def test_threshold_cost_rule(self):
        d = self.get("/api/threshold?tau=0.5")
        self.assertEqual(d["cost_inr"], self.r2["cost_test_at_0.5"])
        self.assertEqual(d["cost_inr"], 50 * d["confusion"]["fp"] + 400 * d["confusion"]["fn"])
        self.assertEqual(d["baselines"]["at_cost_tau"], self.r2["cost_test_at_tau"])
        self.assertEqual(d["baselines"]["flag_nobody"], self.r2["cost_test_flag_nobody"])
        self.assertEqual(self.get("/api/threshold?tau=%s" % self.r2["cost_tau"])["cost_inr"],
                         self.r2["cost_test_at_tau"])

    def test_threshold_extremes_and_validation(self):
        self.assertEqual(self.get("/api/threshold?tau=0")["flagged"], 240)
        self.assertEqual(self.get("/api/threshold?tau=1")["flagged"], 0)
        self.get("/api/threshold?tau=1.5", 422)
        self.get("/api/threshold?tau=-0.1", 422)
        self.get("/api/threshold?tau=abc", 422)

    def test_curve_matches_threshold_csv(self):
        d = self.get("/api/curve")
        self.assertEqual(len(d["roc"]), 101)
        self.assertGreater(d["auc"], 0.5)
        tbl = oio.read_table("part2_thresholds.csv")
        for row in tbl["rows"]:
            pt = d["roc"][int(round(row[0] * 100))]
            self.assertAlmostEqual(pt["tpr"], row[5], places=3)
            self.assertAlmostEqual(pt["fpr"], row[6], places=3)

    def test_knn(self):
        d = self.get("/api/knn?k=5")
        for c in ("tp", "fp", "tn", "fn"):
            self.assertEqual(d["confusion"][c], self.r2["knn_" + c])
        self.assertEqual(d["best_k"], self.r2["knn_best_k"])
        cv = oio.read_table("part2_knn_cv.csv")["rows"]
        self.assertEqual([(x["k"], x["f1"]) for x in d["cv"]], [(r[0], r[1]) for r in cv])
        self.get("/api/knn?k=4", 422)
        self.get("/api/knn?k=0", 422)
        self.get("/api/knn?k=23", 422)

    # ---- score
    def test_score(self):
        d = self.post("/api/score", GOOD)
        self.assertTrue(0 < d["probability"] < 1)
        self.assertEqual(len(d["contributions"]), 7)
        self.assertAlmostEqual(sum(c["contribution"] for c in d["contributions"]) + d["bias"], d["logit"], places=4)
        self.assertEqual(d["flagged"], d["probability"] >= d["tau"])

    def test_score_monotone_in_utilisation(self):
        lo = self.post("/api/score", dict(GOOD, X3=0.1))["probability"]
        hi = self.post("/api/score", dict(GOOD, X3=0.9))["probability"]
        self.assertGreater(hi, lo)  # positive weight on X3

    def test_score_validation(self):
        self.post("/api/score", dict(GOOD, X3=1.5), 422)
        self.post("/api/score", dict(GOOD, X6=2), 422)
        self.post("/api/score", dict(GOOD, X1=-1), 422)
        self.post("/api/score", dict(GOOD, X2=2.5), 422)
        self.post("/api/score", dict(GOOD, tau=2), 422)
        bad = dict(GOOD)
        del bad["X5"]
        self.post("/api/score", bad, 422)
        self.post("/api/score", dict(GOOD, X4="abc"), 422)

    # ---- segments
    def test_segments(self):
        d = self.get("/api/segments")
        self.assertEqual([s["segment"] for s in d["segments"]], ["A", "B", "C"])
        self.assertEqual(d["book_mean"]["size"], 800)

    def test_segment_assign(self):
        seg = {s["segment"]: s for s in self.get("/api/segments")["segments"]}
        for name in "ABC":
            s = seg[name]
            body = {"X%d" % j: s["mean_X%d" % j] for j in range(1, 6)}
            self.assertEqual(self.post("/api/segment/assign", body)["segment"], name)
        self.post("/api/segment/assign", {"X1": 1}, 422)
        self.post("/api/segment/assign", {"X1": 1, "X2": 1, "X3": 9, "X4": 1, "X5": 1}, 422)

    # ---- ensemble
    def test_ensemble_reproduces_voting_csv(self):
        rows = {r[0]: r for r in oio.read_table("part4_voting.csv")["rows"]}
        cases = [(["logistic"], "hard", "logistic"), (["knn"], "hard", "knn"), (["stump"], "hard", "stump"),
                 (["logistic", "knn", "stump"], "hard", "hard_vote"), (["logistic", "knn", "stump"], "soft", "soft_vote")]
        for models, mode, key in cases:
            d = self.post("/api/ensemble", {"models": models, "mode": mode, "tau": 0.3})
            got = [d["confusion"][c] for c in ("tp", "fp", "tn", "fn")]
            self.assertEqual(got, rows[key][1:5], key)

    def test_ensemble_validation(self):
        self.post("/api/ensemble", {"models": [], "mode": "hard", "tau": 0.3}, 422)
        self.post("/api/ensemble", {"models": ["svm"], "mode": "hard", "tau": 0.3}, 422)
        self.post("/api/ensemble", {"models": ["knn", "knn"], "mode": "hard", "tau": 0.3}, 422)
        self.post("/api/ensemble", {"models": ["knn"], "mode": "median", "tau": 0.3}, 422)
        self.post("/api/ensemble", {"models": ["knn"], "mode": "soft", "tau": 3}, 422)
        self.assertEqual(self.c.post("/api/ensemble", content="not json",
                                     headers={"Content-Type": "application/json"}).status_code, 422)

    # ---- static
    def test_static_files(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("PhonePe", r.text)
        for path in ("/static/css/style.css", "/static/js/main.js", "/static/js/pages.js", "/static/js/charts.js",
                     "/static/js/api.js", "/static/js/md.js"):
            self.assertEqual(self.c.get(path).status_code, 200, path)
        self.assertEqual(self.c.get("/static/nope.js").status_code, 404)
        self.assertEqual(self.c.get("/api/unknown").status_code, 404)

    def test_html_has_theme_and_viewport(self):
        css = self.c.get("/static/css/style.css").text
        self.assertIn("prefers-color-scheme: dark", css)
        self.assertIn("viewport", self.c.get("/").text)


class OutputsIoTest(unittest.TestCase):
    def test_readme_sections(self):
        for h in ("Part 1 findings", "Part 4 findings", "Model-choice memo", "Limitations"):
            self.assertTrue(oio.readme_section(h), h)
        self.assertEqual(oio.readme_section("No such heading"), "")

    def test_coerce(self):
        self.assertEqual(oio._coerce("12"), 12)
        self.assertEqual(oio._coerce("0.5"), 0.5)
        self.assertEqual(oio._coerce("yes"), "yes")
        self.assertEqual(oio._coerce("S1"), "S1")


if __name__ == "__main__":
    unittest.main()
