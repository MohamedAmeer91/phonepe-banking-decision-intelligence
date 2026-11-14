"""Cached model state built once from common.py (same split and models as Parts 2-4)."""
import os
import sys
from functools import lru_cache

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import common  # noqa: E402  (graded helpers; import has no side effects)
import part3  # noqa: E402  (for the pinned K-means; import has no side effects)

FEATURES = [
    ("X1", "account_age_months", "Account age (months)"),
    ("X2", "failed_txn_count_90d", "Failed transactions, last 90 days"),
    ("X3", "credit_utilisation_ratio", "Credit utilisation (0-1)"),
    ("X4", "avg_monthly_inflow_inr_thousands", "Average monthly inflow (INR thousands)"),
    ("X5", "repayment_score", "Repayment score (0-100)"),
    ("X6", "irregular_salary_flag", "Irregular salary (0/1)"),
    ("X7", "autopay_flag", "Autopay set up (0/1)"),
]
COST_FALSE_ALARM = 50
COST_MISS = 400
ALL_KS = list(range(1, 22, 2))


def cost_of(y, pred):
    tp, fp, tn, fn = common.confusion(y, pred)
    return COST_FALSE_ALARM * fp + COST_MISS * fn


def auc(tpr, fpr):
    order = np.argsort(fpr, kind="stable")
    x, y = np.asarray(fpr)[order], np.asarray(tpr)[order]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


class State:
    def __init__(self):
        s = common.prepare_split()
        self.s = s
        self.yte, self.ytr = s["yte"], s["ytr"]
        self.p_test, self.p_train = s["p_test"], s["p_train"]
        self.w, self.b = s["w"], s["b"]
        self.mu, self.sd = common.col_stats(s["X"][s["train_idx"]])
        self.cv = common.knn_cv_table(s["Ztr"], self.ytr)
        self.best_k = common.choose_knn_k(self.cv)
        self._knn_pred = {}
        self.knn_p = common.knn_prob(s["Ztr"], self.ytr, s["Zte"], self.best_k)
        Xtr, ytr = s["X"][s["train_idx"]], self.ytr
        yes = Xtr[:, 0] <= 24  # S4, the best split by information gain (Part 4)
        self.rate_yes = float(np.sum(ytr[yes])) / int(np.sum(yes))
        self.rate_no = float(np.sum(ytr[~yes])) / int(np.sum(~yes))
        self.stump_p = np.where(s["X"][s["test_idx"]][:, 0] <= 24, self.rate_yes, self.rate_no)
        X5 = s["X"][:, :5]
        self.seg_mu, self.seg_sd = common.col_stats(X5)
        labels, cent, _, _ = part3.kmeans((X5 - self.seg_mu) / self.seg_sd, 3)
        raw = np.array([np.mean(X5[labels == k][:, 2]) for k in range(3)])
        self.seg_order = [int(k) for k in np.argsort(raw, kind="stable")]  # ascending X3 -> A, B, C
        self.seg_cent = cent
        self.curve = self._curve()

    def knn_pred(self, k):
        if k not in self._knn_pred:
            s = self.s
            self._knn_pred[k] = common.knn_predict(s["Ztr"], self.ytr, s["Zte"], k)
        return self._knn_pred[k]

    def block(self, y, pred):
        tp, fp, tn, fn = common.confusion(y, pred)
        return ({"tp": tp, "fp": fp, "tn": tn, "fn": fn}, common.metrics(tp, fp, tn, fn))

    def _curve(self):
        roc, cost = [], []
        for i in range(101):
            tau = i / 100
            tp, fp, tn, fn = common.confusion(self.yte, self.p_test >= tau)
            m = common.metrics(tp, fp, tn, fn)
            roc.append({"tau": tau, "tpr": m["recall"], "fpr": m["fpr"]})
            cost.append({"tau": tau, "test": cost_of(self.yte, self.p_test >= tau),
                         "train": cost_of(self.ytr, self.p_train >= tau)})
        return {"roc": roc, "cost": cost,
                "auc": auc([r["tpr"] for r in roc], [r["fpr"] for r in roc])}

    def score(self, raw):
        z = (np.array(raw, dtype=float) - self.mu) / self.sd
        contrib = z * self.w
        logit = float(np.sum(contrib) + self.b)
        return float(common.sigmoid(logit)), logit, z, contrib

    def assign(self, raw5):
        z = (np.array(raw5, dtype=float) - self.seg_mu) / self.seg_sd
        d = np.sqrt(np.sum((self.seg_cent - z) ** 2, axis=1))
        dist = {"ABC"[r]: float(d[k]) for r, k in enumerate(self.seg_order)}
        return min(dist, key=dist.get), dist

    def ensemble(self, models, mode, tau):
        if mode == "soft":
            probs = {"logistic": self.p_test, "knn": self.knn_p, "stump": self.stump_p}
            pred = sum(probs[m] for m in models) / len(models) >= tau
        else:
            votes = {"logistic": self.p_test >= tau, "knn": self.knn_pred(self.best_k),
                     "stump": self.stump_p >= tau}
            total = sum(votes[m].astype(int) for m in models)
            pred = total * 2 > len(models)
        return self.block(self.yte, pred)


@lru_cache(maxsize=1)
def get_state():
    return State()
