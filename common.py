"""Shared helpers for part1.py .. part4.py. Importing this module runs nothing.

Everything graded is computed here or in the part scripts with plain Python and
NumPy array arithmetic (no ready-made regression, KNN, k-means, PCA or metric routines).
"""
import csv
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "records.csv")
OUT_DIR = os.path.join(ROOT, "outputs")

PREDICTORS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]


# ------------------------------------------------------------------ data and output

def load_data():
    """Return X (800 x 7 float), y (outcome, int array) and value (float array)."""
    with open(DATA_PATH, newline="") as f:
        rows = list(csv.reader(f))
    body = rows[1:]
    X = np.array([[float(r[j]) for j in range(1, 8)] for r in body])
    y = np.array([int(r[8]) for r in body])
    value = np.array([float(r[9]) for r in body])
    return X, y, value


def fmt(v):
    """Format a value for results/CSV files: ints as ints, bool yes/no, floats 4 dp."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        s = "{:.4f}".format(float(v))
        if s == "-0.0000":
            s = "0.0000"
        return s
    return str(v)


def write_results(name, pairs):
    """Write outputs/<name> as `key = value` lines (LF endings)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
        for k, v in pairs:
            f.write("{} = {}\n".format(k, fmt(v)))


def write_csv(name, header, rows):
    """Write outputs/<name> with the exact header, comma separator, LF endings."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


# ------------------------------------------------------------------ statistics

def mean(a):
    return float(np.sum(a)) / len(a)


def sd(a):
    """Sample standard deviation (divide by n - 1)."""
    m = mean(a)
    return math.sqrt(float(np.sum((a - m) ** 2)) / (len(a) - 1))


def col_stats(M):
    """Column means and sample SDs of a 2-D array."""
    n = M.shape[0]
    m = np.sum(M, axis=0) / n
    s = np.sqrt(np.sum((M - m) ** 2, axis=0) / (n - 1))
    return m, s


# ------------------------------------------------------------------ classification

def stratified_split(y):
    """Task 2.1: first floor(7*count/10) rows of each class go to training."""
    train = []
    test = []
    for cls in (0, 1):
        idx = [i for i in range(len(y)) if y[i] == cls]
        cut = (7 * len(idx)) // 10
        train.extend(idx[:cut])
        test.extend(idx[cut:])
    return np.array(sorted(train)), np.array(sorted(test))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))) / len(y)


def train_logistic(X, y, eta, updates, record_at=()):
    """Batch gradient descent from zero. Returns weights, bias and {update: loss}."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = {}
    for step in range(updates + 1):
        p = sigmoid(X @ w + b)
        if step in record_at or step == updates:
            losses[step] = bce(y, p)
        if step == updates:
            break
        err = p - y
        gw = (X.T @ err) / n
        gb = float(np.sum(err)) / n
        w = w - eta * gw
        b = b - eta * gb
    return w, b, losses


def confusion(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred).astype(bool)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return tp, fp, tn, fn


def _div(a, b):
    return a / b if b != 0 else 0.0


def metrics(tp, fp, tn, fn):
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    return {
        "accuracy": _div(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": _div(2 * precision * recall, precision + recall),
        "specificity": _div(tn, tn + fp),
        "fpr": _div(fp, fp + tn),
    }


def knn_prob(Xref, yref, Xq, K):
    """Share of positives among the K nearest reference rows (stable tie-break)."""
    diff = Xq[:, None, :] - Xref[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    order = np.argsort(dist, axis=1, kind="stable")[:, :K]
    return np.sum(yref[order], axis=1) / K


def knn_predict(Xref, yref, Xq, K):
    """Positive when at least (K + 1) / 2 of the K neighbours are positive (K odd)."""
    diff = Xq[:, None, :] - Xref[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    order = np.argsort(dist, axis=1, kind="stable")[:, :K]
    return np.sum(yref[order], axis=1) >= (K + 1) // 2


def prepare_split():
    """Task 2.1 and 2.2 objects shared by Part 2 and Part 4 (deterministic)."""
    X, y, value = load_data()
    tr, te = stratified_split(y)
    m, s = col_stats(X[tr])
    Ztr = (X[tr] - m) / s
    Zte = (X[te] - m) / s
    w, b, losses = train_logistic(Ztr, y[tr], 0.1, 1000, record_at=(0, 1, 10, 100, 1000))
    return {
        "X": X, "y": y, "value": value, "train_idx": tr, "test_idx": te,
        "Ztr": Ztr, "Zte": Zte, "ytr": y[tr], "yte": y[te],
        "w": w, "b": b, "losses": losses,
        "p_train": sigmoid(Ztr @ w + b), "p_test": sigmoid(Zte @ w + b),
    }


def knn_cv_table(Ztr, ytr):
    """Task 2.7: 5-fold CV (fold = position mod 5), mean held-out F1 for odd K 1..21."""
    n = len(ytr)
    fold = np.arange(n) % 5
    table = []
    for K in range(1, 22, 2):
        f1s = []
        for f in range(5):
            hold = fold == f
            ref = ~hold
            pred = knn_predict(Ztr[ref], ytr[ref], Ztr[hold], K)
            tp, fp, tn, fn = confusion(ytr[hold], pred)
            f1s.append(metrics(tp, fp, tn, fn)["f1"])
        table.append((K, sum(f1s) / 5))
    return table


def choose_knn_k(table):
    """Highest mean F1; ties go to the smaller K."""
    best_k, best = None, -1.0
    for K, score in table:
        if score > best:
            best_k, best = K, score
    return best_k
