"""Part 2: classification and evaluation.

Run from the repository root:  python part2.py
Writes outputs/part2_results.txt, part2_thresholds.csv, part2_knn_cv.csv
"""
import math

import numpy as np

from common import (PREDICTORS, bce, choose_knn_k, confusion, knn_cv_table, knn_predict,
                    metrics, prepare_split, train_logistic, write_csv, write_results)

CELL_KEYS = ["tp", "fp", "tn", "fn"]
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "specificity"]


def threshold_block(prefix, yte, p_test, tau):
    tp, fp, tn, fn = confusion(yte, p_test >= tau)
    m = metrics(tp, fp, tn, fn)
    pairs = [(prefix + k, v) for k, v in zip(CELL_KEYS, (tp, fp, tn, fn))]
    pairs += [(prefix + k, m[k]) for k in METRIC_KEYS]
    return pairs


def cost(y, pred):
    tp, fp, tn, fn = confusion(y, pred)
    return 50 * fp + 400 * fn


def main():
    s = prepare_split()
    ytr, yte, Ztr, Zte = s["ytr"], s["yte"], s["Ztr"], s["Zte"]
    out = []

    # 2.1 split
    out += [("train_rows", len(ytr)), ("train_positives", int(np.sum(ytr))),
            ("test_rows", len(yte)), ("test_positives", int(np.sum(yte)))]

    # 2.2 logistic regression
    w, b, losses = s["w"], s["b"], s["losses"]
    for e in (0, 1, 10, 100, 1000):
        out.append(("loss_epoch_{}".format(e), losses[e]))
    for j, name in enumerate(PREDICTORS):
        out.append(("weight_" + name, float(w[j])))
    out.append(("bias", b))
    out.append(("test_bce", bce(yte, s["p_test"])))

    # 2.3 learning-rate effect
    for eta_text, eta in (("0.01", 0.01), ("0.1", 0.1), ("1.0", 1.0)):
        _, _, ls = train_logistic(Ztr, ytr, eta, 200)
        out.append(("loss_epoch_200_eta_" + eta_text, ls[200]))

    # 2.4 odds ratios
    for j, name in enumerate(PREDICTORS):
        out.append(("odds_ratio_" + name, math.exp(float(w[j]))))

    # 2.5 two thresholds and blind accuracy
    p_test = s["p_test"]
    out += threshold_block("tau_0.5_", yte, p_test, 0.5)
    out += threshold_block("tau_0.3_", yte, p_test, 0.3)
    out.append(("blind_accuracy", float(np.sum(yte == 0)) / len(yte)))

    # 2.6 threshold sweep
    rows = []
    for k in range(1, 10):
        tau = k / 10
        tp, fp, tn, fn = confusion(yte, p_test >= tau)
        m = metrics(tp, fp, tn, fn)
        rows.append((tau, tp, fp, tn, fn, m["recall"], m["fpr"]))
    write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

    # 2.7 KNN with cross-validation
    table = knn_cv_table(Ztr, ytr)
    write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], table)
    best_k = choose_knn_k(table)
    tp, fp, tn, fn = confusion(yte, knn_predict(Ztr, ytr, Zte, best_k))
    m = metrics(tp, fp, tn, fn)
    out.append(("knn_best_k", best_k))
    out += [("knn_" + k, v) for k, v in zip(CELL_KEYS, (tp, fp, tn, fn))]
    out += [("knn_" + k, m[k]) for k in METRIC_KEYS[:4]]

    # 2.8 cost-based threshold, chosen on the training rows only
    p_train = s["p_train"]
    best_tau, best_cost = None, None
    for k in range(1, 20):
        tau = k / 20
        c = cost(ytr, p_train >= tau)
        if best_cost is None or c < best_cost:
            best_tau, best_cost = tau, c
    out += [("cost_tau", best_tau), ("cost_train_at_tau", best_cost),
            ("cost_test_at_tau", cost(yte, p_test >= best_tau)),
            ("cost_test_at_0.5", cost(yte, p_test >= 0.5)),
            ("cost_test_flag_nobody", cost(yte, np.zeros(len(yte), dtype=bool)))]

    write_results("part2_results.txt", out)


if __name__ == "__main__":
    main()
