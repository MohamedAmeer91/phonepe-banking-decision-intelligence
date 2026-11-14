"""Part 4: model-choice and architecture memo numbers.

Run from the repository root:  python part4.py
Writes outputs/part4_results.txt, part4_splits.csv, part4_voting.csv,
part4_adaboost.csv, part4_architecture.csv

Uses the Task 2.1 split and the Task 2.2 / 2.7 models through common.py
(the same code Part 2 runs), so nothing in Part 2 is re-run by importing.
"""
import math

import numpy as np

from common import (choose_knn_k, confusion, knn_cv_table, knn_predict, knn_prob, metrics, prepare_split,
                    write_csv, write_results)


def entropy(pos, n):
    if n == 0:
        return 0.0
    total = 0.0
    for c in (pos, n - pos):
        if c > 0:
            p = c / n
            total -= p * math.log2(p)
    return total


def gini(pos, n):
    if n == 0:
        return 0.0
    p = pos / n
    return 1.0 - p * p - (1 - p) * (1 - p)


def conv_out(size, filt, pad, stride):
    return (size - filt + 2 * pad) // stride + 1


def main():
    s = prepare_split()
    X, y = s["X"], s["y"]
    tr, te = s["train_idx"], s["test_idx"]
    Xtr_o, Xte_o = X[tr], X[te]
    ytr, yte = s["ytr"], s["yte"]
    out = []

    # 4.1 which split is purest (original units, training rows)
    n_tr, pos_tr = len(ytr), int(np.sum(ytr))
    root_e, root_g = entropy(pos_tr, n_tr), gini(pos_tr, n_tr)
    out += [("root_entropy", root_e), ("root_gini", root_g)]
    conditions = {
        "S1": lambda M: M[:, 5] == 1,
        "S2": lambda M: M[:, 6] == 1,
        "S3": lambda M: M[:, 2] > 0.5,
        "S4": lambda M: M[:, 0] <= 24,
    }
    split_rows = []
    best_info, best_gini = None, None
    for name in ("S1", "S2", "S3", "S4"):
        yes = conditions[name](Xtr_o)
        n_yes, p_yes = int(np.sum(yes)), int(np.sum(ytr[yes]))
        n_no, p_no = n_tr - n_yes, pos_tr - p_yes
        ig = root_e - (n_yes / n_tr * entropy(p_yes, n_yes) + n_no / n_tr * entropy(p_no, n_no))
        gg = root_g - (n_yes / n_tr * gini(p_yes, n_yes) + n_no / n_tr * gini(p_no, n_no))
        split_rows.append([name, n_yes, p_yes, n_no, p_no, ig, gg])
        if best_info is None or ig > best_info[1]:
            best_info = (name, ig, n_yes, p_yes, n_no, p_no)
        if best_gini is None or gg > best_gini[1]:
            best_gini = (name, gg)
    write_csv("part4_splits.csv",
              ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
              split_rows)
    out += [("best_split_by_info_gain", best_info[0]), ("best_split_by_gini_gain", best_gini[0])]

    # 4.2 stump and five test-set predictions
    _, _, _, p_yes, n_no, p_no = best_info
    n_yes = best_info[2]
    rate_yes, rate_no = p_yes / n_yes, p_no / n_no
    out += [("stump_rate_yes", rate_yes), ("stump_rate_no", rate_no)]
    stump_yes_test = conditions[best_info[0]](Xte_o)
    p_stump = np.where(stump_yes_test, rate_yes, rate_no)
    p_log = s["p_test"]
    table = knn_cv_table(s["Ztr"], ytr)
    best_k = choose_knn_k(table)
    p_knn = knn_prob(s["Ztr"], ytr, s["Zte"], best_k)
    pred_log = p_log >= 0.3
    pred_knn = knn_predict(s["Ztr"], ytr, s["Zte"], best_k)
    pred_stump = p_stump >= 0.3
    pred_hard = (pred_log.astype(int) + pred_knn.astype(int) + pred_stump.astype(int)) >= 2
    pred_soft = (p_log + p_knn + p_stump) / 3 >= 0.3
    vote_rows = []
    for name, pred in (("logistic", pred_log), ("knn", pred_knn), ("stump", pred_stump),
                       ("hard_vote", pred_hard), ("soft_vote", pred_soft)):
        tp, fp, tn, fn = confusion(yte, pred)
        m = metrics(tp, fp, tn, fn)
        vote_rows.append([name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]])
    write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
              vote_rows)

    # 4.3 two rounds of AdaBoost by hand
    yb = [1, 1, 1, -1, -1, -1, 1, -1]
    h1 = [1, 1, -1, -1, -1, -1, -1, -1]
    h2 = [-1, 1, 1, -1, 1, -1, 1, -1]
    w = [1 / 8] * 8
    ada_rows = []
    alphas = []
    errors = []
    for rnd, h in ((1, h1), (2, h2)):
        eps = sum(w[i] for i in range(8) if h[i] != yb[i])
        alpha = 0.5 * math.log((1 - eps) / eps)
        unnorm = [w[i] * math.exp(-alpha * yb[i] * h[i]) for i in range(8)]
        z = sum(unnorm)
        after = [u / z for u in unnorm]
        for i in range(8):
            ada_rows.append([rnd, i + 1, w[i], unnorm[i], after[i]])
        errors.append(eps)
        alphas.append(alpha)
        w = after
    write_csv("part4_adaboost.csv",
              ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], ada_rows)
    correct = 0
    for i in range(8):
        score = alphas[0] * h1[i] + alphas[1] * h2[i]
        if (1 if score > 0 else -1 if score < 0 else 0) == yb[i]:
            correct += 1
    out += [("round1_error", errors[0]), ("round1_alpha", alphas[0]),
            ("round2_error", errors[1]), ("round2_alpha", alphas[1]),
            ("ensemble_accuracy", correct / 8)]

    # 4.4 out-of-bag share
    out.append(("oob_fraction", (1 - 1 / 560) ** 560))

    # 4.5 parameter counts (nothing is trained)
    mlp = [("mlp_hidden1", 784 * 16 + 16), ("mlp_hidden2", 16 * 16 + 16), ("mlp_output", 16 * 10 + 10)]
    side = 28
    k1, f1 = 5, 6
    conv1 = (k1 * k1 * 1 + 1) * f1
    side = conv_out(side, 5, 2, 1)       # 28
    side = conv_out(side, 2, 0, 2)       # 14
    k2, f2 = 5, 16
    conv2 = (k2 * k2 * f1 + 1) * f2
    side = conv_out(side, 5, 0, 1)       # 10
    side = conv_out(side, 2, 0, 2)       # 5
    flat = side * side * f2              # 400
    dense1 = flat * 32 + 32
    dense2 = 32 * 10 + 10
    cnn = [("cnn_conv1", conv1), ("cnn_conv2", conv2), ("cnn_dense1", dense1), ("cnn_dense2", dense2)]
    write_csv("part4_architecture.csv", ["layer", "parameters"], mlp + cnn)
    fc_layer = (128 * 128 * 3) * 1024 + 1024
    conv_layer = (3 * 3 * 3 + 1) * 64
    rnn = 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16
    out += [("mlp_total", sum(v for _, v in mlp)), ("cnn_total", sum(v for _, v in cnn)),
            ("fc_layer_params", fc_layer), ("conv_layer_params", conv_layer), ("rnn_total", rnn)]

    write_results("part4_results.txt", out)


if __name__ == "__main__":
    main()
