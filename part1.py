"""Part 1: estimation, inference and a regression baseline (all 800 rows).

Run from the repository root:  python part1.py
Writes outputs/part1_results.txt
"""
import math

import numpy as np

from common import load_data, mean, sd, write_results


def simple_ols(x, y):
    """Closed form: slope = cov(x, y) / var(x), divisor n - 1."""
    n = len(x)
    mx, my = mean(x), mean(y)
    cov = float(np.sum((x - mx) * (y - my))) / (n - 1)
    var = float(np.sum((x - mx) ** 2)) / (n - 1)
    slope = cov / var
    intercept = my - slope * mx
    return slope, intercept, intercept + slope * x


def half_sds(fitted, resid):
    """Sample SD of residuals in the lower and upper 400 rows ordered by fitted value."""
    order = np.argsort(fitted, kind="stable")
    n = len(order)
    half = n // 2
    lower = resid[order[:half]]
    upper = resid[order[n - half:]]
    return sd(lower), sd(upper)


def main():
    X, y_out, value = load_data()
    n = len(value)
    x4 = X[:, 3]
    out = []

    # 1.1 estimate with an interval
    mean_value = mean(value)
    sd_value = sd(value)
    se_value = sd_value / math.sqrt(n)
    out += [("n", n), ("mean_value", mean_value), ("sd_value", sd_value), ("se_value", se_value),
            ("ci95_low", mean_value - 1.96 * se_value), ("ci95_high", mean_value + 1.96 * se_value)]

    # 1.2 test H0: mu = 8.0
    z_stat = (mean_value - 8.0) / se_value
    out += [("z_stat", z_stat), ("reject_h0", abs(z_stat) > 1.96)]

    # 1.3 sample size for margin E = 0.5
    n_required = math.ceil((1.96 * sd_value / 0.5) ** 2)
    out += [("n_required", n_required)]

    # 1.4 closed-form simple regression of value on X4
    slope, intercept, fitted = simple_ols(x4, value)
    my = mean(value)
    tss = float(np.sum((value - my) ** 2))
    ess = float(np.sum((fitted - my) ** 2))
    resid = value - fitted
    rss = float(np.sum(resid ** 2))
    out += [("reg_slope", slope), ("reg_intercept", intercept), ("tss", tss), ("ess", ess),
            ("rss", rss), ("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss),
            ("r2", 1 - rss / tss), ("rmse", math.sqrt(rss / n)),
            ("mae", float(np.sum(np.abs(resid))) / n)]

    # 1.5 gradient descent on standardised X4
    mx, sx = mean(x4), sd(x4)
    z = (x4 - mx) / sx
    m = 0.0
    c = 0.0
    costs = {}
    for step in range(501):
        e = value - (m * z + c)
        if step in (0, 1, 10, 100, 500):
            costs[step] = float(np.sum(e ** 2)) / n
        if step == 500:
            break
        gm = -2.0 / n * float(np.sum(z * e))
        gc = -2.0 / n * float(np.sum(e))
        m = m - 0.1 * gm
        c = c - 0.1 * gc
    gd_slope = m / sx
    gd_intercept = c - m * mx / sx
    out += [("gd_J0", costs[0]), ("gd_J1", costs[1]), ("gd_J10", costs[10]),
            ("gd_J100", costs[100]), ("gd_J500", costs[500]),
            ("gd_slope", gd_slope), ("gd_intercept", gd_intercept),
            ("gd_slope_gap", abs(gd_slope - slope))]

    # 1.6 funnel check and log remedy
    lo, up = half_sds(fitted, resid)
    out += [("resid_sd_lower_raw", lo), ("resid_sd_upper_raw", up), ("resid_sd_ratio_raw", up / lo)]
    ln_x4 = np.log(x4)
    ln_v = np.log(value)
    l_slope, l_int, l_fit = simple_ols(ln_x4, ln_v)
    l_res = ln_v - l_fit
    l_rss = float(np.sum(l_res ** 2))
    l_tss = float(np.sum((ln_v - mean(ln_v)) ** 2))
    l_r2 = 1 - l_rss / l_tss
    llo, lup = half_sds(l_fit, l_res)
    out += [("loglog_intercept", l_int), ("loglog_slope", l_slope), ("loglog_r2", l_r2),
            ("loglog_rmse", math.sqrt(l_rss / n)),
            ("resid_sd_lower_log", llo), ("resid_sd_upper_log", lup), ("resid_sd_ratio_log", lup / llo)]

    # 1.7 multiple regression by the normal equation
    A = np.column_stack([np.ones(n), ln_x4, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
    beta = np.linalg.solve(A.T @ A, A.T @ ln_v)
    mr_res = ln_v - A @ beta
    mr_r2 = 1 - float(np.sum(mr_res ** 2)) / l_tss
    p = 7
    mr_adj = 1 - (1 - mr_r2) * (n - 1) / (n - p - 1)
    for i in range(8):
        out.append(("mr_beta_{}".format(i), float(beta[i])))
    out += [("mr_r2", mr_r2), ("mr_adj_r2", mr_adj),
            ("loglog_adj_r2", 1 - (1 - l_r2) * (n - 1) / (n - 1 - 1))]

    write_results("part1_results.txt", out)


if __name__ == "__main__":
    main()
