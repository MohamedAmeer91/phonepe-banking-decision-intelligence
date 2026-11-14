"""Part 3: segmentation and dimensionality reduction (all 800 rows, X1..X5).

Run from the repository root:  python part3.py
Writes outputs/part3_results.txt, part3_segments.csv, part3_hierarchical.csv
"""
import numpy as np

from common import col_stats, load_data, mean, write_csv, write_results

NAMES = ["E1", "E2", "E3", "E4", "E5", "E6"]
DISSIM = [
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
]


def kmeans(Z, K):
    """K-means with the pinned start rows; returns labels, centroids, passes, wcss."""
    n = Z.shape[0]
    step = n // K
    cent = Z[[i * step for i in range(K)]].copy()
    labels = np.full(n, -1)
    passes = 0
    while True:
        passes += 1
        diff = Z[:, None, :] - cent[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        new = np.argmin(d2, axis=1)  # first minimum = lower-numbered centroid
        if np.array_equal(new, labels):
            break
        labels = new
        for k in range(K):
            members = Z[labels == k]
            if len(members) > 0:
                cent[k] = np.sum(members, axis=0) / len(members)
    wcss = float(np.sum((Z - cent[labels]) ** 2))
    return labels, cent, passes, wcss


def linkage_distance(D, a, b, rule):
    ds = [D[i][j] for i in a for j in b]
    if rule == "single":
        return min(ds)
    if rule == "complete":
        return max(ds)
    return sum(ds) / len(ds)


def label(cluster):
    return "+".join(NAMES[i] for i in sorted(cluster))


def agglomerate(D, rule):
    """Returns merges [(cluster_a, cluster_b, height)] for the given linkage rule."""
    clusters = [(i,) for i in range(len(D))]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                h = linkage_distance(D, clusters[i], clusters[j], rule)
                if best is None or h < best[0]:
                    best = (h, i, j)
        h, i, j = best
        a, b = clusters[i], clusters[j]
        if min(b) < min(a):
            a, b = b, a
        merges.append((a, b, h))
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [tuple(sorted(a + b))]
    return merges


def clusters_at_cut(merges):
    heights = [m[2] for m in merges]
    gaps = [heights[i + 1] - heights[i] for i in range(len(heights) - 1)]
    g = max(range(len(gaps)), key=lambda i: (gaps[i], -i))  # first largest gap
    below = g + 1  # merges 1..g+1 lie below the cut
    return 6 - below


def main():
    X, y, value = load_data()
    X5 = X[:, :5]
    n = X5.shape[0]
    mu, sdv = col_stats(X5)
    Z = (X5 - mu) / sdv
    out = []

    # 3.1 K-means for K = 2..6
    wcss = {}
    runs = {}
    for K in range(2, 7):
        labels, cent, passes, w = kmeans(Z, K)
        wcss[K] = w
        runs[K] = labels
        out.append(("wcss_k{}".format(K), w))

    # 3.2 elbow rule
    elbow = 5
    for K in (3, 4, 5):
        if wcss[K] - wcss[K + 1] < 0.5 * (wcss[K - 1] - wcss[K]):
            elbow = K
            break
    out.append(("elbow_k", elbow))

    # 3.3 profile the K = 3 segments (ordered by ascending mean X3)
    lab3 = runs[3]
    order = sorted(range(3), key=lambda k: mean(X5[lab3 == k][:, 2]))
    seg_of = np.zeros(n, dtype=int)
    seg_rows = []
    for rank, k in enumerate(order):
        mask = lab3 == k
        seg_of[mask] = rank
        means = [mean(X5[mask][:, j]) for j in range(5)]
        seg_rows.append(["ABC"[rank], int(np.sum(mask))] + means +
                        [float(np.sum(y[mask])) / int(np.sum(mask)), mean(value[mask])])
    for rank in range(3):
        out.append(("size_segment_" + "ABC"[rank], seg_rows[rank][1]))
    write_csv("part3_segments.csv",
              ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
               "outcome_rate", "mean_value"], seg_rows)

    # 3.4 hierarchical clustering on the supplied table
    hier_rows = []
    cuts = []
    for rule in ("single", "complete", "average"):
        merges = agglomerate(DISSIM, rule)
        for no, (a, b, h) in enumerate(merges, start=1):
            hier_rows.append([rule, no, label(a), label(b), h])
        cuts.append(clusters_at_cut(merges))
    write_csv("part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
              hier_rows)
    out += [("single_clusters_at_cut", cuts[0]), ("complete_clusters_at_cut", cuts[1]),
            ("average_clusters_at_cut", cuts[2])]

    # 3.5 PCA by the seven steps
    mean_vec = np.sum(Z, axis=0) / n          # step 1 (zeros after standardising)
    Xc = Z - mean_vec                         # step 2
    S = (Xc.T @ Xc) / (n - 1)                 # step 3
    vals, vecs = np.linalg.eigh(S)            # step 4
    idx = np.argsort(-vals, kind="stable")    # step 5
    vals = vals[idx]
    vecs = vecs[:, idx]
    for j in range(5):                        # sign convention
        if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
            vecs[:, j] = -vecs[:, j]
    W = vecs[:, :2]                           # step 6
    P = Xc @ W                                # step 7
    total = float(np.sum(vals))
    out.append(("eigenvalue_sum", total))
    for j in range(5):
        out.append(("eigenvalue_{}".format(j + 1), float(vals[j])))
    ratios = [float(v) / total for v in vals]
    for j in range(5):
        out.append(("explained_ratio_{}".format(j + 1), ratios[j]))
    cum = 0.0
    need = None
    for j in range(5):
        cum += ratios[j]
        out.append(("cumulative_ratio_{}".format(j + 1), cum))
        if need is None and cum >= 0.8:
            need = j + 1
    out.append(("components_for_80pct", need))
    for c in (0, 1):
        for j in range(5):
            out.append(("pc{}_loading_X{}".format(c + 1, j + 1), float(W[j, c])))
    for r in range(3):
        out.append(("row{}_pc1".format(r + 1), float(P[r, 0])))
        out.append(("row{}_pc2".format(r + 1), float(P[r, 1])))
    for rank in range(3):
        out.append(("mean_pc1_segment_" + "ABC"[rank], mean(P[seg_of == rank][:, 0])))

    write_results("part3_results.txt", out)


if __name__ == "__main__":
    main()
