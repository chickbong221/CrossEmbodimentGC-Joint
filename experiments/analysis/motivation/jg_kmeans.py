import numpy as np


def kmeans(x, k, seed=0, n_init=50, iters=300):
    """Lloyd's k-means with k-means++ starts; the best of n_init runs (deterministic for a seed)."""
    rs = np.random.RandomState(seed)
    best = None
    for _ in range(n_init):
        centres = [x[rs.randint(len(x))]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
            centres.append(x[rs.choice(len(x), p=d2 / d2.sum())] if d2.sum() > 0 else x[rs.randint(len(x))])
        c = np.array(centres)
        for _ in range(iters):
            lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)
            new = np.array([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)
        inertia = float(((x - c[lab]) ** 2).sum())
        if best is None or inertia < best[0] - 1e-9:
            best = (inertia, lab)
    # Relabel groups by first appearance, so the table does not depend on the order of the centres.
    order = {}
    for g in best[1]:
        order.setdefault(int(g), len(order))
    return np.array([order[int(g)] for g in best[1]]), best[0]


