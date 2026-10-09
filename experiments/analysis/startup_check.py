"""Start-up check: does later training copy the falling episodes more closely in the first steps of an episode?

Input: the start-up dump of morse/coordination.py (``morse_coord_startup=N``): the first N steps of every complete
episode, with the dataset action, both checkpoints' action means (A earlier, B later), and both checkpoints' advantage
weights of the dataset action. Groups: expert = reference episodes; falling = episodes that end in termination.

1. Fit to each group's own actions: MSE of each checkpoint's mean vs the logged action, by step bin, with the change
   B - A and a 95% bootstrap interval over episodes.
2. Next to the expert's path: every falling transition is matched to the nearest expert transition at a step within
   +-2 (standardized joint positions/velocities, contacts, trunk velocities, gravity, height). "Near" = closer than the
   90th percentile of expert-to-expert nearest distances (other episodes, same step window). For near transitions:
   distance of each checkpoint's mean to the falling action (D_f) and to the matched expert action (D_e), and
   P = D_e - D_f (> 0: closer to the falling action than to the expert's); the change P_B - P_A (> 0: later training
   moved toward the falling action) with a bootstrap interval over falling episodes.
3. Critic: mean advantage weights of expert and falling transitions, and the expert share of the total weight vs its
   share of transitions, by step bin; for near transitions, the weight of the falling action vs the matched expert's.

Usage: python analysis/startup_check.py DUMP_DIR [--robots cas,g1] [--out results.json]
"""
import argparse
import json
import os

import numpy as np

BINS = ((0, 10), (10, 25), (25, 50))


def boot(values_by_group, n=2000, seed=0):
    """Mean of per-transition values and a 95% interval, resampling the groups (episodes)."""
    groups = [v for v in values_by_group if len(v)]
    if not groups:
        return None, [None, None]
    rng = np.random.default_rng(seed)
    sums, cnts = np.array([v.sum() for v in groups]), np.array([len(v) for v in groups])
    stats = []
    for _ in range(n):
        k = rng.integers(0, len(groups), len(groups))
        stats.append(sums[k].sum() / cnts[k].sum())
    return float(sums.sum() / cnts.sum()), [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def by_episode(values, episode):
    order = np.argsort(episode, kind="stable")
    e, v = episode[order], values[order]
    cuts = np.flatnonzero(np.diff(e)) + 1
    return np.split(v, cuts)


def nearest(q, c, chunk=4096):
    """Index and squared distance of the nearest row of c for every row of q."""
    cn = (c ** 2).sum(1)
    idx, d2 = np.empty(len(q), np.int64), np.empty(len(q))
    for i in range(0, len(q), chunk):
        x = q[i:i + chunk]
        d = (x ** 2).sum(1)[:, None] + cn[None, :] - 2.0 * x @ c.T
        idx[i:i + chunk] = d.argmin(1)
        d2[i:i + chunk] = np.maximum(d[np.arange(len(x)), idx[i:i + chunk]], 0.0)
    return idx, d2


def analyze(path, seed):
    z = np.load(path)
    robot = str(z["robot"])
    t, ep, ref = z["t"], z["episode"], z["reference"]
    falling = ~ref & z["ep_terminated"][ep]
    act = z["act"].astype(np.float64)
    mu = {"A": z["mean_A"].astype(np.float64), "B": z["mean_B"].astype(np.float64)}
    w = {"A": z["w_A"].astype(np.float64), "B": z["w_B"].astype(np.float64)}
    D = act.shape[1]
    sq = {k: ((m - act) ** 2).mean(1) for k, m in mu.items()}
    out = {"robot": robot, "joints": D, "transitions": {"expert": int(ref.sum()), "falling": int(falling.sum())},
           "episodes": {"expert": int(len(np.unique(ep[ref]))), "falling": int(len(np.unique(ep[falling])))}}

    # 1. Fit to each group's own logged actions.
    fit = {}
    for lo, hi in BINS:
        row = {}
        for name, g in (("expert", ref), ("falling", falling)):
            m = g & (t >= lo) & (t < hi)
            row[name] = {"mse_A": float(sq["A"][m].mean()) if m.any() else None,
                         "mse_B": float(sq["B"][m].mean()) if m.any() else None}
            d, ci = boot(by_episode((sq["B"] - sq["A"])[m], ep[m]), seed=seed)
            row[name]["change"], row[name]["change_ci"] = d, ci
        fit[f"{lo}-{hi}"] = row
    out["fit"] = fit

    # 2. Falling transitions next to the expert's path.
    gen = z["general"].astype(np.float64)
    feats = np.concatenate([z["joint_pos"], z["joint_vel"], z["contact"]], axis=1).astype(np.float64)
    feats = np.concatenate([feats, gen[:, [0, 1, 2, 3, 4, 5, 9, 10, 11, 12]]], axis=1)
    sd = feats[ref].std(0)
    feats = feats / np.where(sd > 1e-6, sd, 1.0)
    T = int(t.max()) + 1
    nn_idx, nn_d2 = np.full(len(t), -1), np.full(len(t), np.inf)
    ref_d2 = []
    for s in range(T):
        cand = np.flatnonzero(ref & (np.abs(t - s) <= 2))
        if len(cand) < 2:
            continue
        q = np.flatnonzero(falling & (t == s))
        if len(q):
            i, d2 = nearest(feats[q], feats[cand])
            nn_idx[q], nn_d2[q] = cand[i], d2
        # expert-to-expert distances (nearest transition of another episode), for the "near" threshold
        qe = np.flatnonzero(ref & (t == s))
        if len(qe):
            cf = feats[cand]
            for j in range(0, len(qe), 2048):
                x = feats[qe[j:j + 2048]]
                d = (x ** 2).sum(1)[:, None] + (cf ** 2).sum(1)[None, :] - 2.0 * x @ cf.T
                d[ep[qe[j:j + 2048]][:, None] == ep[cand][None, :]] = np.inf
                ref_d2.append(np.maximum(d.min(1), 0.0))
    ref_d2 = np.concatenate(ref_d2)
    thr = float(np.percentile(ref_d2, 90))
    near = falling & (nn_idx >= 0) & (nn_d2 <= thr)
    out["near_path"] = {"threshold_d2": thr, "expert_nn_d2_median": float(np.median(ref_d2)),
                        "falling_nn_d2_median": float(np.median(nn_d2[falling & (nn_idx >= 0)])),
                        "share_of_falling_near": float(near.sum() / max(falling.sum(), 1)), "by_bin": {}}
    a_e = np.where(near[:, None], act[np.maximum(nn_idx, 0)], np.nan)
    for lo, hi in BINS:
        m = near & (t >= lo) & (t < hi)
        if m.sum() < 50:
            continue
        row = {"transitions": int(m.sum()), "episodes": int(len(np.unique(ep[m])))}
        P = {}
        for k in ("A", "B"):
            df = ((mu[k][m] - act[m]) ** 2).mean(1)
            de = ((mu[k][m] - a_e[m]) ** 2).mean(1)
            row[f"D_falling_{k}"], row[f"D_expert_{k}"] = float(df.mean()), float(de.mean())
            row[f"closer_to_falling_{k}"] = float(np.mean(df < de))
            P[k] = de - df
        d, ci = boot(by_episode(P["B"] - P["A"], ep[m]), seed=seed + 1)
        row["dP"], row["dP_ci"] = d, ci
        # critic: weight of the falling action vs the matched expert's action at a nearby state
        for k in ("A", "B"):
            row[f"w_falling_{k}"] = float(w[k][m].mean())
            row[f"w_expert_match_{k}"] = float(w[k][nn_idx[m]].mean())
        out["near_path"]["by_bin"][f"{lo}-{hi}"] = row

    # 3. Critic weights by group.
    crit = {}
    for lo, hi in BINS:
        mb = (t >= lo) & (t < hi)
        row = {"raw_expert_share": float((ref & mb).sum() / max(((ref | falling) & mb).sum(), 1))}
        for k in ("A", "B"):
            we, wf = w[k][ref & mb], w[k][falling & mb]
            row[f"w_expert_{k}"], row[f"w_falling_{k}"] = float(we.mean()), float(wf.mean())
            row[f"weighted_expert_share_{k}"] = float(we.sum() / (we.sum() + wf.sum()))
        crit[f"{lo}-{hi}"] = row
    out["critic"] = crit
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump_dir")
    p.add_argument("--robots", default="cas,nao5,g1,h1,tl,op3,go2,anyb")
    p.add_argument("--out", default="")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    res = {}
    for r in [x for x in a.robots.split(",") if x]:
        res[r] = analyze(os.path.join(a.dump_dir, f"{r}.npz"), a.seed)
        print(f"{r}: done", flush=True)
        with open(a.out or os.path.join(a.dump_dir, "startup_check.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
