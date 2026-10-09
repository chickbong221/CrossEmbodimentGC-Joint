"""Report of the gradient-conflict analysis (morse/conflict.py) for one or more checkpoints.

Run from the repo root:
  python experiments/analysis/conflict_report.py CONFLICT.npz [CONFLICT.npz ...] --tables k8.json k16.json
         [--variant iql|bc] [--group all|enc|core|dec|std] [--null 2000] [--json OUT.json]

Metric (as in the cross-embodiment offline-RL paper behind Embodiment Grouping, and PCGrad): the cosine similarity of two
actor gradients (gradient of the IQL/AWR policy loss over the chosen parameters); two gradients conflict when it is
negative. Reported as the mean cosine and the fraction of negative cosines. Two kinds of cosine are shown:
  * batch cosine: gradients of one training-size minibatch (what one optimizer step sees), averaged over the
    minibatches; sampling noise pulls every cosine toward 0 and decides the sign of small ones.
  * persistent cosine: cosine of the mean gradients of two halves of the minibatches, products taken across the halves
    (g_A(half1).g_B(half2) + g_A(half2).g_B(half1)) / 2 over sqrt of the same-set cross products, so the sampling noise
    of either gradient drops out of the expectation of the numerator and the denominator.
Every statistic is compared with the same statistic on random groupings of the same sizes (permutation null).

Parts: A trajectories (success vs fall vs low-return episodes of one robot), B robots (all pairs, and the Embodiment
Grouping groups), C joint groups (the 8 and 16 group tables; units are (robot, joint)).
"""
import argparse
import json
import os

import numpy as np

GROUPS = ("enc", "core", "dec", "std", "all")
VARIANTS = ("iql", "bc")
# Embodiment Grouping, M = 4 (Ward clustering of data/meta/embodiment_distance.npy, as uni_iql_eg)
EG = {"a1": 0, "go1": 0, "go2": 0, "bkv0": 0, "bkvb": 0, "bg": 0, "bit": 0, "hx": 0,
      "anyb": 1, "anyc": 1, "cas": 1, "h1": 2, "tl": 2, "g1": 3, "op3": 3, "nao5": 3}
POSITIONAL = ("a1", "go1", "go2", "anyb", "anyc", "bkv0", "bkvb", "bg", "bit", "h1", "g1", "tl", "op3", "nao5", "cas", "hx")


def cosines(G):
    """Cosine matrices from Gram matrices (..., N, N)."""
    d = np.sqrt(np.maximum(np.einsum("...ii->...i", G), 1e-30))
    return G / (d[..., :, None] * d[..., None, :])


def persistent(Gc, S1=None, S2=None, min_rel=0.1, floor=1e-30):
    """Cosine from a cross-half Gram (rows: half 1, columns: half 2), symmetrized, clipped to [-1, 1]. Entries whose
    split-half reliability d_ii / sqrt(S1_ii S2_ii) is below min_rel (the mean gradient is mostly sampling noise) or whose
    cross product is not positive are nan."""
    X = 0.5 * (Gc + Gc.T)
    d = np.diag(X).copy()
    bad = d <= floor
    if S1 is not None:
        rel = d / np.sqrt(np.maximum(np.diag(S1) * np.diag(S2), floor))
        bad |= rel < min_rel
    d[bad] = np.nan
    s = np.sqrt(d)
    return np.clip(X / (s[:, None] * s[None, :]), -1.0, 1.0)


def membership(labels, n=None):
    n = int(labels.max()) + 1 if n is None else n
    M = np.zeros((len(labels), n))
    M[np.arange(len(labels)), labels] = 1.0
    return M


def pair_stats(C, mask=None):
    """Mean cosine and fraction of negative cosines over the off-diagonal pairs i < j (inside mask)."""
    iu = np.triu_indices(C.shape[-1], 1)
    v = C[..., iu[0], iu[1]]
    if mask is not None:
        v = v[..., mask[iu]]
    return float(np.nanmean(v)), float(np.nanmean(v < 0)), int(v.shape[-1])


def within_between(C, labels, valid=None):
    """Mean cosine of pairs with equal / different labels (i < j, optionally only pairs where valid[i, j])."""
    iu = np.triu_indices(len(labels), 1)
    same = labels[iu[0]] == labels[iu[1]]
    ok = np.ones(len(same), bool) if valid is None else valid[iu]
    v = C[iu]
    w, b = v[same & ok], v[~same & ok]
    return (np.nanmean(w) if len(w) else np.nan), (np.nanmean(b) if len(b) else np.nan), \
           (np.nanmean(w < 0) if len(w) else np.nan), (np.nanmean(b < 0) if len(b) else np.nan), len(w), len(b)


def null_gap(C, labels, valid, rng, n, within_robot=None):
    """Distribution of (within - between) mean cosine over random relabellings with the same group sizes. With
    within_robot (robot index per unit) the labels are shuffled inside every robot (robot composition kept)."""
    out = np.empty(n)
    for i in range(n):
        if within_robot is None:
            lab = rng.permutation(labels)
        else:
            lab = labels.copy()
            for r in np.unique(within_robot):
                idx = np.flatnonzero(within_robot == r)
                lab[idx] = labels[rng.permutation(idx)]
        w, b, *_ = within_between(C, lab, valid)
        out[i] = w - b
    return out


def pv(p, n):
    return f"< {1.0 / n:.4f}" if p == 0 else f"= {p:.3f}"


def fmt(x, nd=3):
    return "  nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}"


def load_table(path, robot_names):
    t = json.load(open(path))
    groups = t["groups"]
    names = list(groups)
    out = []
    for r, n in enumerate(robot_names):
        key = n if n in groups else (names[r] if r < len(names) else None)
        out.append(np.asarray(groups[key], dtype=int))
    return out, int(t["k"])


# ---- Part B / C: units, robots, groups ----------------------------------------------------------------------------

def unit_blocks(z, gi, vi):
    K = z["unit_gram"].shape[0]
    Gk = z["unit_gram"][:, gi, vi].astype(np.float64)                      # (K, U, U)
    Gc = z["half_cross"][gi, vi].astype(np.float64)
    return Gk, Gc, (z["half_same1"][gi, vi].astype(np.float64), z["half_same2"][gi, vi].astype(np.float64))


def group_cos(Gk, Gc, S, M):
    """Batch and persistent cosines between the summed gradients of the groups given by membership matrix M (U, n)."""
    Gg = np.einsum("ua,kuv,vb->kab", M, Gk, M)
    Cb = cosines(Gg)                                                     # (K, n, n)
    Cp = persistent(M.T @ Gc @ M, M.T @ S[0] @ M, M.T @ S[1] @ M)
    return Cb, Cp


def report_units(z, gi, vi, tables, n_null, rng, out):
    names = [str(n) for n in z["robot_names"]]
    robot = z["unit_robot"].astype(int)
    Gk, Gc, S = unit_blocks(z, gi, vi)
    R = len(names)
    Mr = membership(robot, R)
    print("\n### B. Robots (gradient of each robot's policy loss)")
    Cb, Cp = group_cos(Gk, Gc, S, Mr)
    mb, fb, npairs = pair_stats(Cb)
    mp, fp, _ = pair_stats(Cp)
    print(f"all {npairs} robot pairs: batch cosine mean {fmt(mb)}, negative in {100 * fb:.1f}% of (pair, minibatch); "
          f"persistent cosine mean {fmt(mp)}, negative in {100 * fp:.1f}% of pairs")
    out["robots"] = dict(batch_mean=mb, batch_frac_neg=fb, persistent_mean=mp, persistent_frac_neg=fp)
    Cbm = Cb.mean(axis=0)
    eg = np.array([EG.get(n, EG[POSITIONAL[i]] if i < len(POSITIONAL) else 0) for i, n in enumerate(names)])
    print("\nEmbodiment Grouping (M=4) groups: " + "; ".join(
        f"g{g + 1}=" + ",".join(n for n, l in zip(names, eg) if l == g) for g in range(4)))
    for tag, C in (("batch", Cbm), ("persistent", Cp)):
        w, b, wn, bn, nw, nb = within_between(C, eg)
        gap_null = null_gap(C, eg, None, rng, n_null)
        gap = w - b
        p = float((gap_null >= gap).mean())
        print(f"  {tag:10s} robot pairs in the same EG group ({nw}): mean cos {fmt(w)}, negative {100 * wn:4.1f}%;  "
              f"in different groups ({nb}): {fmt(b)}, negative {100 * bn:4.1f}%;  gap {fmt(gap)}  "
              f"random partitions of the same sizes: gap {fmt(gap_null.mean())} +- {gap_null.std():.3f}, p(random >= EG) {pv(p, n_null)}")
        out.setdefault("eg", {})[tag] = dict(within=float(w), between=float(b), within_neg=float(wn), between_neg=float(bn),
                                              gap=float(gap), null_mean=float(gap_null.mean()), null_sd=float(gap_null.std()), p=p)
    # EG group gradients (the sum of the member robots' gradients: what a group update follows)
    Cbg, Cpg = group_cos(Gk, Gc, S, membership(eg[robot], 4))
    print("  cosine between the 4 EG group gradients (batch mean / persistent):")
    for a in range(4):
        print("    g%d " % (a + 1) + "  ".join(f"{Cbg.mean(axis=0)[a, b]:+.2f}/{Cpg[a, b]:+.2f}" for b in range(4)))
    print("  robot x robot batch cosine (mean over minibatches):")
    print("        " + " ".join(f"{n:>5s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:5s} " + " ".join(f"{Cbm[i, j]:+5.2f}" for j in range(R)))
    out["robot_cos_batch"] = Cbm.tolist()
    out["robot_cos_persistent"] = np.where(np.isnan(Cp), 0, Cp).tolist()

    print("\n### C. Joint groups (units are (robot, joint); a group gradient is the sum of its units' gradients)")
    joint = z["unit_joint"].astype(int)
    cross = robot[:, None] != robot[None, :]
    Cu_b = cosines(Gk).mean(axis=0)
    Cu_p = persistent(Gc, S[0], S[1])
    for path in tables:
        labs, k = load_table(path, names)
        lab = np.array([labs[r][j] if j < len(labs[r]) else -1 for r, j in zip(robot, joint)])
        if (lab < 0).any():
            print(f"  {path}: {int((lab < 0).sum())} units without a group, skipped")
            continue
        sizes = np.bincount(lab, minlength=k)
        Cb2, Cp2 = group_cos(Gk, Gc, S, membership(lab, k))
        mb, fb, npairs = pair_stats(Cb2)
        mp, fp, _ = pair_stats(Cp2)
        print(f"\n  k={k} ({os.path.basename(path)}): group sizes {sizes.tolist()}")
        print(f"    {npairs} group pairs: batch cosine mean {fmt(mb)}, negative in {100 * fb:.1f}% of (pair, minibatch); "
              f"persistent mean {fmt(mp)}, negative in {100 * fp:.1f}% of pairs")
        res = dict(k=k, group_batch_mean=mb, group_batch_frac_neg=fb, group_persistent_mean=mp, group_persistent_frac_neg=fp)
        for tag, C in (("batch", Cu_b), ("persistent", Cu_p)):
            w, b, wn, bn, nw, nb = within_between(C, lab, cross)
            g_all = null_gap(C, lab, cross, rng, n_null)
            g_rob = null_gap(C, lab, cross, rng, n_null, within_robot=robot)
            gap = w - b
            print(f"    units of different robots, {tag:10s}: same group ({nw}) mean cos {fmt(w)}, negative {100 * wn:4.1f}%; "
                  f"different group ({nb}) {fmt(b)}, negative {100 * bn:4.1f}%; gap {fmt(gap)}")
            print(f"        null, labels shuffled over all units: gap {fmt(g_all.mean())} +- {g_all.std():.3f} (p {pv(float((g_all >= gap).mean()), n_null)});"
                  f"  null, labels shuffled within each robot: gap {fmt(g_rob.mean())} +- {g_rob.std():.3f} (p {pv(float((g_rob >= gap).mean()), n_null)})")
            res[tag] = dict(within=float(w), between=float(b), gap=float(gap), null_all=float(g_all.mean()), null_all_sd=float(g_all.std()),
                            p_all=float((g_all >= gap).mean()), null_robot=float(g_rob.mean()), null_robot_sd=float(g_rob.std()),
                            p_robot=float((g_rob >= gap).mean()))
        if k <= 8:
            print("    group x group cosine (batch mean / persistent):")
            for a in range(k):
                print(f"      j{a:<2d} " + " ".join(f"{Cb2.mean(axis=0)[a, b]:+.2f}/{Cp2[a, b]:+.2f}" for b in range(k)))
        else:
            print("    group x group persistent cosine:")
            print("      " + " ".join(f"{b:>5d}" for b in range(k)))
            for a in range(k):
                print(f"      j{a:<2d} " + " ".join(f"{Cp2[a, b]:+5.2f}" for b in range(k)))
        out.setdefault("joint_groups", {})[f"k{k}"] = res


# ---- Part A: trajectories -------------------------------------------------------------------------------------------

def report_trajectories(z, gi, vi, out):
    names = [str(n) for n in z["robot_names"]]
    classes = [str(c) for c in z["classes"]]
    print("\n### A. Trajectories: gradient of one episode (mean loss over its transitions) against another")
    print("cosine between episode gradients of the same robot. S = success, F = fall (ended by termination), L = low return (full length).")
    print("pair cosines: mean (fraction negative); persistent = cosine of class-mean gradients from cross-half products, clipped to [-1, 1], n/r (not reliable) when a class's split-half reliability is below 0.1; rel = that reliability")
    head = f"{'robot':6s} {'n S/F/L':>10s} | {'S-S':>14s} {'F-F':>14s} {'S-F':>14s} {'S-L':>14s} | persistent S-F  S-L | rel S   F   L"
    print(head)
    rows = {}
    for name in names:
        key = f"ep_gram_{name}"
        if key not in z.files:
            continue
        G = z[key][gi, vi].astype(np.float64)
        lab = z[f"ep_class_{name}"].astype(int)
        C = cosines(G)
        idx = {c: np.flatnonzero(lab == c) for c in range(3)}
        cnt = "/".join(str(len(idx[c])) for c in range(3))

        def pairs(a, b):
            if len(idx[a]) == 0 or len(idx[b]) == 0:
                return None
            sub = C[np.ix_(idx[a], idx[b])]
            if a == b:
                sub = sub[~np.eye(len(idx[a]), dtype=bool)]
            return float(sub.mean()), float((sub < 0).mean())

        def half_dot(a, b):
            """Cross-half products of class means: (a1.b2 + a2.b1)/2, halves = first/second half of each class."""
            ia, ib = idx[a], idx[b]
            ha, hb = len(ia) // 2, len(ib) // 2
            a1, a2, b1, b2 = ia[:ha], ia[ha:], ib[:hb], ib[hb:]
            f = lambda x, y: G[np.ix_(x, y)].mean()
            return 0.5 * (f(a1, b2) + f(a2, b1))

        rel = {}
        for c in range(3):
            if len(idx[c]) >= 4:
                ic = idx[c]
                h = len(ic) // 2
                x = G[np.ix_(ic[:h], ic[h:])].mean()
                rel[c] = x / np.sqrt(max(G[np.ix_(ic[:h], ic[:h])].mean(), 1e-30) * max(G[np.ix_(ic[h:], ic[h:])].mean(), 1e-30))
        pers = {}
        for a, b in ((0, 1), (0, 2)):
            if a in rel and b in rel and rel[a] >= 0.1 and rel[b] >= 0.1:
                da, db = half_dot(a, a), half_dot(b, b)
                pers[(a, b)] = float(np.clip(half_dot(a, b) / np.sqrt(da * db), -1, 1)) if da > 0 and db > 0 else np.nan
        s = lambda t: "      -      " if t is None else f"{t[0]:+.3f} ({100 * t[1]:3.0f}%)"
        p = lambda k: "  n/r " if k not in pers else f"{pers[k]:+.3f}"
        r = lambda c: "  -  " if c not in rel else f"{rel[c]:+.2f}"
        print(f"{name:6s} {cnt:>10s} | {s(pairs(0, 0)):>14s} {s(pairs(1, 1)):>14s} {s(pairs(0, 1)):>14s} {s(pairs(0, 2)):>14s} | {p((0, 1)):>14s} {p((0, 2)):>6s} | {r(0)} {r(1)} {r(2)}")
        rows[name] = dict(n=[len(idx[c]) for c in range(3)], SS=pairs(0, 0), FF=pairs(1, 1), SF=pairs(0, 1), SL=pairs(0, 2),
                          pers_SF=pers.get((0, 1)), pers_SL=pers.get((0, 2)), rel=rel)
    out["trajectories"] = rows
    sf = [v["SF"][0] for v in rows.values() if v["SF"]]
    ps = [v["pers_SF"] for v in rows.values() if v["pers_SF"] is not None and not np.isnan(v["pers_SF"])]
    if sf:
        print(f"mean over the {len(sf)} robots with success and fall episodes: pair cosine S-F {np.mean(sf):+.3f}"
              + (f", persistent S-F {np.mean(ps):+.3f}" if ps else ""))


def summary_by_group(z, tables, vi, rng, n_null):
    """One table: the headline statistics of A, B and C for every parameter group (the 'all' number is dominated by the
    few parameters with the largest gradients, so the groups must be read next to it)."""
    names = [str(n) for n in z["robot_names"]]
    robot = z["unit_robot"].astype(int)
    joint = z["unit_joint"].astype(int)
    eg = np.array([EG.get(n, EG[POSITIONAL[i]] if i < len(POSITIONAL) else 0) for i, n in enumerate(names)])
    labs = []
    for path in tables:
        tab, k = load_table(path, names)
        lab = np.array([tab[r][j] if j < len(tab[r]) else -1 for r, j in zip(robot, joint)])
        labs.append((k, lab))
    cross = robot[:, None] != robot[None, :]
    print("\n### Summary by parameter group (loss variant: %s)" % VARIANTS[vi])
    print("A: mean over robots of the cosine between episode gradients, success-success / fall-fall / success-fall, and the share of success-fall pairs that are negative")
    print("B: share of negative robot-pair cosines (batch / persistent); EG gap = same-group minus different-group mean persistent cosine (p vs random partitions)")
    print("C: same-group minus different-group mean persistent cosine of joint units of different robots (p vs labels shuffled within each robot)")
    head = f"{'params':6s} {'size':>7s} | {'S-S':>6s} {'F-F':>6s} {'S-F':>6s} {'S-F<0':>6s} | {'B neg%':>6s} {'pers%':>6s} {'EG gap':>7s} {'p':>7s}"
    for k, _ in labs:
        head += f" | k={k} gap  p"
    print(head)
    for gi, gn in enumerate(GROUPS):
        ss, ff, sf, neg = [], [], [], []
        for n in names:
            key = f"ep_gram_{n}"
            if key not in z.files:
                continue
            lab = z[f"ep_class_{n}"].astype(int)
            C = cosines(z[key][gi, vi].astype(np.float64))
            i0, i1 = np.flatnonzero(lab == 0), np.flatnonzero(lab == 1)
            if len(i0) < 4 or len(i1) < 4:
                continue
            off = lambda a: a[~np.eye(len(a), dtype=bool)]
            ss.append(off(C[np.ix_(i0, i0)]).mean())
            ff.append(off(C[np.ix_(i1, i1)]).mean())
            x = C[np.ix_(i0, i1)]
            sf.append(x.mean())
            neg.append((x < 0).mean())
        Gk, Gc, S = unit_blocks(z, gi, vi)
        Cb, Cp = group_cos(Gk, Gc, S, membership(robot, len(names)))
        _, fb, _ = pair_stats(Cb)
        _, fp, _ = pair_stats(Cp)
        w, b, *_ = within_between(Cp, eg)
        gap = w - b
        pe = float((null_gap(Cp, eg, None, rng, n_null) >= gap).mean())
        row = f"{GROUPS[gi]:6s} {int(z['group_sizes'][gi]):7d} | {np.mean(ss):+6.2f} {np.mean(ff):+6.2f} {np.mean(sf):+6.2f} {100 * np.mean(neg):5.0f}% | {100 * fb:5.1f}% {100 * fp:5.1f}% {gap:+7.3f} {pv(pe, n_null):>7s}"
        Cu = persistent(z["half_cross"][gi, vi].astype(np.float64), S[0], S[1])
        for k, lab in labs:
            if (lab < 0).any():
                row += " |   n/a"
                continue
            w, b, *_ = within_between(Cu, lab, cross)
            pj = float((null_gap(Cu, lab, cross, rng, n_null, within_robot=robot) >= (w - b)).mean())
            row += f" | {w - b:+.3f} {pv(pj, n_null):>7s}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tables", nargs="*", default=[])
    ap.add_argument("--variant", default="iql", choices=VARIANTS)
    ap.add_argument("--group", default="all", choices=GROUPS)
    ap.add_argument("--null", type=int, default=2000)
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    result = {}
    for path in args.files:
        z = np.load(path, allow_pickle=True)
        gi, vi = GROUPS.index(args.group), VARIANTS.index(args.variant)
        ck = str(z["checkpoint"])
        print("=" * 110)
        print(f"{os.path.basename(path)}   checkpoint {ck}\nloss variant: {args.variant}; parameters: {args.group} "
              f"({int(z['group_sizes'][gi])}); minibatches {int(z['n_batches'])} x robot batch {int(z['batch_sizes'][0])}")
        out = {}
        report_trajectories(z, gi, vi, out)
        report_units(z, gi, vi, args.tables, args.null, rng, out)
        summary_by_group(z, args.tables, vi, rng, min(args.null, 500))
        result[os.path.basename(path)] = out
    if args.json:
        json.dump(result, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
