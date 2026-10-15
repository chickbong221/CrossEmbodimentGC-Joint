"""Does a grouping reduce gradient conflict? Plain update vs Embodiment Grouping (EG) vs joint groups, on one footing.

Run from the repo root:
  python experiments/analysis/conflict_grouping.py DATA_DIR --tables k8.json k16.json [--groups core,dec,all] [--null 200]
DATA_DIR holds U2-URMA-s<seed>/s<seed>_conflict_at_<updates>.npz (as written by morse/conflict.py). All cosines are
persistent (noise-corrected, conflict_report.persistent) unless marked "batch"; conflict = negative cosine.

Part 1, robots (what EG changes). A plain update steps along the sum of all 16 robot gradients. EG (uni_iql_eg) takes
one optimizer step per group, along the mean of the group's robot gradients, the 4 groups in random order on the same
parameters, so one EG round moves every robot by all 4 group steps.
  pairs all / in / across   share of negative robot-pair cosines (all 120 pairs; the 35 pairs inside an EG group,
                            averaged together in one EG step; the 85 pairs across groups, put in separate steps)
  robot vs step             share of robots whose gradient has a negative cosine with the step that moves them:
                            plain = sum of all robots; EG own = their group's mean; EG round = sum of the 4 group means
  group vs group            share of negative cosines between the 4 EG group means (6 pairs): EG steps undoing each other
Part 2, joint units (EG groups vs joint groups at the same level). Units are (robot, joint); only pairs of units of
different robots. For each grouping: mean cosine and share of negative cosines of the pairs inside a group, and the
same for random groupings of the same sizes (joint tables: labels shuffled inside every robot; EG: robots shuffled
across groups); and the share of units whose gradient has a negative cosine with the summed gradient of the other units
of their group (leave-self-out, which is what a group-owned expert would be trained with).
Part 3, within a robot. Success-episode vs fall-episode gradients of the same robot: episode-level mean cosine and the
cosine of the two class means (split-half, noise-corrected), mean over robots. Neither grouping can split these: the EG
group and the joint group of a sample depend on the robot and joint only, not on the episode it comes from.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conflict_report import EG, GROUPS, POSITIONAL, VARIANTS, cosines, load_table, membership, persistent  # noqa: E402


def robot_part(z, gi, vi, eg):
    R = len(eg)
    robot = z["unit_robot"].astype(int)
    Mr = membership(robot, R)
    ng = int(eg.max()) + 1
    A = np.zeros((R, ng))
    A[np.arange(R), eg] = 1.0
    A /= A.sum(0, keepdims=True)                                     # group mean of robot gradients
    M = np.concatenate([Mr, Mr @ A, Mr.sum(1, keepdims=True), (Mr @ A).sum(1, keepdims=True)], axis=1)
    Gc = z["half_cross"][gi, vi].astype(np.float64)
    S1, S2 = z["half_same1"][gi, vi].astype(np.float64), z["half_same2"][gi, vi].astype(np.float64)
    C = persistent(M.T @ Gc @ M, M.T @ S1 @ M, M.T @ S2 @ M)
    iu = np.triu_indices(R, 1)
    same = eg[iu[0]] == eg[iu[1]]
    v = C[:R, :R][iu]
    neg = lambda x: float(np.nanmean(x[np.isfinite(x)] < 0)) if np.isfinite(x).any() else np.nan
    Gk = z["unit_gram"][:, gi, vi].astype(np.float64)
    Cb = cosines(np.einsum("ua,kuv,vb->kab", Mr, Gk, Mr))[:, iu[0], iu[1]]       # (K, pairs), one minibatch each
    gg = C[R:R + ng, R:R + ng][np.triu_indices(ng, 1)]
    # the same three steps without the robot's own gradient (only the pull of the other robots)
    Xr = Mr.T @ Gc @ Mr
    Xr = 0.5 * (Xr + Xr.T)
    okr = np.isfinite(np.diag(C[:R, :R]))
    lo_plain, lo_own, lo_round = [], [], []
    for r in np.flatnonzero(okr):
        for acc, w in ((lo_plain, okr * 1.0), (lo_own, okr * (eg == eg[r])), (lo_round, okr / np.bincount(eg)[eg])):
            w = w.copy()
            w[r] = 0.0
            den = w @ Xr @ w
            if w.sum() > 0 and den > 0:
                acc.append((Xr[r] @ w) / np.sqrt(Xr[r, r] * den))
    nn = lambda a: float(np.mean(np.array(a) < 0)) if len(a) else np.nan
    return dict(lo_plain=nn(lo_plain), lo_own=nn(lo_own), lo_round=nn(lo_round),
        pair_all=neg(v), pair_in=neg(v[same]), pair_out=neg(v[~same]),
        cos_all=np.nanmean(v), cos_in=np.nanmean(v[same]), cos_out=np.nanmean(v[~same]),
        bpair_all=float((Cb < 0).mean()), bpair_in=float((Cb[:, same] < 0).mean()), bpair_out=float((Cb[:, ~same] < 0).mean()),
        step_plain=neg(C[np.arange(R), R + ng]), step_own=neg(C[np.arange(R), R + eg]), step_round=neg(C[np.arange(R), R + ng + 1]),
        cstep_plain=np.nanmean(C[np.arange(R), R + ng]), cstep_own=np.nanmean(C[np.arange(R), R + eg]),
        cstep_round=np.nanmean(C[np.arange(R), R + ng + 1]),
        gg_neg=neg(gg), gg_cos=np.nanmean(gg), gg_min=np.nanmin(gg) if np.isfinite(gg).any() else np.nan,
        n_rel=int(np.isfinite(np.diag(C[:R, :R])).sum()))


def loo(X, lab, ok, robot=None):
    """Cosine of every unit's gradient with the summed gradient of the other (valid) units carrying the same label;
    with robot given, only the units of the other robots."""
    out = np.full(len(lab), np.nan)
    for g in np.unique(lab):
        for r in ([None] if robot is None else np.unique(robot)):
            m = ((lab == g) & ok).astype(float)
            own = m.copy() if r is None else m * (robot == r)
            if r is not None:
                m = m * (robot != r)
            Xm = X @ m
            tot = m @ Xm
            for u in np.flatnonzero(own > 0):
                if r is None:
                    num, den2 = Xm[u] - X[u, u], tot - 2 * Xm[u] + X[u, u]
                else:
                    num, den2 = Xm[u], tot
                if den2 > 0 and X[u, u] > 0:
                    out[u] = num / np.sqrt(X[u, u] * den2)
    return out


def unit_part(z, gi, vi, eg, labels, rng, n_null):
    robot = z["unit_robot"].astype(int)
    Gc = z["half_cross"][gi, vi].astype(np.float64)
    S1, S2 = z["half_same1"][gi, vi].astype(np.float64), z["half_same2"][gi, vi].astype(np.float64)
    Cu = persistent(Gc, S1, S2)
    ok = np.isfinite(np.diag(Cu))
    X = 0.5 * (Gc + Gc.T)
    iu = np.triu_indices(len(robot), 1)
    cross = (robot[iu[0]] != robot[iu[1]]) & ok[iu[0]] & ok[iu[1]]
    v = Cu[iu]

    def pairs(lab):
        s = cross & (lab[iu[0]] == lab[iu[1]])
        return float(np.mean(v[s])), float(np.mean(v[s] < 0))

    def loo_stats(lab, other_robots=False):
        c = loo(X, lab, ok, robot if other_robots else None)[ok]
        c = c[np.isfinite(c)]
        return float(np.mean(c < 0)), float(np.mean(c))

    out = {"units": float(ok.mean()), "plain": (float(np.mean(v[cross])), float(np.mean(v[cross] < 0))),
           "plain_loo": loo_stats(np.zeros(len(robot), int)), "plain_xr": loo_stats(np.zeros(len(robot), int), True)}
    R = len(eg)
    for name, lab in labels.items():
        nulls, nloo, nxr = [], [], []
        for _ in range(n_null):
            if name == "EG":
                nl = rng.permutation(eg)[robot]
            else:
                nl = lab.copy()
                for r in range(R):
                    idx = np.flatnonzero(robot == r)
                    nl[idx] = lab[rng.permutation(idx)]
            nulls.append(pairs(nl))
            if len(nloo) < 20:
                nloo.append(loo_stats(nl))
                nxr.append(loo_stats(nl, True))
        out[name] = pairs(lab)
        out[name + "_null"] = tuple(np.mean(nulls, axis=0))
        out[name + "_loo"] = loo_stats(lab)
        out[name + "_loo_null"] = tuple(np.mean(nloo, axis=0))
        out[name + "_xr"] = loo_stats(lab, True)
        out[name + "_xr_null"] = tuple(np.mean(nxr, axis=0))
    return out


def within_robot(z, gi, vi):
    names = [str(n) for n in z["robot_names"]]
    ep, cm = [], []
    for n in names:
        key = f"ep_gram_{n}"
        if key not in z.files:
            continue
        G = z[key][gi, vi].astype(np.float64)
        lab = z[f"ep_class_{n}"].astype(int)
        s, f = np.flatnonzero(lab == 0), np.flatnonzero(lab == 1)
        if len(s) < 4 or len(f) < 4:
            continue
        ep.append(cosines(G)[np.ix_(s, f)].mean())
        s1, s2 = s[: len(s) // 2], s[len(s) // 2:]                 # episodes are stored in random order
        f1, f2 = f[: len(f) // 2], f[len(f) // 2:]
        m = lambda a, b: G[np.ix_(a, b)].mean()
        ss, ff = m(s1, s2), m(f1, f2)
        if ss > 0 and ff > 0:
            cm.append(np.clip(0.5 * (m(s1, f2) + m(s2, f1)) / np.sqrt(ss * ff), -1, 1))
    return float(np.mean(ep)), (float(np.mean(cm)) if cm else np.nan), len(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--tables", nargs="*", default=[])
    ap.add_argument("--variant", default="iql", choices=VARIANTS)
    ap.add_argument("--groups", default="core,dec,all")
    ap.add_argument("--null", type=int, default=200)
    args = ap.parse_args()
    vi = VARIANTS.index(args.variant)
    files = []
    for f in glob.glob(os.path.join(args.data, "U2-URMA-s*", "s*_conflict_at_*.npz")):
        m = re.search(r"s(\d+)_conflict_at_(\d+)\.npz$", f)
        files.append((int(m.group(2)), int(m.group(1)), f))
    files.sort()
    rng = np.random.default_rng(0)
    pct = lambda x: "  -" if not np.isfinite(x) else f"{100 * x:3.0f}"
    for g in [x.strip() for x in args.groups.split(",")]:
        gi = GROUPS.index(g)
        rows = []
        for upd, seed, f in files:
            z = np.load(f, allow_pickle=True)
            names = [str(n) for n in z["robot_names"]]
            eg = np.array([EG.get(n, EG[POSITIONAL[i]]) for i, n in enumerate(names)])
            robot, joint = z["unit_robot"].astype(int), z["unit_joint"].astype(int)
            labels = {"EG": eg[robot]}
            for path in args.tables:
                tab, k = load_table(path, names)
                labels[f"k{k}"] = np.array([tab[r][j] for r, j in zip(robot, joint)])
            rows.append((upd, seed, robot_part(z, gi, vi, eg), unit_part(z, gi, vi, eg, labels, rng, args.null),
                         within_robot(z, gi, vi)))
        print(f"\n######## parameters: {g}  (loss variant {args.variant})")
        print("\nPart 1, robots. % negative (persistent; 'batch' = one minibatch). rel = robots with a reliable gradient")
        print(f"{'upd':>6s} {'s':>1s} {'rel':>3s} | pairs all  in out | batch all  in out | mean cos all   in  out |"
              f" robot vs step: plain EGown EGround | same w/o own robot | grp-grp neg  mean   min")
        for upd, seed, r, _, _ in rows:
            print(f"{upd:6d} {seed:1d} {r['n_rel']:3d} |      {pct(r['pair_all'])} {pct(r['pair_in'])} {pct(r['pair_out'])} |"
                  f"      {pct(r['bpair_all'])} {pct(r['bpair_in'])} {pct(r['bpair_out'])} |"
                  f"    {r['cos_all']:+.2f} {r['cos_in']:+.2f} {r['cos_out']:+.2f} |"
                  f"                 {pct(r['step_plain'])}   {pct(r['step_own'])}     {pct(r['step_round'])} |"
                  f"       {pct(r['lo_plain'])}  {pct(r['lo_own'])}  {pct(r['lo_round'])} |"
                  f"         {pct(r['gg_neg'])} {r['gg_cos']:+.2f} {r['gg_min']:+.2f}")
        keys = [k for k in rows[0][3] if k in ("EG",) or re.fullmatch(r"k\d+", k)]
        print("\nPart 2a, unit pairs of different robots inside a group: mean cosine / % negative  [random grouping, same sizes]")
        head = f"{'upd':>6s} {'s':>1s} {'units':>5s} | {'no grouping':>13s}"
        for k in keys:
            head += f" | {k + ' in group':>14s} {'[random]':>14s}"
        print(head)
        for upd, seed, _, u, _ in rows:
            line = f"{upd:6d} {seed:1d} {100 * u['units']:4.0f}% | {u['plain'][0]:+.3f} / {pct(u['plain'][1])}%"
            for k in keys:
                line += f" | {u[k][0]:+.3f} / {pct(u[k][1])}%  [{u[k + '_null'][0]:+.3f} / {pct(u[k + '_null'][1])}%]"
            print(line)
        print("\nPart 2b, % of units whose gradient opposes the summed gradient of the rest of their group (mean cosine)  [random]")
        head = f"{'upd':>6s} {'s':>1s} | {'no grouping':>14s}"
        for k in keys:
            head += f" | {k:>14s} {'[random]':>14s}"
        print(head)
        for upd, seed, _, u, _ in rows:
            line = f"{upd:6d} {seed:1d} | {pct(u['plain_loo'][0])}% ({u['plain_loo'][1]:+.3f})"
            for k in keys:
                line += f" | {pct(u[k + '_loo'][0])}% ({u[k + '_loo'][1]:+.3f})  [{pct(u[k + '_loo_null'][0])}% ({u[k + '_loo_null'][1]:+.3f})]"
            print(line)
        print("\nPart 2c, as 2b but the rest of the group without the unit's own robot (cross-robot pull only)  [random]")
        print(head)
        for upd, seed, _, u, _ in rows:
            line = f"{upd:6d} {seed:1d} | {pct(u['plain_xr'][0])}% ({u['plain_xr'][1]:+.3f})"
            for k in keys:
                line += f" | {pct(u[k + '_xr'][0])}% ({u[k + '_xr'][1]:+.3f})  [{pct(u[k + '_xr_null'][0])}% ({u[k + '_xr_null'][1]:+.3f})]"
            print(line)
        print("\nPart 3, within a robot: success vs fall episodes (mean over robots)")
        print(f"{'upd':>6s} {'s':>1s} | episode cos | class-mean cos (robots)")
        for upd, seed, _, _, w in rows:
            print(f"{upd:6d} {seed:1d} |      {w[0]:+.2f}  |        {w[1]:+.2f} ({w[2]})")


if __name__ == "__main__":
    main()
