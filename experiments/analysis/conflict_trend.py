"""Headline gradient-conflict statistics of several checkpoints in one table (local, from the conflict .npz files).

Run from the repo root:
  python experiments/analysis/conflict_trend.py DATA_DIR [--tables k8.json k16.json] [--variant iql] [--null 300]
DATA_DIR holds U2-URMA-s<seed>/s<seed>_conflict_at_<updates>.npz (as written by morse/conflict.py).

Columns per parameter group (core = shared body representation, dec = action decoder, std = log-std head, all):
  S-F    mean cosine between success-episode and fall-episode gradients of the same robot (mean over robots)
  rel    number of robots (of 16) whose mean gradient is reliable (split-half reliability >= 0.1); the persistent
         cosines below only exist for those robots. The EG gap is printed n/r when fewer than 10 within-group
         pairs are left (a gap from 1-3 pairs is not a result).
  neg    share of robot pairs whose persistent (noise-corrected) cosine is negative (of the valid pairs)
  EG     same-group minus different-group persistent robot cosine for the Embodiment Grouping groups, with p
  k8/k16 same-group minus different-group persistent cosine of joint units of different robots, with p
         (null: labels shuffled within each robot)
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conflict_report import (EG, GROUPS, POSITIONAL, VARIANTS, cosines, group_cos, load_table, membership,  # noqa: E402
                             null_gap, pair_stats, persistent, unit_blocks, within_between)


def stats(z, gi, vi, tables, rng, n_null):
    names = [str(n) for n in z["robot_names"]]
    out = {}
    sf = []
    for n in names:
        key = f"ep_gram_{n}"
        if key not in z.files:
            continue
        lab = z[f"ep_class_{n}"].astype(int)
        i0, i1 = np.flatnonzero(lab == 0), np.flatnonzero(lab == 1)
        if len(i0) >= 4 and len(i1) >= 4:
            sf.append(cosines(z[key][gi, vi].astype(np.float64))[np.ix_(i0, i1)].mean())
    out["S-F"] = float(np.mean(sf)) if sf else np.nan
    robot = z["unit_robot"].astype(int)
    joint = z["unit_joint"].astype(int)
    Gk, Gc, S = unit_blocks(z, gi, vi)
    _, Cp = group_cos(Gk, Gc, S, membership(robot, len(names)))
    out["neg"] = pair_stats(Cp)[1]
    M = membership(robot, len(names))
    Sc = M.T @ z["half_cross"][gi, vi].astype(np.float64) @ M
    d = np.diag(0.5 * (Sc + Sc.T))
    relr = d / np.sqrt(np.maximum(np.diag(M.T @ S[0] @ M) * np.diag(M.T @ S[1] @ M), 1e-30))
    out["rel"] = int((relr >= 0.1).sum())
    eg = np.array([EG.get(n, EG[POSITIONAL[i]] if i < len(POSITIONAL) else 0) for i, n in enumerate(names)])
    iu = np.triu_indices(len(names), 1)
    n_within = int(np.isfinite(Cp[iu])[eg[iu[0]] == eg[iu[1]]].sum())
    w, b, *_ = within_between(Cp, eg)
    out["EG"] = (w - b, float((null_gap(Cp, eg, None, rng, n_null) >= w - b).mean())) if n_within >= 10 else None
    cross = robot[:, None] != robot[None, :]
    Cu = persistent(z["half_cross"][gi, vi].astype(np.float64), S[0], S[1])
    out["units"] = float(np.isfinite(np.diag(Cu)).mean())
    for path in tables:
        tab, k = load_table(path, names)
        lab = np.array([tab[r][j] if j < len(tab[r]) else -1 for r, j in zip(robot, joint)])
        w, b, *_ = within_between(Cu, lab, cross)
        out[f"k{k}"] = (w - b, float((null_gap(Cu, lab, cross, rng, n_null, within_robot=robot) >= w - b).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--tables", nargs="*", default=[])
    ap.add_argument("--variant", default="iql", choices=VARIANTS)
    ap.add_argument("--groups", default="core,dec,std,all")
    ap.add_argument("--null", type=int, default=300)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    files = []
    for f in glob.glob(os.path.join(args.data, "U2-URMA-s*", "s*_conflict_at_*.npz")):
        m = re.search(r"s(\d+)_conflict_at_(\d+)\.npz$", f)
        files.append((int(m.group(1)), int(m.group(2)), f))
    files.sort(key=lambda t: (t[1], t[0]))
    vi = VARIANTS.index(args.variant)
    groups = [g.strip() for g in args.groups.split(",")]
    pv = lambda p: f"<{1 / args.null:.3f}" if p == 0 else f"{p:.3f}"
    for g in groups:
        gi = GROUPS.index(g)
        print(f"\n== parameters: {g}  (loss variant {args.variant})")
        head = f"{'updates':>8s} {'seed':>4s} | {'S-F':>6s} | {'rel':>3s} {'neg':>5s} | {'EG gap':>7s} {'p':>6s} | {'units':>5s}"
        for path in args.tables:
            head += f" | {os.path.basename(path)[:16]:>16s} {'p':>6s}"
        print(head)
        for seed, upd, f in files:
            z = np.load(f, allow_pickle=True)
            s = stats(z, gi, vi, args.tables, rng, args.null)
            eg = f"{s['EG'][0]:+7.3f} {pv(s['EG'][1]):>6s}" if s["EG"] else f"{'n/r':>7s} {'':>6s}"
            row = f"{upd:8d} {seed:4d} | {s['S-F']:+6.2f} | {s['rel']:3d} {100 * s['neg']:4.0f}% | {eg} | {100 * s['units']:4.0f}%"
            for path in args.tables:
                k = int(__import__('json').load(open(path))["k"])
                gap, p = s[f"k{k}"]
                row += f" | {gap:+16.3f} {pv(p):>6s}"
            print(row)


if __name__ == "__main__":
    main()
