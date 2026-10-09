"""Per-robot view of the gradient-conflict analysis over checkpoints and seeds (local, from the conflict .npz files).

Run from the repo root:
  python experiments/analysis/conflict_robots.py DATA_DIR [--variant iql] [--groups core,dec,std]
DATA_DIR holds U2-URMA-s<seed>/s<seed>_conflict_at_<updates>.npz (as written by morse/conflict.py).

Tables (one per parameter group, cell = seed 0 / seed 1, columns = checkpoints):
  S-F    mean cosine between the gradients of a robot's success episodes and its fall episodes (episode level)
  S-L    the same for success vs low-return episodes that run the full length (the only bad class of the hexapod)
  S-S    mean cosine between two success episodes of the robot (agreement inside the good data)
  F-F    mean cosine between two fall episodes of the robot (agreement inside the bad data)
  align  mean persistent (noise-corrected) cosine between the robot's gradient and the gradients of the other 15 robots
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conflict_report import GROUPS, VARIANTS, cosines, group_cos, membership, unit_blocks  # noqa: E402


def per_robot(z, gi, vi):
    names = [str(n) for n in z["robot_names"]]
    sf, ss, ff, sl = {}, {}, {}, {}
    for n in names:
        key = f"ep_gram_{n}"
        if key not in z.files:
            continue
        lab = z[f"ep_class_{n}"].astype(int)
        C = cosines(z[key][gi, vi].astype(np.float64))
        i0, i1, i2 = np.flatnonzero(lab == 0), np.flatnonzero(lab == 1), np.flatnonzero(lab == 2)
        off = lambda a: a[~np.eye(len(a), dtype=bool)]
        if len(i0) >= 4:
            ss[n] = off(C[np.ix_(i0, i0)]).mean()
        if len(i1) >= 4:
            ff[n] = off(C[np.ix_(i1, i1)]).mean()
        if len(i0) >= 4 and len(i1) >= 4:
            sf[n] = C[np.ix_(i0, i1)].mean()
        if len(i0) >= 4 and len(i2) >= 4:
            sl[n] = C[np.ix_(i0, i2)].mean()
    robot = z["unit_robot"].astype(int)
    Gk, Gc, S = unit_blocks(z, gi, vi)
    _, Cp = group_cos(Gk, Gc, S, membership(robot, len(names)))
    C = Cp.copy()
    np.fill_diagonal(C, np.nan)
    align = {n: (np.nanmean(C[i]) if np.isfinite(C[i]).any() else np.nan) for i, n in enumerate(names)}
    return names, dict(SF=sf, SL=sl, SS=ss, FF=ff, align=align)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--variant", default="iql", choices=VARIANTS)
    ap.add_argument("--groups", default="core,dec,std")
    args = ap.parse_args()
    vi = VARIANTS.index(args.variant)
    files = {}
    for f in glob.glob(os.path.join(args.data, "U2-URMA-s*", "s*_conflict_at_*.npz")):
        m = re.search(r"s(\d+)_conflict_at_(\d+)\.npz$", f)
        files[(int(m.group(1)), int(m.group(2)))] = f
    ups = sorted({u for _, u in files})
    seeds = sorted({s for s, _ in files})
    for g in [x.strip() for x in args.groups.split(",")]:
        gi = GROUPS.index(g)
        res = {}
        names = None
        for (s, u), f in files.items():
            z = np.load(f, allow_pickle=True)
            names, res[(s, u)] = per_robot(z, gi, vi)
        for stat, title in (("SF", "S-F  success vs fall episodes"), ("SL", "S-L  success vs low-return (full-length) episodes"), ("SS", "S-S  success vs success"),
                            ("FF", "F-F  fall vs fall"), ("align", "align  persistent cosine to the other robots")):
            print(f"\n== parameters: {g}   {title}   (cell = seed0 / seed1)")
            print("robot ".ljust(7) + "".join(f"{u // 1000:>3d}k".rjust(14) for u in ups))
            for n in names:
                row = n.ljust(7)
                for u in ups:
                    cell = []
                    for s in seeds:
                        v = res.get((s, u), {}).get(stat, {}).get(n, np.nan)
                        cell.append("   -  " if not np.isfinite(v) else f"{v:+.2f}")
                    row += ("/".join(cell)).rjust(14)
                print(row)


if __name__ == "__main__":
    main()
