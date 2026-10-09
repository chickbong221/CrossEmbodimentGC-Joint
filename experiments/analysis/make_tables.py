"""Test 2: fixed decoder routing tables (robots, joints, experts) from a decoder probe.

Writes four one-hot tables to OUT_DIR (four experts; group g of joints -> expert g). Branched from a URMA
checkpoint (morse_branch_from), every expert starts as a copy of URMA's decoder and only sees the gradients of
its own group, so the branched policy starts exactly where URMA was:
  shared   every joint on expert 0 (URMA's single shared decoder, continued);
  camps    the two camps of the probe (sign of the top eigenvector of the decoder's joint-pair cosines),
           camp 0 -> expert 0, camp 1 -> expert 1;
  random   the camp labels permuted within every robot (same number of joints per group in every robot);
  robot    whole robots: the EG group of the A1 (legged robots) -> expert 0, the other robots -> expert 1.

With --avg, PROBE is a run folder and the camps come from the mean cosine matrix of its probes between --min and
--max updates: the persistent part of the structure (a single probe is dominated by a part that changes within
ten updates; analysis/probe_turnover.py).

Usage: python analysis/make_tables.py PROBE_NPZ OUT_DIR [--seed 0]
       python analysis/make_tables.py RUN_DIR OUT_DIR --avg --min 1000 --max 3000
"""
import glob
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_morse as am
from probe_analysis import DIST, cosine, top_split

MAX_JOINTS, N_EXPERTS = 32, 4


def table_from_labels(labels, type_robot, type_elem, n_robots):
    t = np.zeros((n_robots, MAX_JOINTS, N_EXPERTS), dtype=np.float32)
    t[:, :, 0] = 1.0                                          # unused slots (padding joints): expert 0
    for lab, r, e in zip(labels, type_robot, type_elem):
        t[r, e] = 0.0
        t[r, e, int(lab)] = 1.0
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe")
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--avg", action="store_true")
    parser.add_argument("--min", type=int, default=0)
    parser.add_argument("--max", type=int, default=10 ** 9)
    args = parser.parse_args()
    if args.avg:
        files = [f for f in sorted(glob.glob(os.path.join(args.probe, "morse_probe", "probe_*.npz")))
                 if args.min <= int(np.load(f)["step"]) <= args.max]
        cs, oks = [], []
        for f in files:
            z = np.load(f)
            c1, ok1 = cosine(z["Gfull_all"], np.diag(z["Gfull_all"]))
            cs.append(np.clip(c1, -1.0, 1.0))
            oks.append(ok1)
        c, ok = np.mean(cs, axis=0), np.all(oks, axis=0)
        source = f"{len(files)} probes of {args.probe} at {args.min}..{args.max} updates"
    else:
        z = np.load(args.probe)
        gf = z["Gfull_all"]
        c, ok = cosine(gf, np.diag(gf))
        source = f"{args.probe} at {int(z['step'])} updates"
    tr, te = z["type_robot"], z["type_elem"]
    names = [am.canonical(str(n)) for n in z["robot_names"]]
    camps, share = top_split(c, ok)
    rng = np.random.default_rng(args.seed)
    rand = camps.copy()
    for r in np.unique(tr):
        idx = np.where(tr == r)[0]
        rand[idx] = rng.permutation(camps[idx])
    eg4 = am.eg_robot_groups(DIST, names, 4)
    robot = (eg4 != eg4[names.index("unitree_a1")]).astype(int)[tr]
    os.makedirs(args.out, exist_ok=True)
    n_robots = len(names)
    tables = {"shared": table_from_labels(np.zeros_like(camps), tr, te, n_robots),
              "camps": table_from_labels(camps, tr, te, n_robots),
              "random": table_from_labels(rand, tr, te, n_robots),
              "robot": table_from_labels(robot, tr, te, n_robots)}
    for name, t in tables.items():
        np.save(os.path.join(args.out, f"{name}.npy"), t)
    np.savez(os.path.join(args.out, "labels.npz"), camps=camps, random=rand, robot=robot, type_robot=tr, type_elem=te,
             robot_names=np.array(names), source=np.array(source), top_share=share)
    print(f"camps from {source} (top eigenvector {100 * share:.1f}%): "
          f"{np.sum(camps == 0)}/{np.sum(camps == 1)} joints; robot split {np.sum(robot == 0)}/{np.sum(robot == 1)}; "
          f"camps vs robot split agree on {100 * max(np.mean(camps == robot), 1 - np.mean(camps == robot)):.0f}%")
    print("per robot (camp-1 joints / joints): " + ", ".join(
        f"{am.SHORT_NAMES.get(names[r], names[r])} {int(camps[tr == r].sum())}/{int((tr == r).sum())}" for r in np.unique(tr)))


if __name__ == "__main__":
    main()
