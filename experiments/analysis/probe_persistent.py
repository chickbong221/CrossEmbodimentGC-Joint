"""Test 1, persistent part: the decoder's joint-pair compatibility averaged over many updates.

A probe measures the compatibility of one parameter point. Within a run most of it changes within ten updates,
and a smaller part persists for thousands (analysis/probe_turnover.py). Averaging the cosine matrices of probes
spread over a stretch of training keeps the persistent part. This script splits a run's probes into two
interleaved sets (every other probe), so the two averages come from different updates and different data, and
reports:
  * reliability of the persistent structure (correlation of the two averages over joint pairs);
  * its agreement with other runs' persistent structure (other seeds), against that reliability;
  * where it sits (same robot / same EG group / other groups) and how groupings fitted on one set separate harm
    and keep help on the other (joints by gradient, EG robot groups, morphology, matched random).

Usage: python analysis/probe_persistent.py LABEL=RUN_DIR [LABEL=RUN_DIR ...] [--min 1000 --max 3000]
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_morse as am
from probe_analysis import DIST, cosine, offdiag, top_split


def load(spec, lo, hi):
    label, run = spec.split("=", 1)
    probes = []
    for f in sorted(glob.glob(os.path.join(run, "morse_probe", "probe_*.npz"))):
        z = np.load(f)
        if lo <= int(z["step"]) <= hi:
            c, ok = cosine(z["Gfull_all"], np.diag(z["Gfull_all"]))
            probes.append((int(z["step"]), np.clip(c, -1, 1), ok, z))
    return label, probes


def average(probes):
    c = np.mean([p[1] for p in probes], axis=0)
    ok = np.all([p[2] for p in probes], axis=0)
    return c, ok


def pair_corr(a, b, ok):
    x, _ = offdiag(a, ok)
    y, _ = offdiag(b, ok)
    return np.corrcoef(x, y)[0, 1]


def split_gain(labels, c_eval, ok):
    """Harm kept (%), help kept (%) inside groups, and the net help kept minus harm kept, on held-out cosines."""
    v, (i, j) = offdiag(c_eval, ok)
    same = labels[i] == labels[j]
    harm, help_ = np.clip(-v, 0, None), np.clip(v, 0, None)
    kept_h, kept_p = harm[same].sum() / max(harm.sum(), 1e-30), help_[same].sum() / max(help_.sum(), 1e-30)
    return 100 * kept_h, 100 * kept_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--min", type=int, default=0)
    parser.add_argument("--max", type=int, default=10 ** 9)
    args = parser.parse_args()
    runs = [load(s, args.min, args.max) for s in args.runs]
    persistent = {}
    for label, probes in runs:
        steps = [p[0] for p in probes]
        a, b = probes[0::2], probes[1::2]
        ca, oka = average(a)
        cb, okb = average(b)
        ok = oka & okb
        rel = pair_corr(ca, cb, ok)
        call, okall = average(probes)
        persistent[label] = (call, okall, 2 * rel / (1 + rel))
        z = probes[0][3]
        tr, te = z["type_robot"], z["type_elem"]
        names = [am.canonical(str(n)) for n in z["robot_names"]]
        print(f"== {label}: {len(probes)} probes, updates {steps[0]}..{steps[-1]}")
        print(f"   persistent structure: reliability (interleaved halves) {rel:.2f}; single-probe half-data reliability "
              f"{np.mean([pair_corr(*[np.clip(cosine(p[3][g], np.diag(p[3][g]))[0], -1, 1) for g in ('G12_all', 'G34_all')], p[2]) for p in probes]):.2f}")
        v, (i, j) = offdiag(call, okall)
        harm, help_ = np.clip(-v, 0, None), np.clip(v, 0, None)
        eg4 = am.eg_robot_groups(DIST, names, 4)
        same_r, same_g = tr[i] == tr[j], eg4[tr[i]] == eg4[tr[j]]
        for nm, m in (("same robot", same_r), ("same EG group", same_g & ~same_r), ("other groups", ~same_g)):
            print(f"   {nm:<14} pairs {100 * m.mean():5.1f}%  harm {100 * harm[m].sum() / harm.sum():5.1f}%  "
                  f"help {100 * help_[m].sum() / help_.sum():5.1f}%  mean cos {v[m].mean():+.3f}")
        # Groupings fitted on one interleaved set, evaluated on the other.
        camps, share = top_split(ca, ok)
        dist = 1 - ca[np.ix_(ok, ok)]
        np.fill_diagonal(dist, 0)
        k2 = np.zeros(len(ca), int)
        k2[ok] = fcluster(linkage(squareform(np.clip(0.5 * (dist + dist.T), 0, None), checks=False), "average"), 2, "maxclust")
        robot_split = (eg4 != eg4[names.index("unitree_a1")]).astype(int)[tr]
        desc = z["type_desc"]
        dd = np.linalg.norm(desc[:, None] - desc[None], axis=-1)
        morph = fcluster(linkage(squareform(dd, checks=False), "ward"), 2, "maxclust")
        rng = np.random.default_rng(0)
        rand = camps.copy()
        for r in np.unique(tr):
            idx = np.where(tr == r)[0]
            rand[idx] = rng.permutation(camps[idx])
        out = []
        for nm, lab in (("camps (top eigvec)", camps), ("joints by gradient", k2), ("EG robot split", robot_split),
                        ("morphology", morph), ("matched random", rand)):
            h, p = split_gain(lab, cb, ok)
            out.append(f"{nm} {h:.0f}/{p:.0f}")
        print(f"   K=2 split fit on one half, harm kept % / help kept % on the other: " + " | ".join(out))
        print(f"   top eigenvector {100 * share:.1f}% of the persistent structure; camps {np.sum(camps == 0)}/{np.sum(camps == 1)}")
    labels = list(persistent)
    for x in range(len(labels)):
        for y in range(x + 1, len(labels)):
            (cx, okx, rx), (cy, oky, ry) = persistent[labels[x]], persistent[labels[y]]
            ok = okx & oky
            raw = pair_corr(cx, cy, ok)
            print(f"== {labels[x]} vs {labels[y]}: persistent structures correlate {raw:.2f} "
                  f"(normalized {raw / np.sqrt(max(rx, 1e-3) * max(ry, 1e-3)):.2f})")


if __name__ == "__main__":
    main()
