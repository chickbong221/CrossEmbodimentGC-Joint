"""Test 1, time scale: how long does the decoder's joint-pair compatibility structure last within a run?

For probes of one run at nearby updates (``<run>/morse_probe/probe_<updates>.npz``), compares the full-data
cosine matrices of every pair of probes. Each probe has its own measurement noise, so the raw correlation is
divided by the largest correlation two probes of the same parameters could reach: the geometric mean of their
full-data reliabilities. A probe's reliability is that of its two independent halves (Q1Q2 vs Q3Q4), stepped
up to the full data (Spearman-Brown, 2r / (1 + r)). A normalized correlation of 1 means the structure did not
change between the two updates; 0 means it was replaced.

With a reference run (``--ref``, e.g. the long run that wrote the branched checkpoint), probes of the reference
at the same updates are compared as well: the same seed (same initial parameters and data order) reproduces
the structure only if the training is deterministic enough.

Usage: python analysis/probe_turnover.py RUN_DIR [RUN_DIR ...] [--ref RUN_DIR]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_analysis import cosine, offdiag


def load(run):
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "morse_probe", "probe_*.npz"))):
        z = np.load(f)
        c, ok = cosine(z["Gfull_all"], np.diag(z["Gfull_all"]))
        c12, ok12 = cosine(z["G12_all"], np.diag(z["G12_all"]))
        c34, ok34 = cosine(z["G34_all"], np.diag(z["G34_all"]))
        o = ok12 & ok34
        a, _ = offdiag(np.clip(c12, -1, 1), o)
        b, _ = offdiag(np.clip(c34, -1, 1), o)
        r = max(np.corrcoef(a, b)[0, 1], 1e-3)
        out[int(z["step"])] = (np.clip(c, -1, 1), ok, r, 2 * r / (1 + r))
    return out


def corr(p, q):
    ok = p[1] & q[1]
    a, _ = offdiag(p[0], ok)
    b, _ = offdiag(q[0], ok)
    raw = np.corrcoef(a, b)[0, 1]
    return raw, raw / np.sqrt(p[3] * q[3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--ref", default="")
    args = parser.parse_args()
    ref = load(args.ref) if args.ref else {}
    for run in args.runs:
        probes = load(run)
        steps = sorted(probes)
        print(f"== {os.path.basename(os.path.normpath(run))}: probes at {steps}")
        print("   half-data reliability: " + " ".join(f"{s}:{probes[s][2]:.2f}" for s in steps))
        print("   normalized correlation with the first probe, by lag (raw in parentheses):")
        s0 = steps[0]
        print("     " + "  ".join(f"+{s - s0}: {corr(probes[s0], probes[s])[1]:.2f} ({corr(probes[s0], probes[s])[0]:.2f})"
                                  for s in steps[1:]))
        print("   normalized correlation of consecutive probes:")
        print("     " + "  ".join(f"{a}->{b}: {corr(probes[a], probes[b])[1]:.2f}" for a, b in zip(steps, steps[1:])))
        lags = {}
        for i, a in enumerate(steps):
            for b in steps[i + 1:]:
                lags.setdefault(b - a, []).append(corr(probes[a], probes[b])[1])
        print("   mean normalized correlation by lag (all probe pairs): " +
              "  ".join(f"{lag}: {np.mean(v):.2f}" for lag, v in sorted(lags.items())))
        shared = sorted(set(steps) & set(ref))
        if shared:
            print("   same updates in the reference run (same seed): " +
                  "  ".join(f"{s}: {corr(probes[s], ref[s])[1]:.2f}" for s in shared))
        near = [(s, s - 1) for s in steps if (s - 1) in ref]
        if near:
            print("   one update after a reference probe: " +
                  "  ".join(f"{s} vs ref {r}: {corr(probes[s], ref[r])[1]:.2f}" for s, r in near))


if __name__ == "__main__":
    main()
