"""Checks requested by the 2026-10-02 review, on existing probes and snapshots (no new runs).

  transfer   does the compatibility estimate predict measured transfer? T[a, b] is the held-out loss decrease of
             joint type b after a small step along a's gradient direction (probe lookahead, step norm 0.002, on a
             fifth data subset). Compared with the cross-fitted cosine of the same probe (estimated on four other
             subsets), with cosines of EARLIER probes, and with the mean cosine of all earlier probes. Columns of T
             are rescaled to unit RMS, because T[a, b] ~ eta |g_b| cos(a, b) also carries b's gradient norm.
  magnitude  interaction strength, not only reproducibility: per probe, cross-type helpful (P) and harmful (N)
             inner-product mass relative to the types' own squared norms (S), and the measured transfer within
             and across the two camps, for runs with and without the per-joint output calibration.
  activity   did the signed compatibility loss move routing as intended? For pairs flagged at one snapshot (fixed
             sets), the change of their routing overlap r_a . r_b on the flagged expert by the next snapshot,
             relative to all pairs with evidence.

Usage: python analysis/review_checks.py transfer RUN_DIR [RUN_DIR ...]
       python analysis/review_checks.py magnitude LABEL=RUN_DIR [LABEL=RUN_DIR ...] [--min 0 --max 1e9]
       python analysis/review_checks.py activity RUN_DIR [--z 2.0]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_analysis import cosine, top_split


def probes(run, lo=0, hi=10 ** 9):
    out = []
    for f in sorted(glob.glob(os.path.join(run, "morse_probe", "probe_*.npz"))):
        z = np.load(f)
        if lo <= int(z["step"]) <= hi:
            out.append(z)
    return out


def full_cos(z):
    c, ok = cosine(z["Gfull_all"], np.diag(z["Gfull_all"]))
    return np.clip(c, -1, 1), ok


def transfer_matrix(z):
    t = np.array(z["T_0"], dtype=np.float64)
    rms = np.sqrt(np.mean(t ** 2, axis=0))
    return t / np.maximum(rms, 1e-30)[None, :]


def corr_off(c, t, ok):
    m = ok[:, None] & ok[None, :] & ~np.eye(len(ok), dtype=bool)
    return np.corrcoef(c[m], t[m])[0, 1]


def cmd_transfer(args):
    for run in args.runs:
        ps = probes(run)
        print(f"== {os.path.basename(os.path.normpath(run))}: probes at {[int(z['step']) for z in ps]}")
        cs = [full_cos(z) for z in ps]
        ts = [transfer_matrix(z) for z in ps]
        same = [corr_off(c, t, ok) for (c, ok), t in zip(cs, ts)]
        print("   same probe:      " + "  ".join(f"{int(z['step'])}: {r:.2f}" for z, r in zip(ps, same)))
        lags = {}
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                ok = cs[i][1] & cs[j][1]
                lags.setdefault(int(ps[j]["step"]) - int(ps[i]["step"]), []).append(corr_off(cs[i][0], ts[j], ok))
        if lags:
            print("   earlier probe -> later transfer, by lag: " +
                  "  ".join(f"{lag}: {np.mean(v):.2f}" for lag, v in sorted(lags.items())))
        if len(ps) > 2:
            rows = []
            for j in range(2, len(ps)):
                c = np.mean([cs[i][0] for i in range(j)], axis=0)
                ok = np.all([cs[i][1] for i in range(j + 1)], axis=0)
                rows.append(f"{int(ps[j]['step'])}: {corr_off(c, ts[j], ok):.2f}")
            print("   mean of all earlier probes -> transfer: " + "  ".join(rows))


def cmd_magnitude(args):
    for spec in args.runs:
        label, run = spec.split("=", 1)
        rows = []
        for z in probes(run, args.min, args.max):
            g = z["Gfull_all"]
            d = np.diag(g).copy()
            off = g - np.diag(d)
            s, p, n = d.sum(), off[off > 0].sum(), -off[off < 0].sum()
            c, ok = full_cos(z)
            camps, _ = top_split(c, ok)
            t = np.array(z["T_0"], dtype=np.float64)
            m = ok[:, None] & ok[None, :] & ~np.eye(len(ok), dtype=bool)
            within = m & (camps[:, None] == camps[None, :])
            across = m & (camps[:, None] != camps[None, :])
            rows.append((int(z["step"]), p / s, n / s, n / p, t[within].mean() * 1e5, t[across].mean() * 1e5))
        a = np.array(rows)
        print(f"== {label}: {len(rows)} probes, updates {int(a[0, 0])}..{int(a[-1, 0])}")
        for r in rows:
            print(f"   {r[0]:>6}: P/S {r[1]:5.1f}  N/S {r[2]:5.1f}  N/P {r[3]:.2f}  transfer within camps {r[4]:+.2f}  "
                  f"across {r[5]:+.2f} (x1e-5)")
        print(f"   mean: P/S {a[:, 1].mean():5.1f}  N/S {a[:, 2].mean():5.1f}  N/P {a[:, 3].mean():.2f}  "
              f"within {a[:, 4].mean():+.2f}  across {a[:, 5].mean():+.2f}")


def cmd_activity(args):
    files = sorted(glob.glob(os.path.join(args.runs[0], "morse_diagnostics", "step_*.npz")))
    snaps = []
    for f in files:
        z = np.load(f)
        step = int(z["step"])
        if step == 0:
            continue
        tr, te = z["decoder__type_robot"], z["decoder__type_elem"]
        index = {(int(r), int(e)): i for i, (r, e) in enumerate(zip(tr, te))}
        types = np.array([index[(int(r), int(e))] for r, e in zip(z["decoder__robot_ids"], z["decoder__elem_ids"])])
        onehot = np.eye(len(tr))[types]
        r_type = onehot.T @ z["decoder__unit_p"] / np.maximum(onehot.sum(0), 1)[:, None]        # (T, K)
        cos = z["decoder__type_cos"]                                                              # (K, T, T) upper
        sigma = float(z["scalar__morse.decoder.compat_sigma"])
        snaps.append((step, r_type, cos, sigma))
    print(f"== {os.path.basename(os.path.normpath(args.runs[0]))}: overlap r_a.r_b on the flagged expert, change to the "
          f"next snapshot, relative to all pairs with evidence (z = {args.z})")
    for (s0, r0, c0, sg), (s1, r1, _, _) in zip(snaps, snaps[1:]):
        k_n, n_t = c0.shape[0], c0.shape[1]
        iu = np.triu_indices(n_t, 1)
        ev = np.stack([c0[k][iu] for k in range(k_n)])                                          # (K, pairs)
        has = ev != 0
        o0 = np.stack([r0[iu[0], k] * r0[iu[1], k] for k in range(k_n)])
        o1 = np.stack([r1[iu[0], k] * r1[iu[1], k] for k in range(k_n)])
        harm, help_ = has & (ev < -args.z * sg), has & (ev > args.z * sg)
        d_all = (o1 - o0)[has].mean()
        line = (f"   {s0 // 1000:>3}k->{s1 // 1000:>3}k: evidence on {100 * has.mean():4.1f}% of (expert, pair); "
                f"harmful {harm.sum():5d}, helpful {help_.sum():5d}; overlap at {s0 // 1000}k: harmful "
                f"{o0[harm].mean() / o0[has].mean():.2f}, helpful {o0[help_].mean() / o0[has].mean():.2f} (x all); "
                f"change: harmful {(o1 - o0)[harm].mean() - d_all:+.4f}, helpful {(o1 - o0)[help_].mean() - d_all:+.4f} "
                f"(minus all {d_all:+.4f})")
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=("transfer", "magnitude", "activity"))
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--min", type=int, default=0)
    parser.add_argument("--max", type=int, default=10 ** 9)
    parser.add_argument("--z", type=float, default=2.0)
    args = parser.parse_args()
    {"transfer": cmd_transfer, "magnitude": cmd_magnitude, "activity": cmd_activity}[args.cmd](args)


if __name__ == "__main__":
    main()
