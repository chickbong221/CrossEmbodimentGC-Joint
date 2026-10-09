"""Test 1: is there joint-level sharing structure in the decoder, and is it more than output calibration?

Reads the decoder probes (``<run>/morse_probe/probe_<updates>.npz``, written by morse/probe.py) of one or more
runs and prints, per probe:
  * reliability: correlation of the joint-pair cosines from two independent data halves (Q1Q2 vs Q3Q4);
  * structure: share of the top eigenvector of the cosine matrix, and the two "camps" given by its sign;
  * where interaction sits: same robot / same EG group / other groups, against pair-count and norm-weighted
    fair shares;
  * splits fitted on Q1Q2 and evaluated on Q3Q4 (out of sample): harm and help kept inside groups and the
    first-order gain (N_out - P_out) / S of separating them, for robot-level (EG, gradient-clustered robots),
    joint-level (camps, gradient-clustered joints), morphology and matched random groupings;
  * layers: share of the camp contrast carried by each parameter group (first layer, LayerNorm, output
    weights, output bias);
  * lookahead: whether the measured held-out effect of a step along one joint's gradient (T) matches the
    gradient prediction (P), and whether steps help within camps and hurt across them;
  * nonlinear predictability of the camps from the joint descriptors (leave-one-robot-out k-NN and trees);
and, across runs at the same update, the agreement of their cosine matrices and camps.

Usage: python analysis/probe_analysis.py LABEL=RUN_DIR [LABEL=RUN_DIR ...] [--steps 1000,2000,5000]
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

GROUPS = ("dense0", "ln0", "out_w", "out_b")
DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "meta", "embodiment_distance.npy")


def cosine(g, diag):
    ok = diag > 0
    d = np.sqrt(np.where(ok, diag, 1.0))
    c = g / np.outer(d, d)
    c[~ok, :] = 0.0
    c[:, ~ok] = 0.0
    return c, ok


def offdiag(m, ok):
    iu = np.triu_indices(len(m), 1)
    keep = ok[iu[0]] & ok[iu[1]]
    return m[iu][keep], (iu[0][keep], iu[1][keep])


def top_split(c, ok):
    """Camps (sign of the top eigenvector of the clipped cosine matrix) and that eigenvector's share of the
    positive spectrum (cross-data cosines can exceed 1 when a type's signal is weak, so they are clipped)."""
    sub = np.clip(c[np.ix_(ok, ok)], -1.0, 1.0)
    ev, vec = np.linalg.eigh(0.5 * (sub + sub.T))
    v = np.zeros(len(c))
    v[ok] = vec[:, -1] * np.sign(vec[:, -1].sum())
    return (v > 0).astype(int), ev[-1] / max(ev[ev > 0].sum(), 1e-30)


def kept(g, labels, ok):
    """Harm and help kept inside groups (%), and the first-order gain of separating the groups, (N_out - P_out)/S."""
    vals, (i, j) = offdiag(g, ok)
    same = labels[i] == labels[j]
    neg, pos = np.maximum(-vals, 0), np.maximum(vals, 0)
    s = np.trace(g[np.ix_(ok, ok)])
    return (100 * neg[same].sum() / max(neg.sum(), 1e-30), 100 * pos[same].sum() / max(pos.sum(), 1e-30),
            2 * (neg[~same].sum() - pos[~same].sum()) / max(s, 1e-30))


def cluster(c, k):
    d = np.clip(1.0 - 0.5 * (c + c.T), 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    return fcluster(linkage(squareform(d, checks=False), method="average"), t=k, criterion="maxclust") - 1


def matched_random(labels, robots, rng):
    """Random labels with the same number of joints per label within every robot."""
    out = labels.copy()
    for r in np.unique(robots):
        idx = np.where(robots == r)[0]
        out[idx] = rng.permutation(labels[idx])
    return out


def _knn(xtr, ytr, xte, k=5):
    d = ((xte[:, None, :] - xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1)[:, :k]
    return (ytr[idx].mean(1) > 0.5).astype(int)


def _kernel_ridge(xtr, ytr, xte, lam=1e-2):
    """RBF kernel ridge classifier (median-distance bandwidth), a nonlinear model without scikit-learn."""
    d_tr = ((xtr[:, None, :] - xtr[None, :, :]) ** 2).sum(-1)
    gamma = 1.0 / max(np.median(d_tr[d_tr > 0]), 1e-12)
    alpha = np.linalg.solve(np.exp(-gamma * d_tr) + lam * len(xtr) * np.eye(len(xtr)), 2.0 * ytr - 1.0)
    d_te = ((xte[:, None, :] - xtr[None, :, :]) ** 2).sum(-1)
    return (np.exp(-gamma * d_te) @ alpha > 0).astype(int)


def predictability(desc, labels, robots, ok):
    """Leave-one-robot-out accuracy of predicting the camp from the joint descriptor (nonlinear models)."""
    x = (desc - desc[ok].mean(0)) / (desc[ok].std(0) + 1e-8)
    accs = {}
    for name, fit_predict in (("k-NN", _knn), ("RBF kernel", _kernel_ridge)):
        hit = tot = 0
        for r in np.unique(robots[ok]):
            test = ok & (robots == r)
            train = ok & (robots != r)
            if len(np.unique(labels[train])) < 2:
                continue
            hit += (fit_predict(x[train], labels[train], x[test]) == labels[test]).sum()
            tot += test.sum()
        accs[name] = hit / max(tot, 1)
    majority = max(np.mean(labels[ok]), 1 - np.mean(labels[ok]))
    return accs, majority


def analyze(label, z, eg4, rng):
    tr, step = z["type_robot"], int(z["step"])
    names = [str(n) for n in z["robot_names"]]
    gf, g12, g34 = z["Gfull_all"], z["G12_all"], z["G34_all"]
    cf, ok = cosine(gf, np.diag(gf))
    c12, ok12 = cosine(g12, np.diag(g12))
    c34, ok34 = cosine(g34, np.diag(g34))
    okb = ok & ok12 & ok34
    a, _ = offdiag(c12, okb)
    b, _ = offdiag(c34, okb)
    rel = np.corrcoef(a, b)[0, 1]
    sign = np.mean(np.sign(a) == np.sign(b))
    snr = np.diag(gf) / np.maximum(z["N11_all"] * 4, 1e-30)      # (S1+S2).(S3+S4) = 4 S1.S2 in expectation
    camps, share = top_split(cf, ok)
    print(f"\n== {label} at {step} updates: types {ok.sum()}/{len(ok)} with signal; reliability Q12 vs Q34 "
          f"r={rel:.2f} (sign agreement {sign:.2f}); median signal fraction of a type's gradient {np.median(snr[ok]):.2f}")
    vals, (i, j) = offdiag(gf, ok)
    neg, pos = np.maximum(-vals, 0), np.maximum(vals, 0)
    s = np.trace(gf[np.ix_(ok, ok)])
    print(f"   N/S {2 * neg.sum() / s:.2f} P/S {2 * pos.sum() / s:.2f}; top eigenvector {100 * share:.1f}% of the "
          f"cosine structure; camps {np.sum(camps[ok] == 1)}/{np.sum(camps[ok] == 0)} joints")
    same_robot = tr[i] == tr[j]
    same_eg = eg4[tr[i]] == eg4[tr[j]]
    w = np.sqrt(np.outer(np.diag(gf), np.diag(gf)).clip(0))[i, j]
    parts = []
    for name, m in (("same robot", same_robot), ("same EG group", same_eg & ~same_robot), ("other groups", ~same_eg)):
        parts.append(f"{name}: pairs {100 * m.mean():.1f} fair {100 * w[m].sum() / w.sum():.1f} "
                     f"harm {100 * neg[m].sum() / neg.sum():.1f} help {100 * pos[m].sum() / pos.sum():.1f}")
    print("   where: " + " | ".join(parts))
    # Splits fitted on Q1Q2, evaluated on Q3Q4.
    camps12, _ = top_split(c12, okb)
    robots_ids = np.unique(tr)
    agg = np.eye(len(names))[tr]
    rc, rok = cosine(agg.T @ g12 @ agg, np.diag(agg.T @ g12 @ agg))
    eg2 = (eg4 != eg4[names.index("unitree_a1")]).astype(int) if "unitree_a1" in names else (eg4 > 0).astype(int)
    feat = z["type_desc"]
    fx = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
    morph = {k: fcluster(linkage(fx, method="ward"), t=k, criterion="maxclust") - 1 for k in (2, 4)}
    for k in (2, 4):
        cands = {"EG robot groups": (eg2 if k == 2 else eg4)[tr],
                 "robots by gradient": cluster(rc, k)[tr],
                 "joints by gradient": cluster(c12, k) if k > 2 else camps12,
                 "joint morphology": morph[k]}
        out = [f"{n} {h:3.0f}/{p:3.0f}/{gn:+.2f}" for n, (h, p, gn) in ((n, kept(g34, lab, okb)) for n, lab in cands.items())]
        rnd = np.mean([kept(g34, matched_random(cands["joints by gradient"], tr, rng), okb) for _ in range(20)], axis=0)
        out.append(f"matched random {rnd[0]:3.0f}/{rnd[1]:3.0f}/{rnd[2]:+.2f}")
        print(f"   K={k} split (harm kept %/help kept %/gain), fit Q12 eval Q34: " + " | ".join(out))
    # Layers: share of the cross-camp interaction carried by each parameter group (groups add up to the decoder).
    cross = camps[i] != camps[j]
    tot_cross = vals[cross].sum()
    shares = []
    for grp in GROUPS:
        gv, _ = offdiag(z[f"Gfull_{grp}"], ok)
        shares.append(f"{grp} {100 * gv[cross].sum() / tot_cross:5.1f}%")
    print(f"   layers: cross-camp interaction (sum {tot_cross:+.2e}) carried by " + ", ".join(shares))
    for grp in GROUPS:
        cg, okg = cosine(z[f"Gfull_{grp}"], np.diag(z[f"Gfull_{grp}"]))
        cg_camps, sh = top_split(cg, okg)
        agree = np.mean(cg_camps[ok & okg] == camps[ok & okg])
        print(f"      {grp}: top eigenvector {100 * sh:.1f}%, camps agree with the whole decoder's on "
              f"{100 * max(agree, 1 - agree):.0f}% of joints")
    # Lookahead.
    looks = [(f"step norm {eta:g}", str(k)) for k, eta in enumerate(z["step_norms"])]
    if "T_sign" in z:
        looks.append((f"Adam-like sign step (lr {float(z['sign_lr']):g})", "sign"))
    for desc_, key in looks:
        t, p = z[f"T_{key}"], z[f"P_{key}"]
        tv, (ti, tj) = offdiag(0.5 * (t + t.T), ok)
        pv, _ = offdiag(0.5 * (p + p.T), ok)
        cv, _ = offdiag(cf, ok)
        big = np.abs(pv) > np.median(np.abs(pv))
        within = camps[ti] == camps[tj]
        print(f"   lookahead, {desc_}: corr(measured, predicted) {np.corrcoef(tv, pv)[0, 1]:.2f}, sign agreement "
              f"{np.mean(np.sign(tv[big]) == np.sign(pv[big])):.2f} (larger half), corr(measured, cosine) "
              f"{np.corrcoef(tv, cv)[0, 1]:.2f}; mean effect within camps {tv[within].mean():+.2e}, across {tv[~within].mean():+.2e}")
    accs, majority = predictability(z["type_desc"], camps, tr, ok)
    print(f"   camps from joint descriptors (leave one robot out): " +
          ", ".join(f"{n} {100 * v:.0f}%" for n, v in accs.items()) + f" (majority class {100 * majority:.0f}%)")
    return {"cos": cf, "ok": ok, "camps": camps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--steps", default="")
    args = parser.parse_args()
    steps = {int(s) for s in args.steps.split(",") if s.strip()}
    rng = np.random.default_rng(0)
    results = {}
    for spec in args.runs:
        label, pattern = spec.split("=", 1)
        dirs = glob.glob(pattern)
        if not dirs:
            print(f"no run for {spec}")
            continue
        files = sorted(glob.glob(os.path.join(dirs[0], "morse_probe", "probe_*.npz")))
        for f in files:
            z = np.load(f)
            step = int(z["step"])
            if steps and step not in steps:
                continue
            names = [am.canonical(str(n)) for n in z["robot_names"]]
            eg4 = am.eg_robot_groups(DIST, names, 4)
            results[(label, step)] = analyze(label, z, eg4, rng)
    labels = sorted({k[0] for k in results})
    for step in sorted({k[1] for k in results}):
        have = [l for l in labels if (l, step) in results]
        for x in range(len(have)):
            for y in range(x + 1, len(have)):
                rx, ry = results[(have[x], step)], results[(have[y], step)]
                ok = rx["ok"] & ry["ok"]
                a, _ = offdiag(rx["cos"], ok)
                b, _ = offdiag(ry["cos"], ok)
                agree = np.mean(rx["camps"][ok] == ry["camps"][ok])
                print(f"\n== across runs at {step}: {have[x]} vs {have[y]}: corr of joint-pair cosines "
                      f"{np.corrcoef(a, b)[0, 1]:.2f}; camps agree on {100 * max(agree, 1 - agree):.0f}% of joints")


if __name__ == "__main__":
    main()
