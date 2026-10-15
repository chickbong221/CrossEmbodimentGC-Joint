"""Candidate views (following EG, Gradient Vaccine, Recon, M3DT) of the gradient structure of the layers MORSE replaces.

Blocks: enc_moe (joint-encoder hidden layer) and dec (decoder). Units: joints. Cosine: of the actor gradients, each
averaged over the 32 analysis minibatches. Groupings = experts: shared (one expert), robot4 (EG's 4 robot groups),
joint8 (the 8 joint types).
Per checkpoint and block:
  - per-expert mean within-expert pair cosine (averaged over experts)
  - M3DT-style alignment: cos(g_u, update of u's expert without u), mean over joints, and share of joints opposed (< 0)
  - Recon-style: share of pairs that share an expert and conflict (C < 0, C < -0.1), relative to the shared layer
  - EG-style correlations: r(C, joint similarity) and r(C, body similarity) over pairs of joints of different robots
Also saves the cosine matrices for heatmaps.
python views_srv.py TABLE.json DIST.npy ZR.npy OUTDIR FILE.npz [...]
"""
import json
import os
import re
import sys

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform

table, dist, zr, outdir, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
os.makedirs(outdir, exist_ok=True)
FGW = np.load(dist)
X = FGW.copy(); np.fill_diagonal(X, 0)
ROBOT4 = cut_tree(linkage(squareform(X, checks=False), "ward"), n_clusters=4).flatten()
iu16 = np.triu_indices(16, 1)
BODY_SIM = 1 - (FGW - FGW[iu16].min()) / (FGW[iu16].max() - FGW[iu16].min())
T = json.load(open(table))
JT8 = np.concatenate([T["groups"][n] for n in T["groups"]])
ZR = np.load(zr)
JD = np.sqrt(((ZR[:, None] - ZR[None]) ** 2).sum(-1))
iuJ = np.triu_indices(len(ZR), 1)
JOINT_SIM = 1 - (JD - JD[iuJ].min()) / (JD[iuJ].max() - JD[iuJ].min())


def cosines(G):
    d = np.sqrt(np.maximum(np.diag(G), 1e-30))
    return G / (d[:, None] * d[None, :])


def views(G, robot):
    C = cosines(G)
    n = len(robot)
    iu = np.triu_indices(n, 1)
    cross = robot[iu[0]] != robot[iu[1]]
    out = {}
    for name, lab in (("shared", np.zeros(n, int)), ("robot4", ROBOT4[robot]), ("joint8", JT8)):
        same = lab[iu[0]] == lab[iu[1]]
        v = C[iu][same]
        per = []
        for g in np.unique(lab):
            idx = np.flatnonzero(lab == g)
            if len(idx) > 1:
                per.append(C[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean())
        # M3DT-style: each joint vs the update of its expert without itself
        al = np.full(n, np.nan)
        for g in np.unique(lab):
            idx = np.flatnonzero(lab == g)
            if len(idx) < 2:
                continue
            Gs = G[np.ix_(idx, idx)]
            row = Gs.sum(1)
            tot = Gs.sum()
            for a, u in enumerate(idx):
                num = row[a] - Gs[a, a]
                den2 = tot - 2 * row[a] + Gs[a, a]
                if den2 > 0:
                    al[u] = num / np.sqrt(Gs[a, a] * den2)
        out[name] = {"pair_mean_per_expert": float(np.mean(per)), "pair_mean": float(v.mean()),
                     "n_pairs": int(same.sum()), "n_conflict": int((v < 0).sum()), "n_severe": int((v < -0.1).sum()),
                     "n_agree": int((v > 0).sum()),
                     "align_mean": float(np.nanmean(al)), "align_opposed": float(np.nanmean(al < 0)),
                     "hist": np.histogram(v, bins=np.linspace(-1, 1, 41))[0].tolist()}
    c = C[iu][cross]
    js = JOINT_SIM[iu][cross]
    bs = BODY_SIM[robot[iu[0]], robot[iu[1]]][cross]
    out["r_joint"] = float(np.corrcoef(c, js)[0, 1])
    out["r_body"] = float(np.corrcoef(c, bs)[0, 1])
    return out, C


rows = []
for f in files:
    m = re.search(r"s(\d+)_conflict_at_(\d+)\.npz$", f)
    seed, upd = int(m.group(1)), int(m.group(2))
    z = np.load(f, allow_pickle=True)
    groups = [str(g) for g in z["groups"]]
    robot = z["unit_robot"].astype(int)
    row = {"seed": seed, "upd": upd}
    mats = {}
    for b in ("enc_moe", "dec"):
        gi = groups.index(b)
        Gc = z["half_cross"][gi, 0].astype(np.float64)
        G = 0.25 * (z["half_same1"][gi, 0].astype(np.float64) + z["half_same2"][gi, 0].astype(np.float64) + Gc + Gc.T)
        row[b], mats[b] = views(G, robot)
    np.savez_compressed(os.path.join(outdir, f"cos_s{seed}_{upd}.npz"), robot=robot, **{k: v.astype(np.float32) for k, v in mats.items()})
    rows.append(row)
    print("done", seed, upd, flush=True)
rows.sort(key=lambda r: (r["upd"], r["seed"]))
json.dump(rows, open(os.path.join(outdir, "views.json"), "w"))
