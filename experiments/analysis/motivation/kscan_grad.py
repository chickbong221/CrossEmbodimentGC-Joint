"""Which number of joint types k gives the clearest gradient structure in the joint encoder and the decoder, and at which
checkpoint do joint types separate best from robot groups and from no grouping?

Units: single joints (robot, joint). Pairs: joints of different robots. Cosine: EG's plain cosine of the actor gradients
(each averaged over the 32 analysis minibatches). Joint types: k-means (repo kmeans, seed 0) on the 9 standardized joint
features of the revised table. Robot groups: Ward clustering of the FGW body distance cut at M groups (as EG's code).
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cells_metrics import FGW, GROUPS, REPO, ZR, cos_from_gram, mean_gram  # noqa: E402
from jg_kmeans import kmeans  # noqa: E402


def robot_groups(M):
    X = FGW.copy()
    np.fill_diagonal(X, 0)
    return cut_tree(linkage(squareform(X, checks=False), method="ward"), n_clusters=M).flatten()


files = []
for f in glob.glob(os.path.join(REPO, "experiments/analysis/results/conflict/data/U2-URMA-s*/s*_conflict_at_*.npz")):
    m = re.search(r"s(\d+)_conflict_at_(\d+)\.npz$", f)
    files.append((int(m.group(2)), int(m.group(1)), f))
files.sort()

KS = list(range(3, 13))
jt = {k: kmeans(ZR, k, 0)[0] for k in KS}
rg = {M: robot_groups(M) for M in sorted(set(KS) | {4})}
print("robot groups M=4:", rg[4].tolist())

rows = []
for upd, seed, f in files:
    z = np.load(f, allow_pickle=True)
    robot = z["unit_robot"].astype(int)
    iu = np.triu_indices(len(robot), 1)
    cross = robot[iu[0]] != robot[iu[1]]
    row = {"upd": upd, "seed": seed}
    for b in ("enc", "dec"):
        C = cos_from_gram(mean_gram(z, GROUPS.index(b)))[iu]
        v = C[cross]
        row[b] = {"none": (float(v.mean()), float((v < 0).mean()))}
        for k in KS:
            lab = jt[k]
            same = (lab[iu[0]] == lab[iu[1]])[cross]
            # type x type matrix on (robot, type) summed gradients, pairs of different robots
            keys = sorted({(int(r), int(g)) for r, g in zip(robot, lab)})
            K = np.zeros((len(robot), len(keys)))
            for q, (r, g) in enumerate(keys):
                K[(robot == r) & (lab == g), q] = 1.0
            kr = np.array([x[0] for x in keys]); kg = np.array([x[1] for x in keys])
            Cg = cos_from_gram(K.T @ mean_gram(z, GROUPS.index(b)) @ K)
            ju = np.triu_indices(len(keys), 1)
            T = np.full((k, k), np.nan)
            for g in range(k):
                for h in range(k):
                    msk = (kr[ju[0]] != kr[ju[1]]) & (((kg[ju[0]] == g) & (kg[ju[1]] == h)) | ((kg[ju[0]] == h) & (kg[ju[1]] == g)))
                    if msk.any():
                        T[g, h] = Cg[ju][msk].mean()
            diag = np.diag(T)
            off = np.where(np.eye(k, dtype=bool), np.nan, T)
            margin = [diag[g] - np.nanmax(off[g]) for g in range(k) if np.isfinite(diag[g])]
            rlab = rg[k][robot]
            rsame = (rlab[iu[0]] == rlab[iu[1]])[cross]
            r4 = (rg[4][robot][iu[0]] == rg[4][robot][iu[1]])[cross]
            row[b][k] = {"joint_in": (float(v[same].mean()), float((v[same] < 0).mean())),
                         "joint_out": (float(v[~same].mean()), float((v[~same] < 0).mean())),
                         "robot_in": (float(v[rsame].mean()), float((v[rsame] < 0).mean())),
                         "robot4_in": (float(v[r4].mean()), float((v[r4] < 0).mean())),
                         "diag": float(np.nanmean(diag)), "off": float(np.nanmean(off)),
                         "wins": int(sum(m > 0 for m in margin)), "min_margin": float(min(margin)),
                         "T": T.tolist()}
    rows.append(row)
    print("done", upd, seed, file=sys.stderr, flush=True)
json.dump({"rows": rows, "jt": {k: jt[k].tolist() for k in KS}, "rg": {M: rg[M].tolist() for M in rg}},
          open(os.path.join(HERE, "kscan_grad.json"), "w"))

sel = [r for r in rows if r["upd"] <= 75000]
print("\nPer k, mean over 12 checkpoints (2k-75k): joint-level same-type vs different-type cosine; type matrix diag/off; "
      "types whose own type is their best match (wins); worst margin")
for b in ("enc", "dec"):
    print(f"--- {b}")
    for k in KS:
        m = lambda f: np.mean([f(r[b][str(k)] if str(k) in r[b] else r[b][k]) for r in sel])
        print(f" k={k:2d}  same {m(lambda x: x['joint_in'][0]):+.3f}  diff {m(lambda x: x['joint_out'][0]):+.3f}  "
              f"ratio {m(lambda x: x['joint_in'][0]) / m(lambda x: x['joint_out'][0]):.2f} | diag {m(lambda x: x['diag']):+.3f} "
              f"off {m(lambda x: x['off']):+.3f} ratio {m(lambda x: x['diag']) / m(lambda x: x['off']):.2f} | "
              f"wins {m(lambda x: x['wins'] / k):.2f} | min margin {m(lambda x: x['min_margin']):+.3f} | "
              f"robot(M=k) {m(lambda x: x['robot_in'][0]):+.3f}  none {np.mean([r[b]['none'][0] for r in sel]):+.3f}")
