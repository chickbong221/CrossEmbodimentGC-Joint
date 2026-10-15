"""Per-checkpoint metrics in EG's terms (plain cosine of actor gradients, averaged over the 32 analysis minibatches).

Similarity of two robots = 1 - min-max normalized FGW distance (EG Fig. 3a). Similarity of two joints = 1 - min-max
normalized Euclidean distance of their 9 standardized joint features (the features of the revised joint groups).
"""
import glob
import json
import os
import re
import sys

import numpy as np

REPO = "E:/Code/cross_embodiment_offline_rl"
OLD = os.path.dirname(os.path.abspath(__file__))  # holds ZR.npy (standardized revised joint features)
sys.path.insert(0, os.path.join(REPO, "experiments/analysis"))
from conflict_report import EG, GROUPS, POSITIONAL, load_table, membership  # noqa: E402

VI = 0
TABLE = os.path.join(REPO, "cross_embodiment_offline_rl/algorithms/morse/v8_joint_groups_rev_k8.json")
FGW = np.load(os.path.join(REPO, "data/meta/embodiment_distance.npy"))
BODY_SIM = 1 - (FGW - FGW[np.triu_indices(16, 1)].min()) / (FGW.max() - FGW[np.triu_indices(16, 1)].min())
ZR = np.load(os.path.join(OLD, "ZR.npy"))                       # (241, 9) standardized revised joint features
JD = np.sqrt(((ZR[:, None] - ZR[None]) ** 2).sum(-1))
JOINT_SIM = 1 - (JD - JD[np.triu_indices(len(ZR), 1)].min()) / (JD.max() - JD[np.triu_indices(len(ZR), 1)].min())


def cos_from_gram(G):
    d = np.sqrt(np.maximum(np.diag(G), 1e-30))
    return G / (d[:, None] * d[None, :])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def mean_gram(z, gi):
    Gc = z["half_cross"][gi, VI].astype(np.float64)
    return 0.25 * (z["half_same1"][gi, VI].astype(np.float64) + z["half_same2"][gi, VI].astype(np.float64) + Gc + Gc.T)


def metrics(f):
    z = np.load(f, allow_pickle=True)
    names = [str(n) for n in z["robot_names"]]
    eg = np.array([EG.get(n, EG[POSITIONAL[i]]) for i, n in enumerate(names)])
    robot, joint = z["unit_robot"].astype(int), z["unit_joint"].astype(int)
    tab, _ = load_table(TABLE, names)
    jg = np.array([tab[r][j] for r, j in zip(robot, joint)])
    iu = np.triu_indices(len(robot), 1)
    cross = robot[iu[0]] != robot[iu[1]]
    jsim, bsim = JOINT_SIM[iu][cross], BODY_SIM[robot[iu[0]], robot[iu[1]]][cross]
    sj = (jg[iu[0]] == jg[iu[1]])[cross]
    se = (eg[robot][iu[0]] == eg[robot][iu[1]])[cross]
    ru = np.triu_indices(16, 1)
    out = {}
    for b in ("enc", "core", "dec"):
        G = mean_gram(z, GROUPS.index(b))
        C = cos_from_gram(G)[iu][cross]
        M = membership(robot, 16)
        Cr = cos_from_gram(M.T @ G @ M)
        rc = Cr[ru]
        same_r = eg[ru[0]] == eg[ru[1]]
        # EG step: robots whose gradient opposes the step that moves them (plain sum vs one EG round of group means)
        A = np.zeros((16, eg.max() + 1)); A[np.arange(16), eg] = 1; A /= A.sum(0, keepdims=True)
        Gr = M.T @ G @ M
        plain, rnd = np.ones(16), A.sum(1) * 0 + (A @ np.ones(A.shape[1]))
        w_plain, w_round = np.ones(16), A @ np.ones(A.shape[1])          # weight of each robot in the step
        def opp(w):
            step = Gr @ w
            return float(np.mean(step / np.sqrt(np.diag(Gr) * (w @ Gr @ w)) < 0))
        out[b] = {
            "r_joint": pearson(C, jsim), "r_body": pearson(C, bsim),
            "r_joint_diffbody": pearson(C[~se], jsim[~se]),
            "sj_de": float(C[sj & ~se].mean()), "dj_se": float(C[~sj & se].mean()),
            "sj": float(C[sj].mean()), "dj": float(C[~sj].mean()),
            "neg_sj": float((C[sj] < 0).mean()), "neg_dj": float((C[~sj] < 0).mean()),
            "rob_r_body": pearson(rc, BODY_SIM[ru]),
            "rob_in": float(rc[same_r].mean()), "rob_out": float(rc[~same_r].mean()),
            "neg_in": float((rc[same_r] < 0).mean()), "neg_out": float((rc[~same_r] < 0).mean()),
            "neg_all": float((rc < 0).mean()),
            "opp_plain": opp(w_plain), "opp_round": opp(w_round),
        }
    return out


if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, "experiments/analysis/results/conflict/data/U2-URMA-s*/s*_conflict_at_*.npz"))):
        m = re.search(r"s(\d+)_conflict_at_(\d+)\.npz$", f)
        r = {"seed": int(m.group(1)), "upd": int(m.group(2)), **metrics(f)}
        rows.append(r)
    rows.sort(key=lambda r: (r["upd"], r["seed"]))
    json.dump(rows, open(sys.argv[1], "w"), indent=1)
    print(f"{'upd':>6s} s | DEC r_joint r_body r_jDiffBody  SJ&DE DJ&SE | CORE rob_r_body in   out  r_joint r_body | "
          f"EG negC in/out core  dec | opp plain/round core dec")
    for r in rows:
        d, c = r["dec"], r["core"]
        print(f"{r['upd']:6d} {r['seed']} |   {d['r_joint']:+.2f}   {d['r_body']:+.2f}   {d['r_joint_diffbody']:+.2f}     "
              f"{d['sj_de']:+.3f} {d['dj_se']:+.3f} |   {c['rob_r_body']:+.2f}  {c['rob_in']:+.2f} {c['rob_out']:+.2f}  "
              f"{c['r_joint']:+.2f}  {c['r_body']:+.2f} | {100*c['neg_in']:3.0f}/{100*c['neg_out']:3.0f} "
              f"{100*d['neg_in']:3.0f}/{100*d['neg_out']:3.0f} | {100*c['opp_plain']:3.0f}/{100*c['opp_round']:3.0f} "
              f"{100*d['opp_plain']:3.0f}/{100*d['opp_round']:3.0f}")
