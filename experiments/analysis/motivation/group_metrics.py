"""Per-checkpoint metrics on (robot, joint-group) gradients: the gradient of all joints of one type in one robot
(EG's per-embodiment gradient, split by joint type). Plain cosine of gradients averaged over the 32 minibatches."""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells_metrics import BODY_SIM, EG, GROUPS, POSITIONAL, REPO, TABLE, ZR, cos_from_gram, load_table, mean_gram, membership, pearson  # noqa: E402


def analyse(f):
    z = np.load(f, allow_pickle=True)
    names = [str(n) for n in z["robot_names"]]
    eg = np.array([EG.get(n, EG[POSITIONAL[i]]) for i, n in enumerate(names)])
    robot, joint = z["unit_robot"].astype(int), z["unit_joint"].astype(int)
    tab, _ = load_table(TABLE, names)
    jg = np.array([tab[r][j] for r, j in zip(robot, joint)])
    keys = sorted({(int(r), int(g)) for r, g in zip(robot, jg)})
    K = np.zeros((len(robot), len(keys)))
    for k, (r, g) in enumerate(keys):
        K[(robot == r) & (jg == g), k] = 1.0
    kr = np.array([k[0] for k in keys]); kg = np.array([k[1] for k in keys])
    feat = np.stack([ZR[(robot == r) & (jg == g)].mean(0) for r, g in keys])
    fd = np.sqrt(((feat[:, None] - feat[None]) ** 2).sum(-1))
    iu = np.triu_indices(len(keys), 1)
    cross = kr[iu[0]] != kr[iu[1]]
    jsim = (1 - (fd - fd[iu].min()) / (fd[iu].max() - fd[iu].min()))[iu][cross]
    bsim = BODY_SIM[kr[iu[0]], kr[iu[1]]][cross]
    sj = (kg[iu[0]] == kg[iu[1]])[cross]
    se = (eg[kr][iu[0]] == eg[kr][iu[1]])[cross]
    out = {"keys": keys}
    for b in ("enc", "core", "dec"):
        G = K.T @ mean_gram(z, GROUPS.index(b)) @ K
        Cfull = cos_from_gram(G)
        C = Cfull[iu][cross]
        # joint-type x joint-type matrix over pairs of different robots
        T = np.full((8, 8), np.nan)
        for g in range(8):
            for h in range(8):
                m = (kg[iu[0]] == g) & (kg[iu[1]] == h) | (kg[iu[0]] == h) & (kg[iu[1]] == g)
                m = m & (kr[iu[0]] != kr[iu[1]])
                if m.any():
                    T[g, h] = Cfull[iu][m].mean()
        Mr = membership(robot, 16)
        Cr = cos_from_gram(Mr.T @ mean_gram(z, GROUPS.index(b)) @ Mr)
        out[b] = {"r_joint": pearson(C, jsim), "r_body": pearson(C, bsim),
                  "sj_de": float(C[sj & ~se].mean()), "dj_se": float(C[~sj & se].mean()),
                  "diag": float(np.nanmean(np.diag(T))), "off": float(np.nanmean(T[~np.eye(8, dtype=bool)])),
                  "diag_wins": int(sum(np.nanargmax(T[g]) == g for g in range(8))),
                  "T": T.tolist(), "robot_cos": Cr.tolist()}
    return out


if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, "experiments/analysis/results/conflict/data/U2-URMA-s*/s*_conflict_at_*.npz"))):
        m = re.search(r"s(\d+)_conflict_at_(\d+)\.npz$", f)
        rows.append({"seed": int(m.group(1)), "upd": int(m.group(2)), **analyse(f)})
    rows.sort(key=lambda r: (r["upd"], r["seed"]))
    json.dump(rows, open(sys.argv[1], "w"))
    print(f"{'upd':>6s} s | DEC r_joint r_body  SJ&DE  DJ&SE  diag  off  diagwins | CORE r_joint r_body SJ&DE DJ&SE diag off wins")
    for r in rows:
        d, c = r["dec"], r["core"]
        print(f"{r['upd']:6d} {r['seed']} |   {d['r_joint']:+.2f}  {d['r_body']:+.2f}  {d['sj_de']:+.3f} {d['dj_se']:+.3f} "
              f"{d['diag']:+.2f} {d['off']:+.2f}   {d['diag_wins']}/8 |   {c['r_joint']:+.2f}  {c['r_body']:+.2f} "
              f"{c['sj_de']:+.3f} {c['dj_se']:+.3f} {c['diag']:+.2f} {c['off']:+.2f} {c['diag_wins']}/8")
