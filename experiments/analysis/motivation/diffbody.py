"""Decoder joint-type x joint-type matrix restricted to pairs of robots from different EG groups (different bodies)."""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells_metrics import EG, GROUPS, POSITIONAL, REPO, TABLE, cos_from_gram, load_table, mean_gram  # noqa: E402

rows = []
for f in sorted(glob.glob(os.path.join(REPO, "experiments/analysis/results/conflict/data/U2-URMA-s*/s*_conflict_at_*.npz"))):
    m = re.search(r"s(\d+)_conflict_at_(\d+)\.npz$", f)
    upd, seed = int(m.group(2)), int(m.group(1))
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
    res = {"upd": upd, "seed": seed}
    for b in ("dec", "core", "enc"):
        C = cos_from_gram(K.T @ mean_gram(z, GROUPS.index(b)) @ K)
        iu = np.triu_indices(len(keys), 1)
        diffbody = eg[kr][iu[0]] != eg[kr][iu[1]]
        T = np.full((8, 8), np.nan)
        for g in range(8):
            for h in range(8):
                msk = diffbody & (((kg[iu[0]] == g) & (kg[iu[1]] == h)) | ((kg[iu[0]] == h) & (kg[iu[1]] == g)))
                if msk.any():
                    T[g, h] = C[iu][msk].mean()
        off = T[~np.eye(8, dtype=bool)]
        res[b] = {"T": T.tolist(), "diag": float(np.nanmean(np.diag(T))), "off": float(np.nanmean(off)),
                  "wins": int(sum(np.nanargmax(T[g]) == g for g in range(8) if np.isfinite(T[g, g])))}
    rows.append(res)
rows.sort(key=lambda r: (r["upd"], r["seed"]))
json.dump(rows, open(sys.argv[1], "w"))
print(" upd   s | decoder diag / off (wins) | core diag / off (wins) | enc diag / off (wins)   [different-body pairs only]")
for r in rows:
    print(f"{r['upd']:6d} {r['seed']} | " + " | ".join(f"{r[b]['diag']:+.2f} / {r[b]['off']:+.2f} ({r[b]['wins']}/8)" for b in ("dec", "core", "enc")))
