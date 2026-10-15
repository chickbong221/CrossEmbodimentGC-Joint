"""Candidate heatmaps: gradient cosine between (robot, joint type) units of a layer, ordered by joint type vs by robot
group. Unit gradient = sum of the gradients of the robot's joints of that type (from the joint Gram matrix)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
SEED, UPD = int(sys.argv[1]), int(sys.argv[2])
REPO = "E:/Code/cross_embodiment_offline_rl"
T = json.load(open(os.path.join(REPO, "cross_embodiment_offline_rl/algorithms/morse/v8_joint_groups_rev_k8.json")))
JT8 = np.concatenate([T["groups"][n] for n in T["groups"]])
EG4 = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 3, 2, 3, 3, 1, 0])
z = np.load(os.path.join(SP, "views_out", f"cos_s{SEED}_{UPD}.npz"))
robot = z["robot"]
CMAP = LinearSegmentedColormap.from_list("div", ["#e34948", "#f3b3b2", "#f0efec", "#9ec5f4", "#1c5cab"])

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for row, b in enumerate(("enc_moe", "dec")):
    C = z[b].astype(np.float64)
    for col, how in enumerate(("type", "robot")):
        if how == "type":
            order = np.lexsort((robot, JT8))           # by joint type, then robot
            lab = JT8[order]
        else:
            order = np.lexsort((JT8, robot, EG4[robot]))  # by robot group, then robot, then type
            lab = EG4[robot][order]
        M = C[np.ix_(order, order)].copy()
        np.fill_diagonal(M, np.nan)
        ax = axes[row, col]
        vmax = np.nanpercentile(np.abs(M), 98)
        im = ax.imshow(M, cmap=CMAP, norm=SymLogNorm(linthresh=0.05, vmin=-vmax, vmax=vmax), interpolation="nearest")
        cuts = np.flatnonzero(np.diff(lab)) + 0.5
        start = 0
        for c in list(cuts) + [len(lab) - 0.5]:
            ax.add_patch(plt.Rectangle((start - 0.5 if start == 0 else start, start - 0.5 if start == 0 else start),
                                       c - (start - 0.5 if start == 0 else start), c - (start - 0.5 if start == 0 else start),
                                       fill=False, ec="black", lw=1.2))
            start = c
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{b}: ordered by {'joint type' if how == 'type' else 'robot group (EG)'}")
        # block contrast: mean within diagonal blocks vs outside
        same = lab[:, None] == lab[None, :]
        off = ~np.eye(len(lab), dtype=bool)
        ax.set_xlabel(f"inside blocks {np.nanmean(M[same & off]):+.3f}  outside {np.nanmean(M[~same]):+.3f}")
fig.tight_layout()
fig.savefig(os.path.join(SP, f"heat_try_s{SEED}_{UPD}.png"), dpi=90)
print("ok")
