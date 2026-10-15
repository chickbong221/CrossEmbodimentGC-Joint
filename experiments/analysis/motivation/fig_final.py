"""Motivation figure: (a, b) within-expert gradient agreement relative to the shared layer, on the two layers MORSE
replaces with experts; (c) decoder gradient similarity between joint types (colour only), as in EG / Gradient Vaccine."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
SEED, UPD, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
REPO = "E:/Code/cross_embodiment_offline_rl"
r = [x for x in json.load(open(os.path.join(SP, "views_out", "views.json"))) if x["seed"] == SEED and x["upd"] == UPD][0]
z = np.load(os.path.join(SP, "views_out", f"cos_s{SEED}_{UPD}.npz"))
T = json.load(open(os.path.join(REPO, "cross_embodiment_offline_rl/algorithms/morse/v8_joint_groups_rev_k8.json")))
JT = np.concatenate([T["groups"][k] for k in T["groups"]])
robot = z["robot"]

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a988f"
CMAP = LinearSegmentedColormap.from_list("div", ["#e34948", "#f3b3b2", "#f0efec", "#9ec5f4", "#1c5cab"])
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(14.5, 4.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.35)
labels = ["Shared\nlayer", "Robot-group\nexperts (4)", "Joint-type\nexperts (8, ours)"]
for k, (b, title) in enumerate([("enc_moe", "(a) Joint-encoder expert layer"), ("dec", "(b) Decoder")]):
    ax = fig.add_subplot(gs[k])
    base = r[b]["shared"]["pair_mean_per_expert"]
    vals = [1.0, r[b]["robot4"]["pair_mean_per_expert"] / base, r[b]["joint8"]["pair_mean_per_expert"] / base]
    ax.bar(range(3), vals, width=0.62, color=[GRAY, ORANGE, BLUE], zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.1f}x", ha="center", va="bottom", color=INK, fontsize=12,
                weight="bold" if i == 2 else "normal")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if k == 0:
        ax.set_ylabel("gradient agreement within an expert\n(relative to the shared layer)")

# (c) decoder: joint type x joint type, pairs of joints of different robots
C = z["dec"].astype(np.float64)
M = np.full((8, 8), np.nan)
for g in range(8):
    for h in range(8):
        A, B = np.flatnonzero(JT == g), np.flatnonzero(JT == h)
        S = C[np.ix_(A, B)]
        M[g, h] = S[robot[A][:, None] != robot[B][None, :]].mean()
names = ["Roll", "Thigh", "Knee", "Pitch (large)", "Pitch (servo)", "Yaw (fast)", "Yaw (slow)", "Roll (servo)"]
ax = fig.add_subplot(gs[2])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
im = ax.imshow(M, cmap=SEQ, vmin=np.nanmin(M), vmax=np.nanmax(M))
for i in range(8):
    ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.5))
ax.set_xticks(range(8)); ax.set_yticks(range(8))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(names, fontsize=9)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_title("(c) Decoder: gradient similarity\n     between joint types", color=INK, fontsize=11, loc="left")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("gradient cosine similarity", color=INK2)
cb.outline.set_visible(False)
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.22)
fig.savefig(os.path.join(OUT, "motivation_final.png"), dpi=170)
print("ok", [round(v, 2) for v in M.diagonal()])
