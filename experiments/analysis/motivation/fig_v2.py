"""Motivation figures in the style of EG (ICLR'26, Figs. 2-3), one checkpoint: dense URMAv2, seed 1, 50k updates."""
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
OUT = sys.argv[1]
UPD, SEED = int(sys.argv[2]), int(sys.argv[3])
os.makedirs(OUT, exist_ok=True)
pick = lambda rows: [r for r in rows if r["upd"] == UPD and r["seed"] == SEED][0]
G = pick(json.load(open(os.path.join(SP, "group_metrics.json"))))
DB = pick(json.load(open(os.path.join(SP, "diffbody.json"))))
CM = pick(json.load(open(os.path.join(SP, "cells_metrics.json"))))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a988f"
CMAP = LinearSegmentedColormap.from_list("div", ["#e34948", "#f3b3b2", "#f0efec", "#9ec5f4", "#1c5cab"])  # red (conflict) - gray - blue (agree)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "font.family": "DejaVu Sans"})
POS = ["a1", "go1", "go2", "anyb", "anyc", "bkv0", "bkvb", "bg", "bit", "h1", "g1", "tl", "op3", "nao5", "cas", "hx"]
LONG = {"a1": "A1", "go1": "Go1", "go2": "Go2", "anyb": "ANYmal B", "anyc": "ANYmal C", "bkv0": "Barkour v0",
        "bkvb": "Barkour vB", "bg": "Badger", "bit": "Bittle", "h1": "H1", "g1": "G1", "tl": "Talos", "op3": "OP3",
        "nao5": "NAO", "cas": "Cassie", "hx": "Hexapod"}
EG_ORDER = [["a1", "go1", "go2", "bkv0", "bkvb", "bg", "bit", "hx"], ["anyb", "anyc", "cas"], ["h1", "tl"], ["g1", "op3", "nao5"]]
JT = ["1 Roll", "2 Thigh", "3 Knee", "4 Pitch, large", "5 Pitch, servo", "6 Yaw, fast", "7 Yaw, slow", "8 Roll, servo"]
TAG = f"seed {SEED}, {UPD // 1000}k updates"


def heat(ax, M, labels, vmax, annotate=False, title=""):
    im = ax.imshow(M, cmap=CMAP, norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if abs(M[i, j]) > 0.85 * vmax else INK)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    return im


# ======================= Figure 1: main motivation figure =======================
fig = plt.figure(figsize=(15, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.05, 0.8], wspace=0.42)

# (a) core: robot x robot, ordered by EG groups
order = [r for grp in EG_ORDER for r in grp]
idx = [POS.index(r) for r in order]
R = np.array(G["core"]["robot_cos"])[np.ix_(idx, idx)]
np.fill_diagonal(R, np.nan)
ax = fig.add_subplot(gs[0])
im = heat(ax, R, [LONG[r] for r in order], vmax=0.6,
          title=f"(a) Core: robots with similar bodies\n     update the shared weights alike")
start = 0
for grp in EG_ORDER:
    ax.add_patch(Rectangle((start - 0.5, start - 0.5), len(grp), len(grp), fill=False, ec=INK, lw=1.6))
    start += len(grp)
c = CM["core"]
ax.set_xlabel(f"mean cosine inside EG groups {c['rob_in']:+.2f}  vs  across {c['rob_out']:+.2f}\n"
              f"(boxes = EG's 4 robot groups)", color=INK2, fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("gradient cosine similarity", color=INK2)
cb.outline.set_visible(False)

# (b) decoder: joint type x joint type, pairs of different robots
T = np.array(G["dec"]["T"])
ax = fig.add_subplot(gs[1])
im = heat(ax, T, JT, vmax=0.35, annotate=True,
          title="(b) Decoder: joints of the same type update\n     alike, even in different robots")
for i in range(8):
    ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
d = G["dec"]
ax.set_xlabel(f"same type (diagonal) {d['diag']:+.2f}  vs  different types {d['off']:+.2f}\n"
              f"{d['diag_wins']} of 8 joint types agree most with their own type", color=INK2, fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("gradient cosine similarity", color=INK2)
cb.outline.set_visible(False)

# (c) EG's groups and conflict (EG's own metric: fraction of C < 0)
ax = fig.add_subplot(gs[2])
x = np.arange(2)
inn = [100 * CM[b]["neg_in"] for b in ("core", "dec")]
out = [100 * CM[b]["neg_out"] for b in ("core", "dec")]
ax.bar(x - 0.19, inn, 0.36, color=ORANGE, label="robot pairs inside an EG group", zorder=2)
ax.bar(x + 0.19, out, 0.36, color=GRAY, label="robot pairs across EG groups", zorder=2)
for i in range(2):
    ax.text(x[i] - 0.19, inn[i] + 0.8, f"{inn[i]:.0f}%", ha="center", color=INK)
    ax.text(x[i] + 0.19, out[i] + 0.8, f"{out[i]:.0f}%", ha="center", color=INK)
ax.set_xticks(x)
ax.set_xticklabels(["Core", "Decoder"])
ax.set_ylim(0, max(inn + out) * 1.45)
ax.set_ylabel("conflicting robot pairs, C < 0 (%)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
ax.set_title("(c) EG's groups do not separate\n     the conflicting robots", color=INK, fontsize=10.5, loc="left")

fig.text(0.01, 0.01, f"Dense URMA policy, IQL, 70% Suboptimal Forward ({TAG}). Cosine similarity C of actor gradients as in EG "
         f"(Eq. 2), each averaged over 32 minibatches.\n(a) one gradient per robot; (b) one gradient per robot and joint type, "
         f"averaged over all pairs from different robots; (c) robot pairs, EG's metric.", color=INK2, fontsize=8)
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.27)
fig.savefig(os.path.join(OUT, "motivation_main.png"), dpi=170)
plt.close(fig)

# ======================= Figure 2: control - is the joint-type effect only a body effect? =======================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
Td = np.array(DB["dec"]["T"])
im = heat(ax, Td, JT, vmax=0.35, annotate=True,
          title="(a) Decoder, only robots from different EG groups")
for i in range(8):
    ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("gradient cosine similarity", color=INK2)
cb.outline.set_visible(False)
ax = axes[1]
vals = [(DB["dec"]["diag"], DB["dec"]["off"]), (DB["core"]["diag"], DB["core"]["off"])]
x = np.arange(2)
ax.bar(x - 0.19, [v[0] for v in vals], 0.36, color=BLUE, label="same joint type", zorder=2)
ax.bar(x + 0.19, [v[1] for v in vals], 0.36, color=GRAY, label="different joint types", zorder=2)
for i, (a, b) in enumerate(vals):
    ax.text(x[i] - 0.19, a + 0.004, f"{a:+.2f}", ha="center", color=INK)
    ax.text(x[i] + 0.19, b + 0.004, f"{b:+.2f}", ha="center", color=INK)
    ax.text(x[i], max(a, b) + 0.035, f"{a / b:.1f}x" if b > 0 else "", ha="center", color=INK, fontsize=11, weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Decoder", "Core"])
ax.set_ylim(0, max(max(v) for v in vals) * 1.6)
ax.set_ylabel("mean gradient cosine similarity")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title("(b) Joint type matters in the decoder,\n     not in the core", color=INK, fontsize=10.5, loc="left")
fig.text(0.01, 0.01, f"Same checkpoint ({TAG}); only pairs of robots that EG puts in different groups, so the agreement "
         f"cannot come from similar bodies.", color=INK2, fontsize=8)
fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.3, wspace=0.35)
fig.savefig(os.path.join(OUT, "motivation_control.png"), dpi=170)
plt.close(fig)
print("ok")
