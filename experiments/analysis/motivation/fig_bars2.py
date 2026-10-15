"""Encoder and decoder: agreement of the updates each grouping puts together. Robot groups follow EG's appendix (2 and 4
groups, Ward clustering of the FGW body distance); joint types are the 8 groups of the revised table."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = os.path.dirname(os.path.abspath(__file__))
OUT, UPD, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = [x for x in json.load(open(os.path.join(SP, "bars_m24.json"))) if x["upd"] == UPD and x["seed"] == SEED][0]

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
BLUE, ORANGE, ORANGE_L, GRAY = "#2a78d6", "#eb6834", "#f3a37e", "#9a988f"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})

keys = ["none", "robot4", "joint"]
labels = ["No grouping", "Robot groups\n(4 groups)", "Joint types\n(8 types, ours)"]
colors = [GRAY, ORANGE, BLUE]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
for ax, (b, title) in zip(axes, [("enc", "Joint encoder"), ("dec", "Decoder")]):
    vals = [r[b][k][0] for k in keys]
    neg = [r[b][k][1] for k in keys]
    ax.bar(range(3), vals, width=0.62, color=colors, zorder=2)
    top = max(vals)
    for i, v in enumerate(vals):
        ax.text(i, v + top * 0.02, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=10.5)
    ax.text(2, vals[2] + top * 0.12, f"{vals[2] / max(vals[:2]):.1f}x", ha="center", color=BLUE, fontsize=12, weight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, top * 1.3)
    ax.set_title(title, color=INK, fontsize=11.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("agreement of updates placed together\n(mean gradient cosine)")
fig.suptitle("Grouping joints by type places agreeing updates together, in the encoder and in the decoder",
             color=INK, fontsize=12, x=0.01, ha="left")
fig.text(0.01, 0.01, f"Dense URMA policy, IQL, 70% Suboptimal Forward, seed {SEED}, {UPD // 1000}k updates. Bar: mean cosine "
         f"similarity of the actor gradients of two joints from different robots,\nover the pairs that a grouping puts in the "
         f"same group (no grouping: all pairs).\nRobot groups: "
         f"robots clustered by whole-body morphology into 4 groups; joint types: joints clustered by their kinematic and "
         f"dynamic properties.", color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.11, 1, 0.92))
fig.savefig(os.path.join(OUT, "motivation_bars.png"), dpi=170)
print("ok")
