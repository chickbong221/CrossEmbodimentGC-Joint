"""Motivation figures: interference between joints of different robots in the joint-level layers.

Measure. For every joint of every robot (241 joints, 16 robots): the gradient of its term of the IQL actor loss with
respect to a set of weights, averaged over 32 fixed minibatches (identical for every model). For two joints of
different robots, the cosine of their gradients; a pair conflicts strongly when the cosine is below -0.1 (a step for
one joint undoes part of the other's step). Reported: the share of strongly conflicting cross-robot pairs.

Weights: dense policy = the decoder (or the encoder hidden layer); MORSE = the routed experts of the same layer.

python fig_motivation.py [--ckpt 100000] [--out ../results/conflict/figs]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ["Roll", "Thigh", "Knee", "Pitch\n(large)", "Pitch\n(servo)", "Yaw\n(fast)", "Yaw\n(slow)", "Roll\n(servo)"]
TAU = -0.1
DENSE_C, EG_C, MORSE_C = "#8c8c8c", "#5b8fc7", "#d1495b"


def load():
    """Cosine matrices of every model and checkpoint (extract_cos.py on H100), MORSE's final checkpoint as 100000 and
    EG's final checkpoints as eg<seed>."""
    z = dict(np.load(os.path.join(HERE, "cos_small.npz"), allow_pickle=True))
    for name in ("cos_morse_final.npz", "cos_eg.npz"):
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            extra = np.load(path, allow_pickle=True)
            assert (extra["unit_robot"] == z["unit_robot"]).all() and (extra["unit_joint"] == z["unit_joint"]).all()
            z.update({k.replace("morse_final__", "morse_100000__"): extra[k] for k in extra.files if "__" in k})
    g = json.load(open(os.path.join(HERE, "v8_joint_groups_rev_k8.json")))
    names = [str(x) for x in z["robot_names"]]
    robot, joint = z["unit_robot"].astype(int), z["unit_joint"].astype(int)
    jtype = np.array([g["groups"][names[r]][j] for r, j in zip(robot, joint)])
    return z, robot, jtype


def cross_pairs(robot):
    iu = np.triu_indices(len(robot), 1)
    m = robot[iu[0]] != robot[iu[1]]
    return iu[0][m], iu[1][m]


def strong_share(C, a, b):
    return 100.0 * (C[a, b] < TAU).mean()


def type_matrix(C, robot, jtype):
    X = robot[:, None] != robot[None, :]
    H = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            M = X & (jtype[:, None] == i) & (jtype[None, :] == j)
            H[i, j] = 100.0 * (C[M] < TAU).mean()
    return H


def heatmaps(z, robot, jtype, ckpt, layer, out):
    dense_key, morse_key = {"dec": ("dec", "dec_routed"), "enc": ("enc_moe", "enc_routed")}[layer]
    a, b = cross_pairs(robot)
    mats, totals = [], []
    for key in (f"dense_{ckpt}__{dense_key}__cos", f"morse_{ckpt}__{morse_key}__cos"):
        C = z[key]
        mats.append(type_matrix(C, robot, jtype))
        totals.append(strong_share(C, a, b))
    vmax = 45.0
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), constrained_layout=True)
    titles = ["Dense policy", "MORSE"]
    for ax, H, title, tot in zip(axes, mats, titles, totals):
        im = ax.imshow(H, cmap="Reds", vmin=0, vmax=vmax)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f"{H[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if H[i, j] > 0.6 * vmax else "black")
        ax.set_xticks(range(8), TYPES, fontsize=8)
        ax.set_yticks(range(8), TYPES, fontsize=8)
        ax.set_title(f"{title}: {tot:.1f}% of pairs", fontsize=11)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("strongly conflicting joint pairs (%)", fontsize=9)
    path = os.path.join(out, f"motivation_heatmap_{layer}.png")
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"{layer}: dense {totals[0]:.1f}%  MORSE {totals[1]:.1f}%  ->", path)
    return mats, totals


def routing_map(z, jtype, ckpt, layer, out):
    """Joint type x routed expert: how often the router selects each routed expert for joints of each type."""
    S = z[f"morse_{ckpt}__route_sel_{layer}_moe"]
    T = 100.0 * np.array([S[jtype == t].mean(0) for t in range(8)])
    order = np.lexsort((-T.max(0), T.argmax(0)))          # experts sorted by the type that uses them most
    T = T[:, order]
    O = S @ S.T / 3.0                                      # expected share of routed experts two joints have in common
    same = jtype[:, None] == jtype[None, :]
    off = ~np.eye(len(jtype), dtype=bool)
    fig, ax = plt.subplots(figsize=(7.4, 3.9), constrained_layout=True)
    im = ax.imshow(T, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i, j] >= 30:
                ax.text(j, i, f"{T[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if T[i, j] > 60 else "black")
    ax.set_yticks(range(8), [t.replace("\n", " ") for t in TYPES], fontsize=8.5)
    ax.set_xticks(range(T.shape[1]), [str(k + 1) for k in order], fontsize=8)
    ax.set_xlabel("routed expert", fontsize=9)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("selected for joints of the type (%)", fontsize=8.5)
    path = os.path.join(out, f"motivation_routing_{layer}.png")
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"routing {layer}: shared routed experts, same type {O[same & off].mean():.2f}, "
          f"different type {O[~same].mean():.2f} ->", path)


def threshold_curve(z, robot, ckpt, out):
    """Share of cross-robot pairs with cosine below tau, for every tau."""
    a, b = cross_pairs(robot)
    taus = np.linspace(-0.5, 0.0, 101)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), constrained_layout=True, sharey=True)
    for ax, (layer, dk, mk) in zip(axes, (("Decoder", "dec", "dec_routed"), ("Encoder", "enc_moe", "enc_routed"))):
        for key, label, color in ((f"dense_{ckpt}__{dk}__cos", "Dense policy", DENSE_C),
                                  (f"morse_{ckpt}__{mk}__cos", "MORSE", MORSE_C)):
            c = z[key][a, b]
            ax.plot(taus, [100.0 * (c < t).mean() for t in taus], color=color, lw=2, label=label)
        ax.axvline(TAU, color="k", lw=0.8, ls=":")
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel(r"cosine threshold $\tau$", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"pairs with cosine $<\tau$ (%)", fontsize=9)
    axes[0].legend(fontsize=8.5, frameon=False)
    path = os.path.join(out, "motivation_threshold.png")
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print("threshold curve ->", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="100000")
    p.add_argument("--out", default=os.path.join(HERE, "..", "results", "conflict", "figs"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    z, robot, jtype = load()
    for layer in ("dec", "enc"):
        heatmaps(z, robot, jtype, args.ckpt, layer, args.out)
    routing_map(z, jtype, args.ckpt, "enc", args.out)
    routing_map(z, jtype, args.ckpt, "dec", args.out)
    threshold_curve(z, robot, args.ckpt, args.out)


if __name__ == "__main__":
    main()
