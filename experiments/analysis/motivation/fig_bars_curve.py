"""Bar chart (dense / EG / MORSE at the end of training) and conflict over training (dense vs MORSE).
Same measure as fig_motivation.py: share of cross-robot joint pairs whose gradient cosine is below -0.1.
python fig_bars_curve.py [--eg-seed 0]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fig_motivation import DENSE_C, EG_C, MORSE_C, HERE, TAU, cross_pairs, load

STEPS = (25000, 50000, 75000, 100000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eg-seed", default="0")
    p.add_argument("--out", default=os.path.join(HERE, "..", "results", "conflict", "figs"))
    args = p.parse_args()
    z, robot, _ = load()
    a, b = cross_pairs(robot)

    def share(key):
        return 100.0 * (z[key][a, b] < TAU).mean()

    # ---- bar chart: end of training (EG's encoder has no hidden layer, so no encoder bar for EG) ----
    dec = {"Dense policy": share("dense_100000__dec__cos"), "EG": share(f"eg{args.eg_seed}__dec_exp__cos"),
           "MORSE": share("morse_100000__dec_routed__cos")}
    enc = {"Dense policy": share("dense_100000__enc_moe__cos"), "MORSE": share("morse_100000__enc_routed__cos")}
    colors = {"Dense policy": DENSE_C, "EG": EG_C, "MORSE": MORSE_C}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), constrained_layout=True, sharey=True,
                             gridspec_kw={"width_ratios": [3, 2]})
    for ax, vals, title in ((axes[0], dec, "Decoder"), (axes[1], enc, "Encoder")):
        names = list(vals)
        bars = ax.bar(range(len(names)), [vals[n] for n in names], color=[colors[n] for n in names], width=0.62)
        for bar, n in zip(bars, names):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{vals[n]:.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(names)), names, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("strongly conflicting joint pairs (%)", fontsize=9)
    axes[0].set_ylim(0, 1.18 * max(max(dec.values()), max(enc.values())))
    path = os.path.join(args.out, "motivation_bars.png")
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print("bars", {k: round(v, 1) for k, v in dec.items()}, {k: round(v, 1) for k, v in enc.items()}, "->", path)
    for s in "012":
        if f"eg{s}__dec_exp__cos" in z:
            print(f"  EG seed {s} decoder: {share(f'eg{s}__dec_exp__cos'):.1f}%")

    # ---- conflict over training ----
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True, sharey=True)
    for ax, (title, dk, mk) in zip(axes, (("Decoder", "dec", "dec_routed"), ("Encoder", "enc_moe", "enc_routed"))):
        d = [share(f"dense_{s}__{dk}__cos") for s in STEPS]
        m = [share(f"morse_{s}__{mk}__cos") for s in STEPS]
        x = np.array(STEPS) / 1000
        ax.plot(x, d, "o-", color=DENSE_C, lw=2, label="Dense policy")
        ax.plot(x, m, "o-", color=MORSE_C, lw=2, label="MORSE")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, [f"{int(v)}k" for v in x])
        ax.set_xlabel("training updates", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 40)
        print(title, "dense", np.round(d, 1), "morse", np.round(m, 1))
    axes[0].set_ylabel("strongly conflicting\njoint pairs (%)", fontsize=9)
    axes[0].legend(fontsize=8.5, frameon=False, loc="center right")
    path = os.path.join(args.out, "motivation_training.png")
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print("training ->", path)


if __name__ == "__main__":
    main()
