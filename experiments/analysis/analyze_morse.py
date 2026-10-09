"""Figures and tables for MORSE from finished runs.

Each run directory is what RL-X writes (``runs/<project>/<exp_name>/<run_name>``):
TensorBoard event files and, for MORSE runs, ``morse_diagnostics/step_*.npz``.
Several seeds of one method are given as a comma-separated list.

Example (from ``experiments/``):
  python analysis/analyze_morse.py \
      --run "MORSE=runs/cross-embodiment-offline/MORSE-IQL/*" \
      --run "MoE, no conflict=runs/cross-embodiment-offline/MORSE-IQL-moe_no_conflict/*" \
      --run "URMA=runs/cross-embodiment-offline/MORSE-IQL-urma/*" \
      --baseline URMA --out analysis_out

``--upto N`` uses only what was logged up to N policy updates (evaluations, diagnostics, snapshots), so that
runs still in training can be compared with finished ones at the same number of updates.

Outputs (``--out``):
  curves_<metric>.pdf        interference and routing diagnostics during training, per block
  pair_categories.pdf        conflict frequency within a transition, within a robot, across robots
  robot_cos_<run>_<block>.pdf        robot-level gradient cosine matrix (Abe et al. Fig. 3b, per block),
                                     from full-batch per-robot gradients accumulated over updates
  morphology_vs_conflict.csv/.pdf    does morphological similarity explain the conflict structure?
  persistence.csv            are conflict patterns stable across checkpoints (structure, not noise)?
  usage_<run>_<block>.pdf    expert usage per robot
  factors_<run>_<block>.pdf  input overlap vs demand agreement of conflicting pairs (Eq. linear_factorization)
  type_cos_<run>_<block>_e<k>.pdf    type-level alignment accumulated over updates (type-evidence runs)
  joint_pairs.csv, joint_grouping.csv / joint_grouping_<run>_<block>.pdf, joint_cos_<run>_<block>.pdf
                             motivation: are conflicts organized by joints or by robots? (see joint_motivation)
  interaction.csv            harmful and helpful interaction between unit types relative to their self terms:
                             is sharing less conflicting, or only smaller?
  returns.csv / returns.tex  per-robot returns, class averages, worst robot, gains over the baseline
  conflict_vs_gain_<run>.pdf does conflict reduction predict return gains per robot?
  diagnostics_summary.csv    final interference and routing diagnostics per run and block (appendix table)
  cost.csv                   steady-state throughput and update time
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

BLOCKS = ("joint_encoder", "foot_encoder", "core", "decoder")
ROBOTS = ("unitree_a1", "unitree_go1", "unitree_go2", "anymal_b", "anymal_c", "barkour_v0", "barkour_vb",
          "badger", "bittle", "unitree_h1", "unitree_g1", "talos", "robotis_op3", "nao_v5", "cassie", "hexapod")
CLASSES = {
    "quadruped": ("unitree_a1", "unitree_go1", "unitree_go2", "anymal_b", "anymal_c", "barkour_v0",
                  "barkour_vb", "badger", "bittle"),
    "biped": ("unitree_h1", "unitree_g1", "talos", "robotis_op3", "nao_v5", "cassie"),
    "hexapod": ("hexapod",),
}
# The environments report robots by SHORT_NAME (environments/<robot>/environment.py): these codes
# appear in the evaluation tags (eval/<metric>/<code>) and in the snapshots.
SHORT_NAMES = {
    "unitree_a1": "a1", "unitree_go1": "go1", "unitree_go2": "go2", "anymal_b": "anyb", "anymal_c": "anyc",
    "barkour_v0": "bkv0", "barkour_vb": "bkvb", "badger": "bg", "bittle": "bit", "unitree_h1": "h1",
    "unitree_g1": "g1", "talos": "tl", "robotis_op3": "op3", "nao_v5": "nao5", "cassie": "cas", "hexapod": "hx",
}
FULL_NAMES = {code: name for name, code in SHORT_NAMES.items()}


def canonical(name):
    """Configuration name of a robot given either its configuration name or its short code."""
    name = str(name)
    return FULL_NAMES.get(name, name)
# Unit-level cancelled_fraction / destructive_share are not plotted: with thousands of per-sample
# units, independent noise alone drives them to ~1 and ~0.5 (they stay in diagnostics_summary.csv).
CURVE_METRICS = ("robot_neg_frac", "robot_neg_frac_acc", "robot_cancelled_fraction_acc", "robot_cos_mean_acc",
                 "conflict_mass", "conf_coroute", "router_entropy", "q_pos_frac", "type_conflict_frac")
LEVELS = BLOCKS + ("actor",)          # "actor": the whole policy (per-embodiment gradient cosine of Abe et al.)


# ----------------------------------------------------------------------------- loading
def expand_dirs(spec):
    dirs = []
    for part in spec.split(","):
        dirs.extend(sorted(glob.glob(part.strip())) or [part.strip()])
    return [d for d in dirs if os.path.isdir(d)]


def load_tb(run_dir, upto=None):
    """Scalars as {tag: (steps, values)}; with ``upto``, only those logged up to that many policy updates.
    A folder without event files but with ``tb_scalars.npz`` (analysis/tb_cache.py) is read from that cache."""
    cache = os.path.join(run_dir, "tb_scalars.npz")
    out = {}
    if not glob.glob(os.path.join(run_dir, "events.out.tfevents.*")) and os.path.exists(cache):
        z = np.load(cache)
        for key in z.files:
            if key.startswith("s::"):
                tag = key[3:]
                out[tag] = (z[key], z["v::" + tag])
    else:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            events = acc.Scalars(tag)
            out[tag] = (np.array([e.step for e in events]), np.array([e.value for e in events], dtype=np.float64))
    if upto is None:
        return out
    if "steps/nr_updates" not in out:
        print(f"warning: {run_dir} has no steps/nr_updates, --upto is not applied")
        return out
    # TensorBoard steps are training samples; this run's steps/nr_updates maps them to updates (one update of
    # slack for the order in which an update and its evaluation are logged).
    s, u = out["steps/nr_updates"]
    order = np.argsort(s)
    s, u = s[order], u[order]
    kept = {}
    for tag, (steps, values) in out.items():
        keep = np.interp(steps, s, u) <= upto + 1
        kept[tag] = (steps[keep], values[keep])
    return kept


def load_snapshots(run_dir, upto=None):
    files = sorted(glob.glob(os.path.join(run_dir, "morse_diagnostics", "step_*.npz")))
    if upto is not None:
        files = [f for f in files if int(os.path.basename(f)[len("step_"):-len(".npz")]) <= upto]
    return [dict(np.load(f, allow_pickle=False)) for f in files]


def last_eval_update(tb, metric):
    """Policy updates at the last logged evaluation of a run (NaN without evaluations)."""
    tag = f"eval/{metric}/average"
    if tag not in tb or not len(tb[tag][0]):
        return np.nan
    step = tb[tag][0][-1]
    if "steps/nr_updates" not in tb:
        return float(step)
    s, u = tb["steps/nr_updates"]
    order = np.argsort(s)
    return float(np.interp(step, s[order], u[order]))


def aggregate_curve(tbs, tag):
    """Mean and standard error across seeds, on the steps shared by all seeds."""
    series = [tb[tag] for tb in tbs if tag in tb]
    if not series:
        return None
    steps = series[0][0]
    for s, _ in series[1:]:
        steps = np.intersect1d(steps, s)
    values = np.stack([np.interp(steps, s, v) for s, v in series])
    se = values.std(axis=0, ddof=1) / np.sqrt(len(series)) if len(series) > 1 else np.zeros(len(steps))
    return steps, values.mean(axis=0), se


def final_value(tb, tag, last):
    if tag not in tb:
        return np.nan
    return float(np.mean(tb[tag][1][-last:]))


def nanmean(values):
    """Mean of the finite values (NaN if there are none, without a warning)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan


def col_nanmean(a):
    """Column-wise nanmean of a (seeds, robots) array, without warnings for empty columns."""
    a = np.asarray(a, dtype=float)
    return np.array([nanmean(a[:, j]) for j in range(a.shape[1])])


def eval_value(tb, metric, robot, last):
    """Final evaluation value of a robot; the tag uses the robot's short code."""
    value = final_value(tb, f"eval/{metric}/{SHORT_NAMES.get(robot, robot)}", last)
    return value if np.isfinite(value) else final_value(tb, f"eval/{metric}/{robot}", last)


# ---------------------------------------------------------------------------- figures
def to_updates(tbs, steps):
    """Convert TensorBoard steps (training samples) to policy updates using steps/nr_updates."""
    for tb in tbs:
        if "steps/nr_updates" in tb:
            s, u = tb["steps/nr_updates"]
            return np.interp(steps, s, u)
    return steps


def plot_curves(runs, out):
    for metric in CURVE_METRICS:
        levels = [lv for lv in LEVELS
                  if any(f"morse/{lv}/{metric}" in tb for run in runs.values() for tb in run["tb"])]
        if not levels:
            continue
        fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 3), sharex=True, squeeze=False)
        axes = axes[0]
        for ax, level in zip(axes, levels):
            for label, run in runs.items():
                curve = aggregate_curve(run["tb"], f"morse/{level}/{metric}")
                if curve is None:
                    continue
                steps, mean, se = curve
                x = to_updates(run["tb"], steps)
                ax.plot(x, mean, label=label)
                ax.fill_between(x, mean - se, mean + se, alpha=0.2)
            ax.set_title(level)
            ax.set_xlabel("policy updates")
        axes[0].set_ylabel(metric)
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"curves_{metric}.pdf"))
        plt.close(fig)


def plot_pair_categories(runs, out, last):
    cats = ("within_transition", "within_robot", "cross_robot")
    fig, axes = plt.subplots(1, len(BLOCKS), figsize=(4 * len(BLOCKS), 3), sharey=True)
    width = 0.8 / max(len(runs), 1)
    for ax, block in zip(axes, BLOCKS):
        for i, (label, run) in enumerate(runs.items()):
            vals = [nanmean([final_value(tb, f"morse/{block}/conflict_frac_{c}", last) for tb in run["tb"]]) for c in cats]
            ax.bar(np.arange(len(cats)) + i * width, vals, width, label=label)
        ax.set_xticks(np.arange(len(cats)) + width * (len(runs) - 1) / 2)
        ax.set_xticklabels(["within\ntransition", "within\nrobot", "across\nrobots"], fontsize=8)
        ax.set_title(block)
    axes[0].set_ylabel("fraction of conflicting pairs")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pair_categories.pdf"))
    plt.close(fig)


def robot_cos_key(run, block):
    """Snapshot key of the robot-level cosine: accumulated over updates if available."""
    acc = f"{block}__robot_cos_acc"
    return acc if any(snaps and acc in snaps[-1] for snaps in run["snaps"]) else f"{block}__robot_cos"


def final_snapshot_mean(run, key):
    """Average an array from the last snapshot of every seed."""
    arrays = [snaps[-1][key] for snaps in run["snaps"] if snaps and key in snaps[-1]]
    return np.mean(arrays, axis=0) if arrays else None


def robot_names(run):
    """Configuration names of the robots, in the order of the snapshot matrices."""
    for snaps in run["snaps"]:
        if snaps:
            return [canonical(r) for r in snaps[-1]["robot_names"]]
    return list(ROBOTS)


def plot_robot_matrices(runs, out):
    for label, run in runs.items():
        names = robot_names(run)
        for block in BLOCKS:
            m = final_snapshot_mean(run, robot_cos_key(run, block))
            if m is None:
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(m, cmap="RdBu", vmin=-1, vmax=1)
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=6)
            ax.set_yticklabels(names, fontsize=6)
            ax.set_title(f"{label}: robot-level gradient cosine ({block})", fontsize=8)
            fig.colorbar(im)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"robot_cos_{slug(label)}_{block}.pdf"))
            plt.close(fig)


def morphology_similarity(distance_path, names):
    d = np.load(distance_path)
    order = [ROBOTS.index(n) for n in names]
    d = d[np.ix_(order, order)]
    off = ~np.eye(len(names), dtype=bool)
    lo, hi = d[off].min(), d[off].max()
    return 1.0 - (d - lo) / (hi - lo + 1e-12)


def morphology_vs_conflict(runs, out, distance_path):
    if not os.path.exists(distance_path):
        print(f"skip morphology analysis: {distance_path} not found")
        return
    rows = []
    fig, axes = plt.subplots(1, len(BLOCKS), figsize=(4 * len(BLOCKS), 3), sharey=True)
    for label, run in runs.items():
        names = robot_names(run)
        sim = morphology_similarity(distance_path, names)
        iu = np.triu_indices(len(names), 1)
        for ax, block in zip(axes, BLOCKS):
            m = final_snapshot_mean(run, robot_cos_key(run, block))
            if m is None:
                continue
            rho, pval = spearmanr(sim[iu], m[iu])
            rows.append({"run": label, "block": block, "spearman_rho": rho, "p_value": pval,
                         "r_squared": pearsonr(sim[iu], m[iu])[0] ** 2})
            ax.scatter(sim[iu], m[iu], s=6, alpha=0.6, label=f"{label} (rho={rho:.2f})")
            ax.set_title(block)
            ax.set_xlabel("morphological similarity")
    axes[0].set_ylabel("robot-level gradient cosine")
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "morphology_vs_conflict.pdf"))
    plt.close(fig)
    write_csv(os.path.join(out, "morphology_vs_conflict.csv"), rows)


def persistence(runs, out):
    rows = []
    for label, run in runs.items():
        for block in BLOCKS:
            for seed, snaps in enumerate(run["snaps"]):
                key = robot_cos_key(run, block)
                mats = [(int(s["step"]), s[key]) for s in snaps if key in s]
                if len(mats) < 2:
                    continue
                iu = np.triu_indices(mats[0][1].shape[0], 1)
                consecutive = [spearmanr(a[iu], b[iu])[0] for (_, a), (_, b) in zip(mats[:-1], mats[1:])]
                rows.append({"run": label, "seed": seed, "block": block,
                             "consecutive_rho_mean": float(np.mean(consecutive)),
                             "first_last_rho": float(spearmanr(mats[0][1][iu], mats[-1][1][iu])[0]),
                             "first_step": mats[0][0], "last_step": mats[-1][0]})
    write_csv(os.path.join(out, "persistence.csv"), rows)


def plot_usage(runs, out):
    for label, run in runs.items():
        names = robot_names(run)
        for block in BLOCKS:
            u = final_snapshot_mean(run, f"{block}__robot_usage")
            if u is None or u.shape[1] < 2:
                continue
            fig, ax = plt.subplots(figsize=(2 + 0.4 * u.shape[1], 5))
            im = ax.imshow(u, cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=6)
            ax.set_xlabel("expert")
            ax.set_title(f"{label}: {block}", fontsize=8)
            fig.colorbar(im)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"usage_{slug(label)}_{block}.pdf"))
            plt.close(fig)


def plot_factors(runs, out, bins=12):
    for label, run in runs.items():
        snaps = next((s for s in run["snaps"] if s), None)
        if snaps is None:
            continue
        s = snaps[-1]
        for block in BLOCKS:
            if f"{block}__pair_input_cos" not in s:
                continue
            keep = s[f"{block}__pair_keep"].astype(bool)
            x_cos = s[f"{block}__pair_input_cos"][keep]
            d_cos = s[f"{block}__pair_delta_cos"][keep]            # (N, K)
            g_cos = s[f"{block}__pair_cos"][keep]                  # (N, K)
            kappa = float(s.get(f"scalar__morse.{block}.kappa", 0.0))
            x_rep = np.repeat(x_cos[:, None], d_cos.shape[1], axis=1).ravel()
            d, g = d_cos.ravel(), g_cos.ravel()
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            hb = axes[0].hexbin(x_rep, d, C=g, reduce_C_function=np.mean, gridsize=30, cmap="RdBu", vmin=-0.5, vmax=0.5)
            axes[0].set_xlabel("input overlap cos(x_u, x_v)")
            axes[0].set_ylabel("demand agreement cos(delta_u, delta_v)")
            fig.colorbar(hb, ax=axes[0], label="mean alignment c_uv")
            edges = np.linspace(x_rep.min(), x_rep.max(), bins + 1)
            idx = np.clip(np.digitize(x_rep, edges) - 1, 0, bins - 1)
            frac = [np.mean(g[idx == b] < -kappa) if np.any(idx == b) else np.nan for b in range(bins)]
            axes[1].plot(0.5 * (edges[1:] + edges[:-1]), frac, marker="o")
            axes[1].set_xlabel("input overlap cos(x_u, x_v)")
            axes[1].set_ylabel("fraction conflicting")
            fig.suptitle(f"{label}: {block}", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"factors_{slug(label)}_{block}.pdf"))
            plt.close(fig)


# ----------------------------------------------------------------------------- tables
def returns_table(runs, out, baseline, last, metric):
    names = list(ROBOTS)
    per_run = {}
    for label, run in runs.items():
        per_seed = np.array([[eval_value(tb, metric, r, last) for r in names] for tb in run["tb"]])
        per_run[label] = per_seed
    rows = []
    for label, per_seed in per_run.items():
        mean = col_nanmean(per_seed)
        if not np.any(np.isfinite(mean)):
            continue                                      # no evaluation logged for this run
        row = {"run": label, "seeds": per_seed.shape[0]}
        ups = [last_eval_update(tb, metric) for tb in runs[label]["tb"]]
        row["last_eval_update"] = f"{np.nanmin(ups):.0f}" if np.any(np.isfinite(ups)) else "n/a"
        row.update({r: f"{m:.2f}" for r, m in zip(names, mean)})
        row["average"] = fmt([nanmean(s) for s in per_seed])
        for cls, members in CLASSES.items():
            cols = [names.index(r) for r in members]
            row[f"avg_{cls}"] = fmt([nanmean(s[cols]) for s in per_seed])
        row["worst_robot"] = fmt([np.min(s[np.isfinite(s)]) if np.any(np.isfinite(s)) else np.nan for s in per_seed])
        if baseline in per_run and label != baseline:
            gain = mean - col_nanmean(per_run[baseline])
            row["n_robots_improved"] = int(np.sum(gain > 0))
        rows.append(row)
    if not rows:
        print("skip returns table: no evaluation logged")
        return per_run
    write_csv(os.path.join(out, "returns.csv"), rows)
    cols = ["run", "average", "avg_quadruped", "avg_biped", "avg_hexapod", "worst_robot"]
    with open(os.path.join(out, "returns.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(["Method", "All", "Quadrupeds", "Bipeds", "Hexapod", "Worst robot"]) + " \\\\\n\\midrule\n")
        for row in rows:
            f.write(" & ".join(str(row[c]).replace("+-", "$\\pm$") for c in cols) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return per_run


def robot_conflict_by_name(run, block):
    """Per-robot conflict (see robot_conflict) in the order of ROBOTS; NaN for robots not trained."""
    m = final_snapshot_mean(run, robot_cos_key(run, block))
    if m is None:
        return None
    values = dict(zip(robot_names(run), robot_conflict(m)))
    return np.array([values.get(r, np.nan) for r in ROBOTS])


def conflict_vs_gain(runs, out, baseline, per_run_returns, block="decoder"):
    if baseline not in runs:
        return
    base_conf = robot_conflict_by_name(runs[baseline], block)
    if base_conf is None:
        return
    for label, run in runs.items():
        if label == baseline:
            continue
        conf = robot_conflict_by_name(run, block)
        if conf is None:
            continue
        reduction = base_conf - conf
        gain = col_nanmean(per_run_returns[label]) - col_nanmean(per_run_returns[baseline])
        ok = np.isfinite(gain) & np.isfinite(reduction)
        r, p = pearsonr(reduction[ok], gain[ok]) if ok.sum() > 2 else (np.nan, np.nan)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.scatter(reduction, gain)
        for name, x, y in zip(ROBOTS, reduction, gain):
            ax.annotate(name, (x, y), fontsize=5)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(f"conflict reduction vs {baseline} ({block})")
        ax.set_ylabel(f"return gain vs {baseline}")
        ax.set_title(f"{label}: r={r:.2f} (p={p:.3f})", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"conflict_vs_gain_{slug(label)}.pdf"))
        plt.close(fig)


def robot_conflict(cos):
    """Mean negative robot-level cosine of each robot with the others."""
    m = cos.copy()
    np.fill_diagonal(m, np.nan)
    return np.nanmean(np.maximum(-m, 0.0), axis=1)


def cost_table(runs, out):
    """Training cost. samples_per_s is the median throughput over the second half of training
    (steady state; the median ignores evaluation steps). diag_update_s is the time of a policy
    update on diagnostic steps, which always include the Gram computation; the first one,
    which includes compilation, is excluded."""
    rows = []
    for label, run in runs.items():
        sps = [np.median(tb["time/sps"][1][len(tb["time/sps"][1]) // 2:]) for tb in run["tb"] if "time/sps" in tb]
        rows.append({"run": label,
                     "samples_per_s": fmt(sps),
                     "diag_update_s": fmt([np.mean(tb["morse/update_s"][1][1:]) for tb in run["tb"]
                                           if "morse/update_s" in tb and len(tb["morse/update_s"][1]) > 1])})
    write_csv(os.path.join(out, "cost.csv"), rows)


SUMMARY_METRICS = ("robot_cancelled_fraction_acc", "robot_neg_frac_acc", "robot_cos_mean_acc",
                   "robot_cancelled_fraction", "cancelled_fraction", "destructive_share", "conflict_frac",
                   "conflict_mass", "conf_coroute", "cos_std", "kappa", "router_entropy", "eff_experts",
                   "usage_max", "usage_kl", "conflict_frac_within_transition", "conflict_frac_within_robot",
                   "conflict_frac_cross_robot", "type_conflict_frac", "loss_conf")


def diagnostics_summary(runs, out, last):
    """Final value (mean of the last diagnostic points, then mean +- s.e. over seeds) per run, block, metric."""
    rows = []
    for label, run in runs.items():
        for block in LEVELS:
            row = {"run": label, "block": block}
            for metric in SUMMARY_METRICS:
                vals = [final_value(tb, f"morse/{block}/{metric}", last) for tb in run["tb"]]
                if np.any(np.isfinite(vals)):
                    row[metric] = fmt(vals, digits=3)
            if len(row) > 2:
                rows.append(row)
        for tag in ("morse/router_grad_task", "morse/router_grad_aux"):
            vals = [final_value(tb, tag, last) for tb in run["tb"]]
            if np.any(np.isfinite(vals)):
                rows.append({"run": label, "block": "all", tag.split("/")[1]: fmt(vals, digits=6)})
    write_csv(os.path.join(out, "diagnostics_summary.csv"), rows)


# ------------------------------------------------------ motivation: joints vs robots
def eg_robot_groups(distance_path, names, n_groups):
    """Robot groups of Embodiment Grouping: Ward clustering of FGW distances (as utils.create_robot_groups)."""
    from scipy.cluster.hierarchy import cut_tree, linkage
    from scipy.spatial.distance import squareform
    if n_groups <= 1:
        return np.zeros(len(names), dtype=int)
    if n_groups >= len(names):
        return np.arange(len(names))
    d = np.load(distance_path)
    order = [ROBOTS.index(n) for n in names]
    d = d[np.ix_(order, order)].copy()
    np.fill_diagonal(d, 0.0)
    return cut_tree(linkage(squareform(0.5 * (d + d.T), checks=False), method="ward"), n_clusters=n_groups).ravel()


def type_groups_by_features(features, n_groups):
    """Ward clustering of unit types by their standardized morphology features (the router's morphology input)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    if n_groups <= 1:
        return np.zeros(len(features), dtype=int)
    f = (features - features.mean(0)) / (features.std(0) + 1e-8)
    return fcluster(linkage(f, method="ward"), t=n_groups, criterion="maxclust") - 1


def type_groups_by_alignment(cos, n_groups):
    """Oracle: average-linkage clustering of unit types by gradient alignment (distance 1 - cos)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if n_groups <= 1:
        return np.zeros(len(cos), dtype=int)
    d = np.clip(1.0 - 0.5 * (cos + cos.T), 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    return fcluster(linkage(squareform(d, checks=False), method="average"), t=n_groups, criterion="maxclust") - 1


def kept_within_groups(gram, groups):
    """Shares of the destructive and constructive pairwise interaction that stay inside groups."""
    iu = np.triu_indices(len(groups), 1)
    g, same = gram[iu], groups[iu[0]] == groups[iu[1]]
    neg, pos = np.maximum(-g, 0.0), np.maximum(g, 0.0)
    return neg[same].sum() / max(neg.sum(), 1e-30), pos[same].sum() / max(pos.sum(), 1e-30)


def joint_motivation(runs, out, distance_path, blocks=("decoder", "joint_encoder"), group_counts=(1, 2, 4, 8, 16)):
    """Are gradient conflicts organized by robots or by joints?

    Uses the contributions of unit types (joint e of robot r) accumulated over updates, summed over
    experts (best from a URMA run, i.e. K=1). Writes
      joint_pairs.csv       frequency and share of destructive / constructive interaction for joint pairs
                            within a robot, across robots of one EG group, and across EG groups;
      joint_grouping.csv/_<run>_<block>.pdf   interaction kept inside groups when robots are grouped (EG)
                            versus when joint types are grouped (by morphology features, or by gradients as an
                            oracle), for the same number of groups;
      joint_cos_<run>_<block>.pdf   joint-type alignment for three robots.
    Rows carry the estimator: "independent pairs" (unit pairs from different transitions, runs with
    unbiased_types) and "same batch" (all pairs; includes the noise covariance of joints sharing a transition).
    """
    if not os.path.exists(distance_path):
        print(f"skip joint motivation: {distance_path} not found")
        return
    pair_rows, curve_rows = [], []
    for label, run in runs.items():
        snaps = next((s for s in run["snaps"] if s), None)
        if snaps is None:
            continue
        s = snaps[-1]
        names = [canonical(r) for r in s["robot_names"]]
        # Runs with unbiased_types save sums over independent unit pairs (type_gram_acc) and the same-batch
        # sums (type_gram_acc_samebatch); older runs saved the same-batch sums only, as type_gram_acc.
        estimators = ((("independent pairs", "type_gram_acc"), ("same batch", "type_gram_acc_samebatch"))
                      if bool(s.get("unbiased_types", False)) else (("same batch", "type_gram_acc"),))
        for block, (estimator, key) in ((b, e) for b in blocks for e in estimators):
            if f"{block}__{key}" not in s:
                continue
            main = estimator == estimators[0][0]          # figures for the run's own estimator only
            gram = s[f"{block}__{key}"].sum(axis=0)
            gram = 0.5 * (gram + gram.T)
            ok = np.diag(gram) > 1e-12 * np.diag(gram).max()
            gram = gram[np.ix_(ok, ok)]
            t_robot, t_elem = s[f"{block}__type_robot"][ok], s[f"{block}__type_elem"][ok]
            features = s[f"{block}__type_morph"][ok]
            norm = np.sqrt(np.diag(gram))
            cos = gram / np.outer(norm, norm)
            iu = np.triu_indices(len(t_robot), 1)
            eg = eg_robot_groups(distance_path, names, 4)
            same_robot = t_robot[iu[0]] == t_robot[iu[1]]
            same_eg = eg[t_robot[iu[0]]] == eg[t_robot[iu[1]]]
            g, c = gram[iu], cos[iu]
            neg, pos = np.maximum(-g, 0.0), np.maximum(g, 0.0)
            for name, mask in (("within a robot", same_robot), ("across robots in one EG group", same_eg & ~same_robot),
                               ("across EG groups", ~same_eg)):
                pair_rows.append({"run": label, "block": block, "estimator": estimator, "pairs": name,
                                  "n_pairs": int(mask.sum()),
                                  "frac_negative": float((c[mask] < 0).mean()) if mask.any() else np.nan,
                                  "mean_cos": float(c[mask].mean()) if mask.any() else np.nan,
                                  "share_of_destructive": float(neg[mask].sum() / max(neg.sum(), 1e-30)),
                                  "share_of_constructive": float(pos[mask].sum() / max(pos.sum(), 1e-30))})
            groupings = {"robots: EG (FGW morphology)": lambda m: eg_robot_groups(distance_path, names, m)[t_robot],
                         "joints: morphology features": lambda m: type_groups_by_features(features, m),
                         "joints: gradients (oracle)": lambda m: type_groups_by_alignment(cos, m)}
            curves = {k: [] for k in groupings}
            for m in group_counts:
                for gname, make in groupings.items():
                    kept_neg, kept_pos = kept_within_groups(gram, make(m))
                    curves[gname].append((kept_neg, kept_pos))
                    curve_rows.append({"run": label, "block": block, "estimator": estimator, "groups": m, "grouping": gname,
                                       "destructive_kept": float(kept_neg), "constructive_kept": float(kept_pos)})
            if not main:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            for gname, vals in curves.items():
                axes[0].plot(group_counts, [v[0] for v in vals], marker="o", label=gname)
                axes[1].plot(group_counts, [v[1] for v in vals], marker="o", label=gname)
            for ax, title in zip(axes, ("destructive interaction kept inside groups", "constructive interaction kept inside groups")):
                ax.set_xscale("log", base=2)
                ax.set_xlabel("number of groups")
                ax.set_title(title, fontsize=9)
            axes[0].legend(fontsize=7)
            fig.suptitle(f"{label}: {block}", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"joint_grouping_{slug(label)}_{block}.pdf"))
            plt.close(fig)
            # Joint-type alignment for a quadruped, a biped, and the hexapod (if present).
            chosen = [r for r in ("unitree_a1", "unitree_h1", "hexapod") if r in names] or names[:3]
            idx = [i for r in chosen for i in np.where(t_robot == names.index(r))[0]]
            if idx:
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(cos[np.ix_(idx, idx)], cmap="RdBu", vmin=-1, vmax=1)
                edges = np.cumsum([np.sum(t_robot == names.index(r)) for r in chosen])[:-1] - 0.5
                for e in edges:
                    ax.axhline(e, color="k", lw=0.8)
                    ax.axvline(e, color="k", lw=0.8)
                labels = [f"{SHORT_NAMES.get(names[t_robot[i]], names[t_robot[i]])}:{t_elem[i]}" for i in idx]
                ax.set_xticks(range(len(idx)))
                ax.set_yticks(range(len(idx)))
                ax.set_xticklabels(labels, rotation=90, fontsize=5)
                ax.set_yticklabels(labels, fontsize=5)
                ax.set_title(f"{label}: joint-type gradient alignment ({block})", fontsize=8)
                fig.colorbar(im)
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"joint_cos_{slug(label)}_{block}.pdf"))
                plt.close(fig)
    write_csv(os.path.join(out, "joint_pairs.csv"), pair_rows)
    write_csv(os.path.join(out, "joint_grouping.csv"), curve_rows)


def interaction_summary(runs, out, blocks=BLOCKS):
    """Is sharing less conflicting, or only smaller? From the accumulated type Gram summed over experts (the
    Gram of the whole block: experts' parameters are disjoint), with S the self terms and P, N the constructive
    and destructive cross terms, ||sum_a g_a||^2 = S + P - N. Writes interaction.csv for every snapshot:
    harmful_per_self N/S, helpful_per_self P/S, harmful_per_helpful N/P (does routing keep helpful sharing and
    remove harmful sharing, or remove both?), and cancelled_share N/(S+P) (same-batch sums, where the identity holds)."""
    rows = []
    for label, run in runs.items():
        for seed, snaps in enumerate(run["snaps"]):
            for s in snaps:
                step = int(s["step"])
                if step == 0:
                    continue
                estimators = ((("independent pairs", "type_gram_acc"), ("same batch", "type_gram_acc_samebatch"))
                              if bool(s.get("unbiased_types", False)) else (("same batch", "type_gram_acc"),))
                for block in blocks:
                    for estimator, key in estimators:
                        if f"{block}__{key}" not in s:
                            continue
                        g = s[f"{block}__{key}"].sum(axis=0)
                        g = 0.5 * (g + g.T)
                        off = g - np.diag(np.diag(g))
                        self_, pos, neg = max(np.trace(g), 1e-30), np.maximum(off, 0.0).sum(), np.maximum(-off, 0.0).sum()
                        rows.append({"run": label, "seed": seed, "block": block, "estimator": estimator, "step": step,
                                     "harmful_per_self": neg / self_, "helpful_per_self": pos / self_,
                                     "harmful_per_helpful": neg / max(pos, 1e-30),
                                     # ||sum_a g_a||^2 = S + P - N holds for the same-batch sums only
                                     "cancelled_share": neg / (self_ + pos) if estimator == "same batch" else np.nan})
    write_csv(os.path.join(out, "interaction.csv"), rows)


def plot_type_cos(runs, out):
    """Type-level alignment (element of a robot, or robot) accumulated over updates; type_evidence runs only."""
    for label, run in runs.items():
        for block in BLOCKS:
            m = final_snapshot_mean(run, f"{block}__type_cos")
            if m is None:
                continue
            for k in range(m.shape[0]):
                full = m[k] + m[k].T
                np.fill_diagonal(full, 1.0)
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(full, cmap="RdBu", vmin=-1, vmax=1)
                ax.set_title(f"{label}: type-level alignment, {block}, expert {k}", fontsize=8)
                ax.set_xlabel("unit type")
                ax.set_ylabel("unit type")
                fig.colorbar(im)
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"type_cos_{slug(label)}_{block}_e{k}.pdf"))
                plt.close(fig)


# ---------------------------------------------------------------------------- helpers
def sem(a):
    a = np.asarray(a, dtype=float)
    return np.nanstd(a, axis=0, ddof=1) / np.sqrt(a.shape[0]) if a.shape[0] > 1 else np.zeros(a.shape[1:])


def fmt(values, digits=2):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a"
    if v.size == 1:
        return f"{v.mean():.{digits}f}"
    return f"{v.mean():.{digits}f}+-{v.std(ddof=1) / np.sqrt(v.size):.{digits}f}"


def slug(label):
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="LABEL=DIR[,DIR...] (glob patterns allowed)")
    parser.add_argument("--baseline", default=None, help="label of the reference run for gains")
    parser.add_argument("--out", default="analysis_out")
    parser.add_argument("--last", type=int, default=10, help="number of final evaluations averaged per run")
    parser.add_argument("--metric", default="episode_return", help="eval metric: episode_return or track_perf_perc")
    parser.add_argument("--upto", type=int, default=None, help="use only what was logged up to this many updates")
    parser.add_argument("--distance", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "meta", "embodiment_distance.npy"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = {}
    for spec in args.run:
        label, dirs = spec.split("=", 1)
        dirs = expand_dirs(dirs)
        if not dirs:
            raise SystemExit(f"no run directories found for {label}")
        runs[label] = {"dirs": dirs, "tb": [load_tb(d, args.upto) for d in dirs],
                       "snaps": [load_snapshots(d, args.upto) for d in dirs]}
        print(f"{label}: {len(dirs)} run(s), {sum(len(s) for s in runs[label]['snaps'])} snapshot(s)")
        if args.upto is not None:
            last = np.min([last_eval_update(tb, args.metric) for tb in runs[label]["tb"]])
            if not last >= args.upto - 100:
                print(f"warning: {label} is evaluated only up to {last} of {args.upto} updates")

    plot_curves(runs, args.out)
    plot_pair_categories(runs, args.out, args.last)
    plot_robot_matrices(runs, args.out)
    morphology_vs_conflict(runs, args.out, args.distance)
    persistence(runs, args.out)
    plot_usage(runs, args.out)
    plot_factors(runs, args.out)
    plot_type_cos(runs, args.out)
    joint_motivation(runs, args.out, args.distance)
    interaction_summary(runs, args.out)
    per_run_returns = returns_table(runs, args.out, args.baseline, args.last, args.metric)
    conflict_vs_gain(runs, args.out, args.baseline, per_run_returns)
    diagnostics_summary(runs, args.out, args.last)
    cost_table(runs, args.out)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
