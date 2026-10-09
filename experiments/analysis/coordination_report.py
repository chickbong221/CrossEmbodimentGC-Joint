"""Tables and figures of the coordination tests (analysis/coordination_tests.py) for two checkpoints A and B.

Exact-step returns come from the deterministic evaluation of the saved checkpoints (morse_ckpt_eval, <out>_eval.json);
nearby evaluation windows (mean of the last ``--window`` logged evaluations up to the checkpoint's update, and the
share of full-length episodes among them) from the runs' TensorBoard caches (tb_scalars.npz, analysis/tb_cache.py).

Usage: python analysis/coordination_report.py TESTS_JSON EVAL_JSON TB_A TB_B --updates 60000,90000 [--pairs DIR]
       [--focus cas,g1] [--out report.md]
"""
import argparse
import json
import os

import numpy as np

ORDER = ["a1", "go1", "go2", "anyb", "anyc", "bkv0", "bkvb", "bg", "bit", "h1", "g1", "tl", "op3", "nao5", "cas", "hx"]


def window(tb, robot, upd, n):
    s = (tb[f"s::eval/episode_return/{robot}"] // 16384).astype(int)
    v, ln = tb[f"v::eval/episode_return/{robot}"].astype(float), tb[f"v::eval/episode_length/{robot}"].astype(float)
    sel = np.flatnonzero(s <= upd)[-n:]
    return float(v[sel].mean()), float(np.mean(ln[sel] >= 999)), [int(s[sel[0]]), int(s[sel[-1]])]


def f(x, d=3, pct=False):
    if x is None:
        return "-"
    return f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}"


def ci(c, d=3, pct=False):
    return f"[{f(c[0], d, pct)}, {f(c[1], d, pct)}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tests")
    p.add_argument("eval")
    p.add_argument("tb_a")
    p.add_argument("tb_b")
    p.add_argument("--updates", default="60000,90000")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--pairs", default="")
    p.add_argument("--focus", default="cas,g1")
    p.add_argument("--out", default="coordination_report.md")
    a = p.parse_args()
    res = json.load(open(a.tests))
    ev = json.load(open(a.eval))
    paths = list(ev)
    tbs = [np.load(a.tb_a), np.load(a.tb_b)]
    upds = [int(u) for u in a.updates.split(",")]
    robots = [r for r in ORDER if r in res] + [r for r in res if r not in ORDER]
    focus = [r for r in a.focus.split(",") if r in res]
    L = []

    L.append("## Main table (test states of the reference episodes)\n")
    L.append("Exact-step return: deterministic evaluation of the saved checkpoint (two repeats; the first is shown). "
             f"Window: mean return of the last {a.window} logged evaluations up to the checkpoint's update, and the "
             "share of them that reach the full 1000 steps. MSE: action mean vs dataset action, mean over joints. "
             "Full / diagonal exceedance: share of test states whose D^2 exceeds the 95th percentile of the "
             "calibration reference actions in the same context (reference actions themselves: about 5%).\n")
    L.append("| Robot | Ckpt | Exact-step return (length) | Window mean (full-length share) | MSE | Full D² p95 | Full exceed. % | Diag. exceed. % |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in robots:
        t2 = res[r].get("test2")
        if t2 is None:
            continue
        for k, (key, path, tb, upd) in enumerate(zip("AB", paths, tbs, upds)):
            e = ev[path][0]
            exact = f"{e[f'eval/episode_return/{r}']:.1f} ({int(e[f'eval/episode_length/{r}'])})"
            wm, ws, rng = window(tb, r, upd, a.window)
            L.append(f"| {r if k == 0 else ''} | {upd // 1000}k | {exact} | {wm:.1f} ({100 * ws:.0f}%) | {f(t2[f'mse_{key}'], 4)} | "
                     f"{f(t2[f'full_p95_{key}'], 2)} | {f(t2[f'exceed_full_{key}'], 2, True)} | {f(t2[f'exceed_diag_{key}'], 2, True)} |")
    L.append("")
    L.append("### Change B - A with 95% paired bootstrap intervals over test episodes\n")
    L.append("| Robot | Test episodes | ΔMSE | Δ full median | Δ full p95 | Δ full exceed. (pp) | Δ diag. exceed. (pp) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in robots:
        t2 = res[r].get("test2")
        if t2 is None:
            continue
        L.append(f"| {r} | {t2['episodes']} | {f(t2['mse_diff'], 5)} {ci(t2['mse_diff_ci'], 5)} | "
                 f"{f(t2['full_median_diff'], 3)} {ci(t2['full_median_diff_ci'])} | {f(t2['full_p95_diff'], 2)} {ci(t2['full_p95_diff_ci'], 2)} | "
                 f"{f(t2['exceed_full_diff'], 2, True)} {ci(t2['exceed_full_diff_ci'], 2, True)} | "
                 f"{f(t2['exceed_diag_diff'], 2, True)} {ci(t2['exceed_diag_diff_ci'], 2, True)} |")
    L.append("")

    L.append("## Test 1: reference structure (held-out reference episodes)\n")
    L.append("NLL gain = NLL(diagonal) - NLL(full) on the test actions, nats per transition (positive: the full "
             "covariance predicts better), 95% bootstrap interval over test episodes. Shuffle: mean D^2_full of real "
             "test actions vs actions with every joint permuted independently within its context (AUC = P(real < "
             "shuffled)). Stability: similarity (Pearson r of the off-diagonal entries) of the correlation matrices of "
             "two disjoint halves of the fit episodes, minimum over contexts. Noise share: var of step differences / "
             "(2 var) of the reference actions, median over joints.\n")
    L.append("| Robot | Ref. episodes (fit/cal/test) | Contexts | Min. fit episodes per context | α, ε | NLL full / diag | NLL gain | Shuffle D² real / shuffled (AUC) | Stability (halves, partial) | Noise share |")
    L.append("|---|---|---:|---:|---|---|---|---|---|---:|")
    for r in robots:
        x = res[r]
        if "test1" not in x:
            L.append(f"| {r} | {x['episodes']['reference']} | - | - | - | - | - | - | - | - |")
            continue
        t1, sp, rg = x["test1"], x["split_episodes"], x["regularization"]
        sh = t1["shuffle"]
        stab = t1["stability"].values()
        hs = min(s["halves_corr_similarity"] for s in stab if s["halves_corr_similarity"] is not None)
        ps = min(s["halves_partial_corr_similarity"] for s in stab if s["halves_partial_corr_similarity"] is not None)
        min_eps = min(c["fit"]["episodes"] for c in x["contexts"].values())
        L.append(f"| {r} | {x['episodes']['reference']} ({sp['fit']}/{sp['calibration']}/{sp['test']}) | {len(x['contexts'])} | {min_eps} | "
                 f"{rg['alpha']}, {rg['eps_full']:g} | {f(t1['test_nll_full'], 2)} / {f(t1['test_nll_diag'], 2)} | "
                 f"{f(t1['gain_nats_per_transition'], 2)} {ci(t1['gain_ci'], 2)} | {f(sh['mean_d2_full_real'], 1)} / {f(sh['mean_d2_full_shuffled'], 1)} "
                 f"({f(sh['auc_real_below_shuffled'], 2)}) | {f(hs, 3)}, {f(ps, 3)} | {f(x['action_noise']['white_share_median'], 2)} |")
    L.append("")
    for r in focus:
        x = res[r]
        L.append(f"### {r}: contexts\n")
        L.append("| Context | Fit transitions / episodes (≥10 transitions) | Test transitions / episodes | Max. episode share | NLL gain | Mean abs. corr. | Top pair (fit / test corr.) |")
        L.append("|---|---|---|---:|---|---:|---|")
        for c, s in x["contexts"].items():
            b, st = x["test1"]["by_context"][c], x["test1"]["stability"][c]
            tp = st["top_pairs"][0]
            L.append(f"| {c} | {s['fit']['transitions']} / {s['fit']['episodes']} ({s['fit']['episodes_ge10']}) | "
                     f"{s['test']['transitions']} / {s['test']['episodes']} | {f(s['fit']['max_episode_share'], 3)} | "
                     f"{f(b['gain_nats_per_transition'], 2)} {ci(b['gain_ci'], 2)} | {f(b['mean_abs_corr_fit'], 2)} | "
                     f"{tp['pair'][0]} / {tp['pair'][1]} ({f(tp['corr_fit'], 2)} / {f(tp['corr_test'], 2)}) |")
        L.append("")
        L.append(f"### {r}: Test 2 by context\n")
        L.append("| Context | MSE A → B | Full median A → B | Full exceed. % A → B (Δ pp, CI) | Diag. exceed. % A → B (Δ pp, CI) |")
        L.append("|---|---|---|---|---|")
        for c, t in x["test2"]["by_context"].items():
            L.append(f"| {c} | {f(t['mse_A'], 4)} → {f(t['mse_B'], 4)} | {f(t['full_median_A'], 2)} → {f(t['full_median_B'], 2)} | "
                     f"{f(t['exceed_full_A'], 2, True)} → {f(t['exceed_full_B'], 2, True)} ({f(t['exceed_full_diff'], 2, True)}, {ci(t['exceed_full_diff_ci'], 2, True)}) | "
                     f"{f(t['exceed_diag_A'], 2, True)} → {f(t['exceed_diag_B'], 2, True)} ({f(t['exceed_diag_diff'], 2, True)}, {ci(t['exceed_diag_diff_ci'], 2, True)}) |")
        L.append("")

    L.append("## Test 3A: drift A → B, size vs coordination cost\n")
    L.append("Ratio = (δ'Σ⁻¹δ / δ'diag(Σ)⁻¹δ) relative to the mean of an isotropic drift; 1 = no preference, < 1: the "
             "drift follows the reference correlations, > 1: it runs against them. Null: drifts of random direction in "
             "the same contexts. Spearman: rank correlation of drift size and coordination cost over test states.\n")
    L.append("| Robot | Drift size per joint, median (p95) | Spearman size vs cost | Ratio median / p99 (null) | States with ratio > 2 (null) | MSE improved: share with D²_full up / D²_diag up |")
    L.append("|---|---|---:|---|---|---|")
    for r in robots:
        t3 = res[r].get("test3a")
        if t3 is None:
            continue
        L.append(f"| {r} | {f(t3['amp_per_joint_median'], 4)} ({f(t3['amp_per_joint_p95'], 4)}) | {f(t3['spearman_amp_cost'], 2)} | "
                 f"{f(t3['ratio_median'], 2)} / {f(t3['ratio_p99'], 2)} ({f(t3['null_ratio_median'], 2)} / {f(t3['null_ratio_p99'], 2)}) | "
                 f"{f(t3['frac_ratio_gt2'], 2, True)}% ({f(t3['null_frac_ratio_gt2'], 2, True)}%) | "
                 f"{f(t3['mse_improved_dD2_full_up'], 0, True)}% / {f(t3['mse_improved_dD2_diag_up'], 0, True)}% |")
    L.append("")
    for r in focus:
        L.append(f"### {r}: by drift size\n")
        L.append("| Drift size per joint | States | Mean ΔD²_full | Mean ΔD²_diag | Ratio median / p90 | Share ratio > 2 |")
        L.append("|---|---:|---:|---:|---|---:|")
        for b in res[r]["test3a"]["by_amplitude"]:
            lo, hi = b["amp_per_joint"]
            L.append(f"| {lo:g}–{'∞' if hi is None else f'{hi:g}'} | {b['n']} | {f(b['mean_dD2_full'], 3)} | {f(b['mean_dD2_diag'], 3)} | "
                     f"{f(b['ratio_median'], 2)} / {f(b['ratio_p90'], 2)} | {f(b['frac_ratio_gt2'], 2, True)}% |")
        L.append("")

    L.append("## Test 3B: largest relationship change (predefined ranking)\n")
    L.append("Context: largest increase of mean D²_full from A to B. Pairs: largest increase of the 2-D coordination "
             "excess (2-D Mahalanobis distance minus the two 1-D terms, reference 2x2 covariance) of the action means, "
             "95% bootstrap interval over test episodes. These are changed statistical relationships, not causes of falls.\n")
    for r in robots:
        t3b = res[r].get("test3b")
        if t3b is None or (focus and r not in focus):
            continue
        inc = t3b["context_increase_mean_dD2_full"]
        L.append(f"**{r}**: contexts by mean ΔD²_full: " + ", ".join(f"{c} {v:+.3f}" for c, v in inc.items()) + "\n")
        L.append("| Pair | Reference corr. | Mean excess A → B | Δ excess (CI) |")
        L.append("|---|---:|---|---|")
        for q in t3b["top_pairs"]:
            L.append(f"| {q['pair'][0]} / {q['pair'][1]} | {f(q['ref_corr'], 2)} | {f(q['mean_excess_A'], 3)} → {f(q['mean_excess_B'], 3)} | "
                     f"{f(q['d_excess'], 3)} {ci(q['d_excess_ci'])} |")
        L.append("")
    text = "\n".join(L)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)

    if a.pairs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(focus), figsize=(5.2 * len(focus), 4.6))
        axes = np.atleast_1d(axes)
        for ax, r in zip(axes, focus):
            z = np.load(os.path.join(a.pairs, f"pair_{r}.npz"))
            ax.scatter(z["ref"][:, 0], z["ref"][:, 1], s=3, c="0.75", label="held-out reference actions", rasterized=True)
            ax.scatter(z["A"][:, 0], z["A"][:, 1], s=3, c="tab:blue", alpha=0.5, label=f"{upds[0] // 1000}k means", rasterized=True)
            ax.scatter(z["B"][:, 0], z["B"][:, 1], s=3, c="tab:orange", alpha=0.5, label=f"{upds[1] // 1000}k means", rasterized=True)
            w, v = np.linalg.eigh(z["cov"])
            th = np.linspace(0, 2 * np.pi, 200)
            ell = (v @ (np.sqrt(w * 5.991)[:, None] * np.stack([np.cos(th), np.sin(th)]))).T + z["mean"]
            ax.plot(ell[:, 0], ell[:, 1], "k--", lw=1, label="reference 95% ellipse")
            ax.set_xlabel(str(z["pair"][0])), ax.set_ylabel(str(z["pair"][1]))
            ax.set_title(f"{r}: {z['context']}")
        axes[0].legend(fontsize=7, markerscale=3, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.splitext(a.out)[0] + "_pairs.png", dpi=130)


if __name__ == "__main__":
    main()
